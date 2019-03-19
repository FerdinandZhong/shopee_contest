import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from sklearn.metrics import f1_score

def poster_file(item):
    return input_path / f"{item[3]}"


def has_poster(item):
    return Path(poster_file(item)).is_file()

def get_image(item):
    with Image.open(poster_file(item)) as poster:
        poster = poster.resize((poster_width, poster_height), resample=Image.LANCZOS)
        poster = poster.convert('RGB')
        return np.asarray(poster) / 255


def unique(list):
    seen = set()
    return [e for e in list
            if not (e in seen or seen.add(e))]

def bitmap(category, uniques):
    bmp = []
    for u in range(0, len(uniques)):
        if uniques[u] == category:
            bmp.append(1.0)
        else:
            bmp.append(0.0)
    return bmp


def encode(categories, uniques):
    return [bitmap(category, uniques) for category in categories]

def load_data(path):
    csv = path / "train.csv"
    items = pd.read_csv(csv, encoding="ISO-8859-1", keep_default_na=False)
    items = items[items['image_path'].str.contains("mobile_image")]
    items = items[items.apply(lambda d: has_poster(d), axis=1)]
    items = items.sample(frac=0.2).reset_index(drop=True)
    posters = list(map(get_image, items.values))
    categories = items['Category'].apply(lambda x: x - 31).tolist()
    unique_categories = unique(categories)
    x = np.array(posters)
    y = np.array(encode(categories, unique_categories))
    print("unique categories: {}".format(unique_categories))
    print("x shape: {}, y shape: {}".format(x.shape, y.shape))
    return x, y, unique_categories

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, stride_x, stride_y):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,stride_x,stride_y,1], padding="SAME")

def compute_accuracy(v_xs, v_ys, session, prediction, xs, ys, keep_prob):
    y_pre = session.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    # -- evaluation for binary classification
    # print("pre: {}, ture: {}".format(session.run(tf.argmax(y_pre, 1)), session.run(tf.argmax(v_ys, 1))))
    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(v_ys, 1), predictions=tf.argmax(y_pre, 1))
    session.run(tf.local_variables_initializer())
    result = session.run(acc_op, feed_dict={xs: v_xs, keep_prob: 1})
    # --

    # macro f1 score for multi-label classification
    # predicted_class = session.run(tf.round(y_pre))
    # macro_f1s = []
    # for prediction, true in zip(predicted_class, v_ys):
    #     macro_f1s.append(f1_score(y_true=true, y_pred=prediction, average='macro'))
    # result = session.run(tf.reduce_mean(tf.cast(macro_f1s, tf.float32)))
    return result

def tf_training(x_train, y_train, x_validation, y_validation):

    # placeholders for giving data from outside
    xs = tf.placeholder(tf.float32)
    ys = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, (-1, poster_height, poster_width, poster_channels))

    ## conv1 layer ##
    W_conv1 = weight_variable([3, 3, 3, 16])  # patch 3x3, in size 3, out size 16
    b_conv1 = bias_variable([16])
    conv_layer1 = conv2d(x_image, W_conv1) + b_conv1
    # 非线性化处理
    h_conv1 = tf.nn.relu(conv_layer1)  # output size 64x64x16
    h_pool1 = max_pool_2x2(h_conv1, 2, 2)  # output size 32x32x16

    ## conv2 layer ##
    W_conv2 = weight_variable([5, 5, 16, 32])  # patch 5x5, in size 16, out size 32
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 32x32x32
    h_pool2 = max_pool_2x2(h_conv2, 2, 2)  # output size 16x16x32

    ## fc1 layer ##
    h_pool_dropout = tf.nn.dropout(h_pool2, 0.25)
    W_fc1 = weight_variable([16 * 16 * 32, 128])
    b_fc1 = bias_variable([128])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64] (flatten)
    h_pool3_flat = tf.reshape(h_pool_dropout, [-1, 16 * 16 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  # matmul means matrix multiply
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)  # avoid overfitting

    ## fc2 layer ##
    W_fc2 = weight_variable([128, len(unique_categories)])
    b_fc2 = bias_variable([len(unique_categories)])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=prediction))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        session.run(local_init)
        num_batches = int(x_train.shape[0] / batch_size)
        matrix_x = np.array_split(x_train, num_batches)
        matrix_y = np.array_split(y_train, num_batches)
        for i in range(epochs):
            for batch_xs, batch_ys in zip(matrix_x, matrix_y):
                # tf_training(batch_xs, batch_ys, session)
                session.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1})

            print("epochs: {0}, accuracy: {1}".format(i, compute_accuracy(x_validation, y_validation, session, prediction, xs, ys, keep_prob)))

if __name__ == "__main__":
    python_platform_path = os.path.abspath(__file__ + "/../../")
    input_path = Path(python_platform_path+"/images")
    data_path = Path(python_platform_path+"/data/train.csv")

    poster_width = 64  # 182 / 3.7916
    poster_height = 64  # 268 / 4.1875
    poster_channels = 3  # RGB

    epochs = 20
    batch_size = 100

    x_data, y_data, unique_categories = load_data(data_path)
    separator = len(x_data) * 3 // 4
    x_train = x_data[0:separator]
    y_train = y_data[0:separator]
    x_validation = x_data[separator:len(x_data)]
    y_validation = y_data[separator:len(y_data)]

    tf_training(x_train, y_train, x_validation, y_validation)

