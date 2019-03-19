import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
from sklearn.model_selection import train_test_split
from scipy import sparse as sp_sparse
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, precision_score

def read_data(filename):
    data = pd.read_csv(filename)
    return data

def text_prepare(text):
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)
    text = re.sub(BAD_SYMBOLS_RE, '', text)
    new_text = ''
    for substring in text.split():
        if substring not in STOPWORDS:
            new_text += substring + ' '
    return new_text.strip()

def generate_most_popular_words_dicts(word_list):
    words_to_index = {}
    index_to_words = {}
    for index, word_popularity in enumerate(word_list):
        words_to_index[word_popularity[0]] = index
        index_to_words[index] = word_popularity[0]
    return words_to_index, index_to_words


def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary

        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] = 1
        else:
            continue
    return result_vector


if __name__ == "__main__":
    python_platform_path = os.path.abspath(__file__ + "/../../")
    data_path = Path(python_platform_path+"/data/train.csv/train.csv")
    train = read_data(data_path)
    print(train.head())
    X_train, y_train = train['title'].values, train['Category'].values
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    X_train = [text_prepare(x) for x in X_train]
    print(X_train[:3], y_train[:3])
    tags_counts = {}
    words_counts = {}
    for tag in y_train:
        if tag in tags_counts:
            tags_counts[tag] = tags_counts[tag] + 1
        else:
            tags_counts[tag] = 1
    for sentence in X_train:
        for word in sentence.split():
            if word in words_counts:
                words_counts[word] = words_counts[word] + 1
            else:
                words_counts[word] = 1
    most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    print(most_common_words)

    DICT_SIZE = 10000
    dictionaries = generate_most_popular_words_dicts(
        sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:5000])
    WORDS_TO_INDEX = dictionaries[0]
    INDEX_TO_WORDS = dictionaries[1]
    ALL_WORDS = WORDS_TO_INDEX.keys()
    print(INDEX_TO_WORDS[0], INDEX_TO_WORDS[19], INDEX_TO_WORDS[340])

    x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.3, random_state = 42)
    x_train_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in x_train])
    x_test_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in x_test])
    print('X_train shape ', x_train_mybag.shape)
    print('X_test shape ', x_test_mybag.shape)
    y_train = label_binarize(y_train, classes=sorted(tags_counts.keys()))
    y_val = label_binarize(y_test, classes=sorted(tags_counts.keys()))
    import itertools
    a = [0.1, 1]
    b = ['l1', 'l2']
    parameters = list(itertools.product(a, b))
    print(parameters)
    for C_value, penalty_value in parameters:
        print(C_value, penalty_value)
        clf = OneVsRestClassifier(LogisticRegression(penalty=penalty_value, C=C_value))
        clf.fit(x_train_mybag, y_train)
        y_val_predicted_labels_mybag = clf.predict_proba(x_test_mybag)
        y_val_labels = [[tag for tag in list(enumerate(tags)) if tag[1] == 1][0][0] for tags in y_val]
        print(y_val_labels[:10])
        y_val_predicted_labels_mybag = [sorted(list(enumerate(tags)), key=lambda x: x[1], reverse=True)[0][0] for tags in y_val_predicted_labels_mybag]
        print(y_val_predicted_labels_mybag[:10])
        print("Result with parameter: C: {}, penalty: {}".format(C_value, penalty_value))
        print('F1 score weighted: {}'.format(f1_score(y_val_labels, y_val_predicted_labels_mybag, average='micro')))