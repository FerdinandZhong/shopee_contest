from PIL import Image
import glob, os

def resize_images():
    size = 128, 128

    for infile in glob.glob("D:\\shopee_images\\mobile_image\\mobile_image\\*.jpg"):
        file, ext = os.path.splitext(infile)
        file = file[file.rfind('\\')+1:]
        im = Image.open(infile)
        im.thumbnail(size, Image.ANTIALIAS)
        im.save(python_platform_path + '/images/mobile_image/' + file + ".jpg", "JPEG")

if __name__ == "__main__":
    python_platform_path = os.path.abspath(__file__ + "/../../")
    resize_images()