import sys
from cv2 import CascadeClassifier, imwrite
from keras.models import load_model
import matplotlib.pyplot as plt
from util import load_image, detect_cascade, detect_facial_features, add_sunglasses

def main():
    cascade = CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    model = load_model('my_models/my_model_small.h5')

    image_BGR, image_gray = load_image('test_images/test_image_1.jpg')
    faces = detect_cascade(image_gray, cascade, scaleFactor=4)
    facialFeatures = detect_facial_features(image_gray, faces, model)
    image_BGR = add_sunglasses(image_BGR, faces, facialFeatures)

    imwrite('output.jpg',image_BGR)

def parseArgs(args):
    from optparse import OptionParser

    parser = OptionParser()

    raise notImplementedError

if __name__ == '__main__':
    """
    Function to be called when the python file is executed.
    """
    #args = parseArgs(sys.argv[1:])
    main()
