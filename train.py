import numpy as np
import tensorflow as tf
import cv2
from cv2 import imwrite

from cnn_util import train_valid_test_data, train, predict
from util import plot_features

def main():
    print('Loading data...')
    train_images, train_coords, valid_images, valid_coords, test_images = \
            train_valid_test_data(
            train_data_file='data/training.csv',
            test_data_file='data/test.csv',
            validation_split=0.8)
    test_image = test_images[np.random.randint(test_images.shape[0])]
    print('Data loaded. Training...')
    with tf.Session() as sess:
        train(sess, train_images, train_coords, valid_images, valid_coords,
              batch_size=512,
              n_epochs=10,
              meta_graph='my_models/my_model_small.ckpt.meta',
              new=False,
              silent=False,
              save_model=False,
              save_file_name='my_model.ckpt')
        print('Finished training. Testing it on an image.')
        coords = predict(sess, test_image)
    test_image = plot_features(test_image, coords[0], color_BGR=[0,255,0], thickness=1)
    test_image_path = 'test_facial_feature.jpg'
    imwrite(test_image_path,255*test_image)
    print('Test image saved as ' + test_image_path)

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
