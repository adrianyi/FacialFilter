import tensorflow as tf
import numpy as np
from pandas import read_csv

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

def load_face_data(data_fname='data/train.csv', test=False):
    '''
    Loads training.csv and test.csv datasets from Kaggle

    Load the train and test data, normalize the pizels to [0,1], and reshape
    all images to 96x96
    '''
    df = read_csv(data_fname)
    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda x: np.fromstring(x, sep=' '))
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)
    X = X.reshape(-1, 96, 96, 1) # return each images as 96 x 96 x 1

    if not test:  # only FTRAIN has target columns
        y = df[df.columns[:-1]].values
        # scale target coordinates to [-1, 1]
        y = (y - 48) / 48
        # shuffle train data
        np.random.seed(42)
        np.random.shuffle(X)
        np.random.seed(42)
        np.random.shuffle(y)
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def train_valid_test_data(
        train_data_file='data/training.csv',
        test_data_file='data/test.csv',
        validation_split=0.8):
    train_images, train_coordinates = load_face_data(train_data_file)
    test_images, _ = load_face_data(test_data_file,test=True)

    n = train_images.shape[0]
    n_train = int(validation_split*n)
    indices = np.random.permutation(n)
    train_idx, valid_idx = indices[:n_train], indices[n_train:]
    train_images, valid_images = train_images[train_idx], train_images[valid_idx]
    train_coordinates, valid_coordinates = train_coordinates[train_idx], train_coordinates[valid_idx]
    return train_images, train_coordinates, valid_images, valid_coordinates, test_images

def print_epoch_stats(tfSess, epoch_i, loss, accuracy,
                    last_images, last_coords, valid_images, valid_coords):
    """
    Print training/validation loss and accuracy of an epoch
    """
    graph = tf.get_default_graph()
    train_x = graph.get_tensor_by_name("train_x:0")
    train_y = graph.get_tensor_by_name("train_y:0")
    predictions = graph.get_tensor_by_name("predictions/BiasAdd:0")
    accuracy = tf.subtract(1.0,tf.reduce_mean(tf.minimum(tf.abs(tf.subtract(train_y, predictions)),1)))

    train_loss = tfSess.run(
        loss,
        feed_dict={train_x: last_images, train_y: last_coords})
    train_accuracy = tfSess.run(
        accuracy,
        feed_dict={train_x: last_images, train_y: last_coords})

    valid_loss = tfSess.run(
        loss,
        feed_dict={train_x: valid_images, train_y: valid_coords})
    valid_accuracy = tfSess.run(
        accuracy,
        feed_dict={train_x: valid_images, train_y: valid_coords})

    print_string = 'Epoch: {:<4} - Loss: {:<8.3}, Accuracy: {:<5.3}\n' + \
                '              Valid Loss: {:<8.3}, Valid Accuracy: {:<5.3}'
    print(print_string.format(
            epoch_i,
            train_loss, train_accuracy,
            valid_loss, valid_accuracy))

def train(tfSess, train_images, train_coords, valid_images, valid_coords,
          batch_size = 512,
          n_epochs = 50,
          meta_graph = 'my_models/my_model_small.ckpt.meta',
          new=True,
          silent=False,
          save_model=True,
          save_file_name='my_model.ckpt'):
    saver = tf.train.import_meta_graph(meta_graph)
    graph = tf.get_default_graph()
    train_x = graph.get_tensor_by_name("train_x:0")
    train_y = graph.get_tensor_by_name("train_y:0")
    predictions = graph.get_tensor_by_name("predictions/BiasAdd:0")

    loss = tf.losses.mean_squared_error(labels=train_y, predictions=predictions)
    accuracy = tf.subtract(1.0,tf.reduce_mean(tf.minimum(tf.abs(tf.subtract(train_y, predictions)),1)))
    optimizer = graph.get_tensor_by_name("RMSProp/update_predictions/bias/ApplyRMSProp:0")

    if new:
        init = tf.global_variables_initializer()
        tfSess.run(init)
    else:
        saver.restore(tfSess, tf.train.latest_checkpoint('my_models/'))
    # Training cycle
    n_train = train_images.shape[0]
    for epoch_i in range(1,n_epochs+1):
        # Generate batch indices
        batch_indices = np.split(np.random.permutation(n_train),range(0,n_train,batch_size))
        # Iterate over all the batches
        for batch_index in batch_indices:
            train_feed_dict = {
                train_x: train_images[batch_index],
                train_y: train_coords[batch_index]}
            tfSess.run(optimizer, feed_dict=train_feed_dict)
        # Print cost and validation accuracy of an epoch
        if not silent:
            print_epoch_stats(tfSess, epoch_i, loss, accuracy,
                train_images[batch_index], train_coords[batch_index],
                valid_images, valid_coords)
    # Save model
    if save_model:
        saver = tf.train.Saver()
        file_str = 'my_model/' + save_file_name
        saver.save(tfSess, file_str)
        print('Model saved in my_model/')

def predict(tfSess, image_gray_ndarray):
    image = np.reshape(image_gray_ndarray,(1,96,96,1))
    graph = tf.get_default_graph()
    train_x = graph.get_tensor_by_name("train_x:0")
    train_y = graph.get_tensor_by_name("train_y:0")
    predictions = graph.get_tensor_by_name("predictions/BiasAdd:0")
    coords = tfSess.run(predictions, feed_dict={train_x: image})
    return coords
