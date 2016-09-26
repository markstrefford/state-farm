"""
Train the model
"""

import config
import numpy as np
import pandas as pd
from keras.utils import np_utils
import cv2
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Flatten, Activation, Dropout, MaxPooling1D
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback
import matplotlib.pyplot as plt

def split_drivers_into_train_and_validate(driver_list, split = 0.95):
    print "split_drivers_into_train_and_validate(): "
    driver_valid_list = []
    # Take a random sample of drivers into the training list
    driver_train_list = np.random.choice(driver_list, int(len(driver_list ) *split), replace = False)
    # Take the remaining drivers into the validation list
    driver_valid_list = [ driver for driver in driver_list if driver not in driver_train_list]
    print "Driver train list: {}".format(driver_train_list)
    print "Driver validation list: {}".format(driver_valid_list)
    return driver_train_list, driver_valid_list

def create_train_validation_data(driver_list, filter):
    # sample = driver_list[driver_list.subject.isin(filter)].ix[:, 'classname':'img']
    images = []
    labels = []
    total = 0
    for driver_row in [drvr for drvr in driver_list[driver_list.subject.isin(filter)].ix[:, 'classname':'img'].iterrows()]:
        driver = driver_row[1]  # Drop the index created by the Pandas Dataframe
        # print driver
        label = int(driver['classname'][1:])
        filename = config.train_images_dir + str(label) + "/" + driver['img']
        if config.color_type == 1:
            image = cv2.imread(filename,
                               0).transpose()  # Is the color_type needed here as these are pre-processed images??
        elif config.color_type == 3:
            image = cv2.imread(
                filename).transpose()  # Is the color_type needed here as these are pre-processed images??
        images.append(image)
        labels.append(label)
        total += 1
        if total % 1000 == 0:
            print "Processed {} samples".format(total)
    print "Processed {} samples...Done!\n".format(total)

    # TODO - Is this sufficient normalisation??
    images = np.array(images, dtype=np.uint8)
    images = images.reshape(images.shape[0], config.color_type, config.image_width, config.image_height)
    images = images.astype('float32')
    images /= 255

    labels = np.array(labels, dtype=np.uint8)
    labels = np_utils.to_categorical(labels, 10)

    return images, labels


def custom_keras_model(num_classes, weights_path=None):
    num_filters = 8  # number of filters to apply/learn in the 1D convolutional layer
    num_pooling = 2
    filter_length = 2  # linear length of each filter (this is 1D)

    # Create callback for history report
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))

    # from keras.utils.dot_utils import Grapher

    model = Sequential()

    # Now create the NN architecture (version 1)
    # Going with grayscale for now!!
    model.add(Convolution2D(num_filters, filter_length, filter_length, border_mode="valid",
                            activation="relu",
                            input_shape=(config.color_type, config.image_width, config.image_height)))

    model.add(Convolution2D(num_filters, filter_length, filter_length))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(num_pooling, num_pooling)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    if weights_path:
        print "Loading weights from {}".format(weights_path)
        model.load_weights(weights_path)

    model.summary()
    # grapher.plot(model, 'nn_model.png')

    # TODO - Handle loading existing weights

    return model, LossHistory

# Show loss history over training
def graph_training_loss_history(losses):
    plt.figure(figsize=(6, 3))
    plt.plot(losses)
    plt.ylabel('error')
    plt.xlabel('batch')
    plt.title('training error')
    plt.show()

# Split the driver list into training and validation sets
driver_list = pd.read_csv(config.driver_image_list)
print "Sample training data: {}".format(driver_list.head())
driver_ids = config.get_driver_ids(driver_list)
print "Driver ids in training set: {}".format(driver_ids)
training_list, validation_list = split_drivers_into_train_and_validate(driver_ids)

# Now create training and validation data sets
#
# index = np.random.choice(range(0, num_training_samples), num_training_samples, replace = False) # Random ordering
# ...driver_list[index], training_list[index]
print "Creating training data:"
X_train, y_train = create_train_validation_data(driver_list, training_list)
print "Creating validation data:"
X_valid, y_valid = create_train_validation_data(driver_list, validation_list)

# Print some stats about the data
print "Training data shape: {}".format(X_train.shape)
num_training_samples = X_train.shape[0]
print "Validation data shape: {}".format(X_valid.shape)
num_validation_samples = X_valid.shape[0]

# Create the model
keras_model, weights, train_model = 'custom', None, True
#keras_model, weights, train_model = 'vgg16', 'model/vgg16_weights.h5', False
loss_function='categorical_crossentropy'
num_classes = 10

if keras_model == 'custom':
    model, LossHistory = custom_keras_model(num_classes, weights)
    sgd = SGD(lr=0.1, decay=0, momentum=0, nesterov=False)
# elif keras_model == 'vgg16':
#     model, LossHistory = vgg16(num_classes, weights)
#     sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

# Now compile the model
model.compile(loss=loss_function, optimizer=sgd)

# Show the model
#from IPython.display import SVG
#from keras.utils.visualize_util import model_to_dot
#SVG(model_to_dot(model).create(prog='dot', format='svg'))

# Now train the model
batch_size = 32
num_epochs = 5

history = LossHistory()
#index = np.random.choice(range(0, num_training_samples), num_training_samples, replace = False) # Random ordering
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=num_epochs,
              show_accuracy=True, verbose=1, validation_data=(X_valid, y_valid),
              callbacks=[history])

graph_training_loss_history(history.losses)

# Save weights
model.save_weights('./model/saved_weights.h5')
