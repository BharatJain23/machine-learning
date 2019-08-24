import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
##import cv2

##def data_summary(X_train, y_train, X_test, y_test):
##    """Summarize current state of dataset"""
##    print('Train images shape:', x_train.shape)
##    print('Train labels shape:', y_train.shape)
##    print('Test images shape:', x_test.shape)
##    print('Test labels shape:', y_test.shape)
##    print('Train labels:', y_train)
##    print('Test labels:', y_test)

##print(tf.__version__)

## 28 x 28 images of hand written digits 0-9
mnist = tf.keras.datasets.mnist 

## tf.keras.datasets.mnist.load_data(path='')
## Arguments: path: path where to cache the datasets locally
## Return: Tuple of numpy arrays -> (x_train, y_train),(x_test, y_test)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#### Check state of dataset
##data_summary(x_train, y_train, x_test, y_test)

## Normalize or scale the data
## It scales the data from 0 to 1 which makes it easier for the network
## to learn, we dont have to do this
## so we can test ,later, how it affects the network.
x_train = tf.keras.utils.normalize(x_train, axis = 1 )
x_test = tf.keras.utils.normalize(x_test, axis = 1 )

## Most common
## Feed forward model (No cyles present)
model = tf.keras.models.Sequential()

## Flattening is just converting multidimensional array into 1 dimensional array
## We can use numpy, reshape or we can use built layer in keras i.e. Flatten()
model.add(tf.keras.layers.Flatten())
## Now we will add hidden layers
## The first argument is the number of neurons the layer has
## tf.nn.relu (rectified linear) is default go-to activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
## This is output layer
## The number of neurons are equal to number of classification (in cases of classification anyways)
## In this case it is 10 i.e. number from 0-9
## Since this is like a probaility distribution function, we will use softmax
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

## We will define parameters for training of the model
## The network tries to minimize the loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save('num_reader.model')
new_model = tf.keras.models.load_model('num_reader.model')

predictions = new_model.predict([x_test])

print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()

## For black and white image
##cv2.imshow('5',x_train[0])
##cv2.waitKey(0)
##cv2.destroyAllWindows()

#### Image seems colored because of RGB convention of matplotlib
#### thus using cmap = plt.cm.binary to produce black and white image
##plt.imshow(x_train[0], cmap = plt.cm.binary)
##plt.show()


