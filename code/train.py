# Author: Mon Cedrick G. Frias
# Date: May 11, 2018
# Filename: train.py
# Code Reference: This program creates a classification model using Convolutional Neural Network (CNN)

# Installing libraries
# (Theano)      pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# (Tensorflow)  pip install tensorflow
# (Keras)       pip install --upgrade keras

# Part 1 - Building the CNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam

#Image Dimensions
img_width, img_height = 64, 64

# Step 1 - Initialising the CNN
model = Sequential()

# Step 2 - First Convolutional Layer + Pooling
input_shape = (img_width, img_height, 3)
model.add(Conv2D(32, (3, 3), input_shape = input_shape, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Add more layers
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 4 - Flattening
model.add(Flatten())

# Step 5 - Full connection
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Step 6 - Compiling the CNN
optimizer = 'adam'
metrics=['accuracy']
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = metrics)


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

# Batch size
bs = 16

# Epochs                         
epochs = 25

# Perform Image Augmentation to training data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   fill_mode = 'nearest',
                                   horizontal_flip = True)

# Perform Image Augmentation to test data
test_datagen = ImageDataGenerator(rescale = 1./255)

# Generate augmented training data set
training_set = train_datagen.flow_from_directory('../dataset/training_set',
                                                 target_size = (img_width, img_height),
                                                 batch_size = bs,
                                                 class_mode = 'binary')

# Generate augmented test data set
test_set = test_datagen.flow_from_directory('../dataset/test_set',
                                            target_size = (img_width, img_height),
                                            batch_size = bs,
                                            class_mode = 'binary')

# Load pretrained weights from previously created models
model.load_weights('../models/coconuts_old.h5')

# Train the model
model.fit_generator(training_set,
                    steps_per_epoch = 576 / bs,
                    epochs = epochs,
                    validation_data = test_set,
                    validation_steps = 144 / bs)

# Save the model for future uses
model.save('../models/coconuts_pretrained_15.h5')
del model
