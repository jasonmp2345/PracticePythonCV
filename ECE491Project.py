import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from random import randint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

import requests
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

images_path_fire = 'C:/pythonProjectsVS/fire/fire'
images_path_nofire = 'C:/pythonProjectsVS/fire/nofire'

labels_df_nofire = np.zeros(3363, dtype=int)
labels_df_fire = np.ones(6567, dtype=int)

class_names = ('fire', 'nofire')
num_classes = len(class_names)

img_size = (128, 128, 3)

print(f'{num_classes} classes: {class_names}\nimage size: {img_size}')


images_fire = []
images_nofire = []
max_height = 0
max_width = 0

# Load in the images
for filepath in os.listdir(images_path_fire):
    if filepath.endswith(".jpg") or filepath.endswith(".png"):
        image = Image.open(os.path.join(images_path_fire, filepath))
        image_array = np.asarray(image)
        height, width = image_array.shape[:2]
        print('here')
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width
        images_fire.append(image_array)

max_height2 = 0
max_width2 = 0
for filepath2 in os.listdir(images_path_nofire):
    if filepath2.endswith(".jpg") or filepath2.endswith(".png"):
        image = Image.open(os.path.join(images_path_nofire, filepath2))
        image_array = np.asarray(image)
        height, width = image_array.shape[:2]
        print('here2')
        if height > max_height2:
            max_height2 = height
        if width > max_width2:
            max_width2 = width
        images_nofire.append(image_array)

padded_image_list = []
for image_array in images_fire:
    pad_height = max_height - image_array.shape[0]
    pad_width = max_width - image_array.shape[1]
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    padded_image = np.pad(image_array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    padded_image_list.append(padded_image)

padded_image_list2 = []
for image_array in images_nofire:
    pad_height = max_height - image_array.shape[0]
    pad_width = max_width - image_array.shape[1]
    if (pad_width < 0):
        continue
    
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    padded_image = np.pad(image_array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    padded_image_list2.append(padded_image)

labels_df_fire = np.asarray(labels_df_fire)
labels_df_nofire = np.asarray(labels_df_nofire)
images_fire_list = np.asarray(padded_image_list)
images_nofire_list = np.asarray(padded_image_list2)

print(f'\nlabels shape: {labels_df_fire.shape}')
print(f'images shape: {images_fire_list.shape}')
print(f'images2 shape: {images_nofire_list.shape}')

# Combine the data and labels along the last axis
data_with_labels1 = np.concatenate((images_fire_list, labels_df_fire), axis=-1)
data_with_labels2 = np.concatenate((images_nofire_list, labels_df_nofire), axis=-1)
data_with_labels = zip(data_with_labels1, data_with_labels2)
data_with_labels = np.asarray(data_with_labels)

# Shuffle the data and labels randomly
np.random.shuffle(data_with_labels)

# Compute the split index
split_idx = len(data_with_labels) // 10

# Split the data and labels into two parts
train_data, train_labels = data_with_labels[split_idx:, :-1], data_with_labels[split_idx:, -1]
test_data, test_labels = data_with_labels[:split_idx, :-1], data_with_labels[:split_idx, -1]

#Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(331, 331, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on your data
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))
