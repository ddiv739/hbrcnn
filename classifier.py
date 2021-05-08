import tensorflow as tf
import numpy as np
import os 

def gen_model(input_dims,num_classes):
    model = None
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=input_dims),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(8, activation='relu'),
        # Only output neurons. Softmax if >2 else sigmoid for binary crossent
        tf.keras.layers.Dense(num_classes, activation='softmax') if num_classes>2 else tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])

    return model

def save_model(fn,model):
    #Creates and saves to a folder
    model.save(f'./SavedModels/{fn}')
    return True

def load_model(fn,model):
    return keras.models.load_model(f'./SavedModels/{fn}')