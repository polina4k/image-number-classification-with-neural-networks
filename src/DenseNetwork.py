"""
Dense neural network for MNIST digit classification.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input

class DenseNetwork:
    def __init__(self, input_shape = (28,28), num_classes = 10):
        self.input_shape = input_shape
        self.num_classes = num_classes 
        self.model = Sequential()
        self.model.add(keras.Input(shape=self.input_shape, name="input_layer"))
        self.model.add(Flatten(name="flatten_layer"))
        self.model.add(Dense(128, activation='relu', name="hidden_layer_1"))
        self.model.add(Dense(64, activation='relu', name="hidden_layer_2"))
        self.model.add(Dense(32, activation='relu', name="hidden_layer_3"))
        self.model.add(Dense(self.num_classes, activation='softmax', name="output_layer"))

    def compile_model(self):
        self.model.compile(optimizer= keras.optimizers.SGD(learning_rate=0.01),
                           loss='sparse_categorical_crossentropy', #labels are integers not one hot encoded
                           metrics=['accuracy']) 
        print(self.model.summary())
