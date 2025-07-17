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

    def fit(self, x_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return history
    
    def evaluate(self, x_test, y_test):
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test)
        print(f"Test accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy
    
    #get the predictions for the input data
    def predict(self, x):
        predictions = self.model.predict(x)
        predicted_classes = np.argmax(predictions, axis=1)
        return predicted_classes

    def prediction_showcase(self, x, y, num_images=5):
        predictions = self.predict(x)
        for i in range(num_images):
            plt.imshow(x[i], cmap='gray')
            plt.title(f"True: {y[i]}, Predicted: {predictions[i]}")
            plt.axis('off')
            plt.show()