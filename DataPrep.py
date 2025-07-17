import matplotlib.pyplot as plt
from DataLoader import MnistDataloader
import numpy as np
from src.DenseNetwork import DenseNetwork
from src.DenseNetworkScratch import DenseNetworkScratch

loader = MnistDataloader(
    training_images_filepath="data/train-images.idx3-ubyte",
    training_labels_filepath="data/train-labels.idx1-ubyte",
    test_images_filepath="data/t10k-images.idx3-ubyte",
    test_labels_filepath="data/t10k-labels.idx1-ubyte"
)

(train_images, train_labels), (test_images, test_labels) = loader.load_data()

train_images = np.array(train_images) / 255.0
test_images = np.array(test_images) / 255.0
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# # Display a few training images
# for i in range(5):
#     plt.imshow(train_images[i], cmap='gray')
#     plt.title(f"Label: {train_labels[i]}")
#     plt.axis('off')
#     plt.show()

#Testing Networks:

#Dense neural net with TensorFlow and Keras
dense_network = DenseNetwork(num_classes=10, input_shape=(28, 28))
dense_network.compile_model()
keras_history = dense_network.fit(train_images, train_labels, epochs=5, batch_size=32)
#dense_network.prediction_showcase(test_images, test_labels, num_images=5)

# Dense network from scratch
dense_network_scratch = DenseNetworkScratch()
scratch_history = dense_network_scratch.fit(train_images, train_labels, epochs=5, batch_size=32)

# Model plot evaluation
keras_loss = keras_history.history['loss']
scratch_loss = scratch_history

plt.plot(keras_loss, label='Tensorflow Loss')
plt.plot(scratch_loss, label='Numpy model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Tensorflow vs Numpy Dense Network Loss')
plt.show()