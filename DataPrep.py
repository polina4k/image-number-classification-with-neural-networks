import matplotlib.pyplot as plt
from DataLoader import MnistDataloader
import numpy as np
from src.DenseNetwork import DenseNetwork

loader = MnistDataloader(
    training_images_filepath="data/train-images.idx3-ubyte",
    training_labels_filepath="data/train-labels.idx1-ubyte",
    test_images_filepath="data/t10k-images.idx3-ubyte",
    test_labels_filepath="data/t10k-labels.idx1-ubyte"
)

(train_images, train_labels), (test_images, test_labels) = loader.load_data()


# normalize data
train_images = np.array(train_images) / 255.0
test_images = np.array(test_images) / 255.0


# # Display a few training images
# for i in range(5):
#     plt.imshow(train_images[i], cmap='gray')
#     plt.title(f"Label: {train_labels[i]}")
#     plt.axis('off')
#     plt.show()

# Initialize and compile the Dense Network
dense_network = DenseNetwork(num_classes=10, input_shape=(28, 28))
dense_network.compile_model()
