"""
Dense network from scratch for MNIST digit classification.
This implementation uses numpy and does not rely on high-level libraries like TensorFlow or Keras.
"""
import numpy as np

class DenseNetworkScratch:
    def __init__(self, input_shape=(28, 28), num_classes=10):
        self.input_shape = input_shape  # (28, 28)
        self.input_size = np.prod(input_shape) # 784 input neurons
        self.num_classes = num_classes
        self.weights = {
            'hidden1': np.random.randn(self.input_size, 128) * 0.01,
            'hidden2': np.random.randn(128, 64) * 0.01,
            'hidden3': np.random.randn(64, 32) * 0.01,
            'output': np.random.randn(32, num_classes) * 0.01
        }
        self.biases = {
            'hidden1': np.zeros((1, 128)),
            'hidden2': np.zeros((1, 64)),
            'hidden3': np.zeros((1, 32)),
            'output': np.zeros((1, num_classes))
        }
    
    def loss(self, y_true, y_pred):
        # Sparse categorical crossentropy loss
        return np.mean(-np.log(y_pred[np.arange(len(y_true)), y_true] + 1e-12))

    def activation(self, x):
        # ReLU activation function
        return np.maximum(0, x)
    
    def softmax(self, x):
        #take in output array for a single sample, return softmax probabilities
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) #subtract max for stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forwardPropagate(self, input_data):
        #Takes in some number of data points each w 784 neurons 
        input_data = input_data.reshape(-1, self.input_size)
        if input_data.shape[1] != self.input_size:
            raise ValueError(f"Input data shape mismatch. Expected {self.input_size} features, got {input_data.shape[1]}.")
        """
        forward cache stores:
            { "a_prev_hidden1": input_data, <-(n_samples, 784)
            "z_hidden1": z1,
            "a_prev_hidden2": a1,
            "z_hidden2": z2,
            "a_prev_hidden3": a2,
            "z_hidden3": z3,
            "a_prev_output": a3,
            "z_output": z_out}
        """

        forward_cache = {} 
        hidden_layers = ['hidden1', 'hidden2', 'hidden3']
        a = input_data

        for layer in hidden_layers:
            z = np.dot(a, self.weights[layer]) + self.biases[layer] #z = pre-activation
            forward_cache[f"z_{layer}"] = z
            forward_cache[f"a_prev_{layer}"] = a
            a = self.activation(z) #a = post-activation 
        
        output_z = np.dot(a, self.weights['output']) + self.biases['output']
        y_pred = self.softmax(output_z) #probabilities that output is some given number
        forward_cache["z_output"] = output_z
        forward_cache["a_prev_output"] = a
        return y_pred, forward_cache 
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def backpropagate(self, y_pred, y_true, forward_cache):
        grads = {}
        m = y_true.shape[0] #batch size
        
        if y_true.ndim == 1:  # sparse labels
            y_true_onehot = np.zeros_like(y_pred)
            y_true_onehot[np.arange(m), y_true] = 1
        else:
            y_true_onehot = y_true
        
        a_prev = forward_cache["a_prev_output"]
        
        dz = y_pred - y_true_onehot 
        grads["dW_output"] = (a_prev.T @ dz) / m
        grads["db_output"] = np.mean(dz, axis=0, keepdims=True)
        
        dz = dz @ self.weights["output"].T
        for i in range(3,0,-1):
            layer = f'hidden{i}'
            z = forward_cache[f"z_{layer}"]
            a_prev = forward_cache[f"a_prev_{layer}"]
            
            dz = dz * self.relu_derivative(z)
            grads[f"dW_{layer}"] = (a_prev.T @ dz) / m
            grads[f"db_{layer}"] = np.mean(dz, axis=0, keepdims=True)
            if i > 1: #dont prop to last hidden layer, could be done better
                dz = dz @ self.weights[layer].T

        return grads

    def update_weights(self, grads, learning_rate = 0.1):
        for layer in self.weights:
            self.weights[layer] -= learning_rate * grads[f"dW_{layer}"]
            self.biases[layer] -= learning_rate * grads[f"db_{layer}"]
        return None
    """
    Instead of updating weights after a single forward prop we run it on 32 images at once
    calculate average loss, then update once.
    """
    def sgd_fit(self, X_train, y_train, batch_size=32): #one minibatch, one epoch is when all the data is passed through
        indices = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
        minibatch_X = X_train[indices]
        minibatch_y = y_train[indices]

        y_pred, forward_cache = self.forwardPropagate(minibatch_X)
        print(self.loss(minibatch_y, y_pred))  # for testing
        grads = self.backpropagate(y_pred, minibatch_y, forward_cache)
        self.update_weights(grads)

    def evaluate(self, X_test, y_test):
            y_pred, _ = self.forwardPropagate(X_test)
            predicted_classes = np.argmax(y_pred, axis=1)
            accuracy = np.mean(predicted_classes == y_test)
            print(f"Test accuracy: {accuracy:.4f}")
            return accuracy
    
    def fit(self, X_train, y_train, epochs=10, batch_size=32):
        loss = []
        accuracy = []
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            indices = np.arange(X_train.shape[0]) #shuffle data each epoch
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            num_batches = X_train.shape[0] // batch_size
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                minibatch_X = X_train_shuffled[start:end]
                minibatch_y = y_train_shuffled[start:end]
                
                y_pred, forward_cache = self.forwardPropagate(minibatch_X)
                grads = self.backpropagate(y_pred, minibatch_y, forward_cache)
                self.update_weights(grads)
            loss_value = self.loss(y_train, self.forwardPropagate(X_train)[0])
            loss.append(loss_value)
            accuracy_value = self.evaluate(X_train, y_train)
            accuracy.append(accuracy_value)
            print(f"Epoch {epoch+1} completed, Loss: {loss_value:.4f}")
        return loss, accuracy
    
    

    
    

