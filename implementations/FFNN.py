import numpy as np

class FeedForwardNeuralNetwork:
    def __init__(self, input_dim, hidden_dims, output_dim, learning_rate=0.001):
        """Initialize the neural network."""
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        self.weights.append(np.random.randn(input_dim, hidden_dims[0]) * 0.01)
        self.biases.append(np.zeros((1, hidden_dims[0])))

        for i in range(1, len(hidden_dims)):
            self.weights.append(np.random.randn(hidden_dims[i - 1], hidden_dims[i]) * 0.01)
            self.biases.append(np.zeros((1, hidden_dims[i])))

        self.weights.append(np.random.randn(hidden_dims[-1], output_dim) * 0.01)
        self.biases.append(np.zeros((1, output_dim)))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward_propagation(self, X):
        A = X
        self.cache = {'A0': A}
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.relu(Z)
            self.cache[f'Z{i + 1}'] = Z
            self.cache[f'A{i + 1}'] = A

        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self.softmax(Z)
        self.cache[f'Z{len(self.weights)}'] = Z
        self.cache[f'A{len(self.weights)}'] = A
        return A

    def compute_loss(self, Y, Y_hat):
        m = Y.shape[0]
        log_probs = -np.log(Y_hat[range(m), Y])
        loss = np.sum(log_probs) / m
        return loss

    def predict(self, X):
        Y_hat = self.forward_propagation(X)
        return np.argmax(Y_hat, axis=1)

    def evaluate(self, X, Y):
        """Evaluate the model on a dataset and return the loss and accuracy."""
        Y_hat = self.forward_propagation(X)
        loss = self.compute_loss(Y, Y_hat)
        Y_pred = np.argmax(Y_hat, axis=1)
        accuracy = np.mean(Y_pred == Y)
        return loss, accuracy

    def backward_propagation(self, X, Y):
        m = X.shape[0]
        grads = {}

        A_out = self.cache[f'A{len(self.weights)}']
        A_out[range(m), Y] -= 1
        grads[f'dZ{len(self.weights)}'] = A_out / m

        for i in reversed(range(len(self.weights))):
            dZ = grads[f'dZ{i + 1}']
            A_prev = self.cache[f'A{i}'] if i > 0 else X
            grads[f'dW{i + 1}'] = np.dot(A_prev.T, dZ)
            grads[f'db{i + 1}'] = np.sum(dZ, axis=0, keepdims=True)

            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                dZ_prev = dA_prev * self.relu_derivative(self.cache[f'Z{i}'])
                grads[f'dZ{i}'] = dZ_prev

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads[f'dW{i + 1}']
            self.biases[i] -= self.learning_rate * grads[f'db{i + 1}']

    def train(self, X, Y, epochs=20, batch_size=32):
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                Y_batch = Y_shuffled[i:i + batch_size]

                Y_hat = self.forward_propagation(X_batch)
                self.backward_propagation(X_batch, Y_batch)

            Y_hat_full = self.forward_propagation(X)
            loss = self.compute_loss(Y, Y_hat_full)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
