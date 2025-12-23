'''
Perceptron implementation using NumPy to simulate a logical AND Gate.
'''
import numpy as np

class Perceptron:
    '''
    Perceptron implementation
    '''
    def __init__(self, learning_rate=0.1, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def activation(self, x):
        return 1 if x >= 0 else 0

    def train(self, X, y):
        # Initialize weights to zeros
        self.weights = np.zeros(X.shape[1])
        
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):      
                # Calculate Weighted Sum
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self.activation(linear_output)
                
                # Update weights based on error (Perceptron Learning Rule)
                error = y[i] - prediction
                #print(f"{epoch=} {i=} {X[i]=} {self.weights=} {self.bias=} {prediction=} {error=}")
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(i) for i in linear_output])


if __name__ == "__main__":
    # Data for AND Gate
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 0, 0, 1])

    # Training
    model = Perceptron()
    #print(f"{X=} {y=}")
    model.train(X, y)

    # Testing
    print(f"Predictions: {model.predict(X)}") # Should output [0, 0, 0, 1]
