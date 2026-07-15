import numpy as np
import pandas as pd
import pickle

def init_params():
    W1 = np.random.randn(64, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((64, 1))

    W2 = np.random.randn(10, 64) * np.sqrt(2 / 64)
    b2 = np.zeros((10, 1))

    W3 = np.random.randn(10, 10) * np.sqrt(2 / 10)
    b3 = np.zeros((10, 1))

    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)  
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)
    
def forward_prop(W1, b1, W2, b2, W3, b3, X):

    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)

    Z2 = W2 @ A1 + b2
    A2 = ReLU(Z2)

    Z3 = W3 @ A2 + b3
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, Z3, A3,
                  W1, W2, W3, X, Y):

    m = X.shape[1]
    one_hot_Y = one_hot(Y)

    # Output layer
    dZ3 = A3 - one_hot_Y
    dW3 = (1/m) * dZ3 @ A2.T
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

    # Hidden layer 2 (10)
    dZ2 = (W3.T @ dZ3) * ReLU_deriv(Z2)
    dW2 = (1/m) * dZ2 @ A1.T
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    # Hidden layer 1 (64)
    dZ1 = (W2.T @ dZ2) * ReLU_deriv(Z1)
    dW1 = (1/m) * dZ1 @ X.T
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

def update_params(
    W1, b1,
    W2, b2,
    W3, b3,
    dW1, db1,
    dW2, db2,
    dW3, db3,
    alpha):

    W1 -= alpha * dW1
    b1 -= alpha * db1

    W2 -= alpha * dW2
    b2 -= alpha * db2

    W3 -= alpha * dW3
    b3 -= alpha * db3

    return W1, b1, W2, b2, W3, b3

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def save_model(W1, b1, W2, b2, W3, b3, path):
    model = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_train_data():
    data = pd.read_csv("data/train.csv")
    data = np.array(data)
    np.random.shuffle(data)

    m, n = data.shape

    data_train = data[1000:m].T

    Y_train = data_train[0]
    X_train = data_train[1:n] / 255.0

    return X_train, Y_train

def train(X, Y, alpha, iterations):

    W1, b1, W2, b2, W3, b3 = init_params()

    for i in range(iterations):

        Z1, A1, Z2, A2, Z3, A3 = forward_prop(
            W1, b1, W2, b2, W3, b3, X
        )

        dW1, db1, dW2, db2, dW3, db3 = backward_prop(
            Z1, A1, Z2, A2, Z3, A3,
            W1, W2, W3,
            X, Y
        )

        W1, b1, W2, b2, W3, b3 = update_params(
            W1, b1, W2, b2, W3, b3,
            dW1, db1,
            dW2, db2,
            dW3, db3,
            alpha
        )

        if i % 10 == 0:
            predictions = np.argmax(A3, axis=0)
            print(i, get_accuracy(predictions, Y))

    return W1, b1, W2, b2, W3, b3

def main():
    X_train, Y_train = load_train_data()
    W1, b1, W2, b2, W3, b3 = train(X_train, Y_train, 0.10, 500)
    print("Model trained!")

    path = "data/model.pickle"
    save_model(W1, b1, W2, b2, W3, b3, path)
    print(f"Model saved at: {path}")

if __name__ == "__main__":
    main()