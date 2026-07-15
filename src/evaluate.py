import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from train import forward_prop, get_predictions, gradient_descent


def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    return get_predictions(A3)


def test_prediction(index, W1, b1, W2, b2, W3, b3, X):
    current_image = X[:, index, None]

    prediction = make_predictions(
        current_image,
        W1, b1,
        W2, b2,
        W3, b3
    )

    print("Prediction:", prediction[0])

    image = current_image.reshape(28, 28) * 255

    plt.gray()
    plt.imshow(image, interpolation="nearest")
    plt.show()


def train(X_train, Y_train):
    return gradient_descent(X_train, Y_train, 0.10, 500)


def load_train_data():
    data = pd.read_csv("data/train.csv")
    data = np.array(data)
    np.random.shuffle(data)

    m, n = data.shape

    data_train = data[1000:m].T

    Y_train = data_train[0]
    X_train = data_train[1:n] / 255.0

    return X_train, Y_train


def load_test_data():
    data = pd.read_csv("data/test.csv")
    data = np.array(data)

    X_test = data.T / 255.0

    return X_test


def main():
    X_train, Y_train = load_train_data()
    W1, b1, W2, b2, W3, b3 = train(X_train, Y_train)

    X_test = load_test_data()
    index = np.random.randint(0, X_test.shape[1])

    test_prediction(
        index,
        W1, b1,
        W2, b2,
        W3, b3,
        X_test
    )


if __name__ == "__main__":
    main()