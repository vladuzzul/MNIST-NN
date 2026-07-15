import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle

from train import forward_prop, get_predictions

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

def load_test_data():
    data = pd.read_csv("data/test.csv")
    data = np.array(data)

    X_test = data.T / 255.0

    return X_test

def load_model(path):
    model = None

    with open(path, "rb") as f:
        model = pickle.load(f)
    
    if model != None:
        W1 = model["W1"]
        b1 = model["b1"]
        W2 = model["W2"]
        b2 = model["b2"]
        W3 = model["W3"]
        b3 = model["b3"]
        return W1, b1, W2, b2, W3, b3
    else:
        return None

def main():
    path = "data/model.pickle"
    W1, b1, W2, b2, W3, b3 = load_model(path)

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