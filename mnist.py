import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import predict, train
from softmax import Softmax

def preprocess_data(x, y, limit):
    t = ()
    for i in range(9):
        l = list()
        l.append(np.where(y==i)[0][:limit])
        t += tuple(l)

    # zero_index = np.where(y==0)[0][:limit]
    # one_index = np.where(y==1)[0][:limit]
    # two_index = np.where(y==2)[0][:limit]
    # all_indices = np.hstack((zero_index, one_index, two_index))

    all_indices = np.hstack(t)

    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 9, 1)
    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 9),
    Softmax()
]
epochs = 5
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs,
    learning_rate = 0.01
)
percent_error = 0
counter = 0
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    percent_error += 1 if(np.argmax(output) == np.argmax(y)) else 0
    counter += 1
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
print(f"{round(percent_error/counter * 100, 0)}% correct")

