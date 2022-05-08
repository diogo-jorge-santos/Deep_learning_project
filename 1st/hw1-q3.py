#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            # W_old = np.copy(self.W)
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        # Predict value
        y_i_hat = self.predict(x_i)

        # If the predicted value is not correct
        if y_i_hat != y_i:
            # Increase weight of incorrect class
            self.W[y_i, :] = self.W[y_i, :] + y_i * x_i
            # Decrease weight of incorrect class
            self.W[y_i_hat, :] = self.W[y_i_hat, :] - y_i * x_i

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        # encode y_i in one-hot
        n_classes = self.W.shape[0]
        correct_label_one_hot = np.zeros((n_classes, 1))
        correct_label_one_hot[y_i, 0] = 1
        # Predict probabilities
        scores = np.dot(self.W, x_i.T)
        probabilities = (np.exp(scores) / np.sum(np.exp(scores)))[:, None]
        # Adjust weights
        self.W = self.W - learning_rate * (probabilities - correct_label_one_hot) * x_i[None, :]


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, layers):
        # Initialize an MLP with a single hidden layer.
        self.W1 = np.random.normal(0.1, 0.05, (hidden_size, n_features))
        self.W2 = np.random.normal(0.1, 0.05, (n_classes, hidden_size))
        self.B1 = np.zeros((hidden_size))[:, None] # None # UNUSED, because data already has a bias feature   # np.zeros((hidden_size)) 
        self.B2 = np.zeros((n_classes))[:, None]
        self.n_classes = n_classes

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        #_, output, _z = self.forward(X.T)
        z_1 = np.add(self.W1.dot(X.T), self.B1)
        h_1 = np.maximum(z_1, 0)
        z_2 = np.add(self.W2.dot(h_1), self.B2)
        # Softmax: probability of each class
        probabilies = MLP.calc_probabilites(z_2)

        # Returns the index of the highest probability
        return np.argmax(probabilies, axis=0)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        for x_i, y_i in zip(X, y):
            #foward to compute the constants to do the back propagation
            z_1 = np.add(self.W1.dot(x_i[:, None]), self.B1)
            h_1 = np.maximum(z_1, 0)
            z_2 = np.add(self.W2.dot(h_1), self.B2)

            correct_label_one_hot = np.zeros((self.n_classes, 1))
            correct_label_one_hot[y_i, 0] = 1

            # Softmax: probability of each class
            probabilies = MLP.calc_probabilites(z_2)

            #backpropagation

            #grad of loss
            grad_z2 = probabilies - correct_label_one_hot

            #grad of weigth of 2nd layer

            grad_W2 = grad_z2 * h_1.T
            
            #grad of weigth of 2nd layer
            grad_B2 = grad_z2

            #grad of activation of the 1st layer
            
            grad_h1 = self.W2.T @ grad_z2
            

            #grad of z of 1st layer (({z>0} is derivative of relu activation function))
            grad_z1 = grad_h1 * (z_1>0).astype(int)

            #grad of weigths of 1st layer
            grad_W1 = np.outer(grad_z1,x_i)
            
            #grad of bias of 1st layer
            grad_B1 = grad_z1

            #sgd update using the computed gradients
            
            self.W2 = self.W2 - learning_rate*grad_W2
            self.B2 = self.B2 - learning_rate*grad_B2
            
            self.W1 = self.W1 - learning_rate*grad_W1
            self.B1 = self.B1 - learning_rate*grad_B1

    def calc_probabilites(output):
        output_max = np.max(output, axis=0)
        return np.exp(output - output_max) / np.sum(np.exp(output - output_max))

    def relu(x):
        return x * (x > 0)


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    """train_X = np.concatenate(np.ones((train_X.shape[0], 1)), train_X)
    dev_X = np.concatenate(np.ones(dev_X.shape[0], 1), dev_X)
    test_X = np.concatenate(np.ones(test_X.shape[0], 1), test_X)"""

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )

        val = model.evaluate(dev_X, dev_y)
        tes = model.evaluate(test_X, test_y)

        print("Train accuracy", model.evaluate(train_X, train_y))
        print("Validation accuracy", val)
        print("Test accuracy", tes)


        valid_accs.append(val)
        test_accs.append(tes)

    # plot
    plot(epochs, valid_accs, test_accs)
    input()


if __name__ == '__main__':
    main()
