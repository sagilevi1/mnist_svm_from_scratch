import numpy as np


class SVM:
    # create svm class
    def __init__(self, learning_rate=0.0001, lambda_param=0.01, n_iter=10000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def fit(self, x, y):
        n_samples, n_features = x.shape

        # initialized the wights and b to zero (can also set to random)
        self.w = np.zeros(n_features)
        self.b = 0

        # gradient descent
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(x):
                # linear hyperplane - constraints state
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # the gradient of the cost function if we out the margin
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # the gradient of the cost function if we inside the margin
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y[idx])
                    )
                    self.b -= self.lr * y[idx]

    # predict with the final weights
    def predict(self, x):
        approx = np.dot(x, self.w) - self.b
        return np.sign(approx)





