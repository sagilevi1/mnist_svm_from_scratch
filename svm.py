import numpy as np

class SVM:
    #create svm class
    def __init__(self, learning_rate=0.0001, lambda_param=0.01, n_iters=10000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialized the wights and b to zero (can also set to random)
        self.w = np.zeros(n_features)
        self.b = 0

        #gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                #linear hperplane - constraints state
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    #the gradient of the cost function if we out the margin
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # the gradient of the cost function if we inside the margin
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y[idx])
                    )
                    self.b -= self.lr * y[idx]

    #after calulating the final whights can make a prediction
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)





