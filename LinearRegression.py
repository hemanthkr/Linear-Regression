import numpy as np


class LinearRegression:
    def __init__(self, alpha, iters, X, Y, B):
        self.alpha = alpha
        self.iters = iters
        self.X = X
        self.Y = Y
        self.B = B

    def costfunction(self):
        m = len(self.Y)
        J = np.sum((self.X.dot(self.B) - self.Y) **2)/(2*m)
        return J

    def gradient_descent(self):
        costList = []

        m = len(self.Y)

        for iter in range(self.iters):
            h = self.X.dot(self.B)
            loss = h - self.Y
            gradient = self.X.T.dot(loss) / m
            self.B = self.B - self.alpha * gradient
            cost = self.costfunction()
            costList.append(cost)
            if (iter % 10000 == 0) or (iter == self.iters - 1):
                print("Iter=", iter, "cost = ", cost)

        return self.B, costList

    def predict(self, X):
        Y_pred = X.dot(self.B)
        return Y_pred

    def accuracy(self, Y_pred):
        total_error = 0
        m = len(self.Y)
        for i in range(0, m):
            error = abs((Y_pred[i] - self.Y[i]) / self.Y[i])
            total_error += error
        total_error = (total_error / m)
        acc = 1 - total_error
        return acc * 100

    def rmse(self, Y, Y_pred):
        rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
        return rmse

    def r2_score(self, Y, Y_pred):
        mean_y = np.mean(Y)
        ss_tot = sum((Y - mean_y) ** 2)
        ss_res = sum((Y - Y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
