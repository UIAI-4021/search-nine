import numpy as np
import pandas as pd
import time
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


class LinearRegression:
    def __init__(self, learning_rate, n_iters=10000):
        # init parameters
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _init_params(self):
        self.weights = np.zeros(self.n_features)
        self.bias = 0

    def _update_params(self, dw, db):
        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    def _get_prediction(self, X):
        return np.dot(X, self.weights) + self.bias

    def _get_gradients(self, X, y, y_pred):
        # get distance between y_pred and y_true
        error = y_pred - y
        # compute the gradients of weight & bias
        dw = (1 / self.n_samples) * np.dot(X.T, error)
        db = (1 / self.n_samples) * np.sum(error)
        return dw, db

    def fit(self, X, y):
        # get number of samples & features
        self.n_samples, self.n_features = X.shape
        # init weights & bias
        self._init_params()

        # perform gradient descent for n iterations
        for _ in range(self.n_iters):
            # get y_prediction
            y_pred = self._get_prediction(X)
            # compute gradients
            dw, db = self._get_gradients(X, y, y_pred)
            # update weights & bias with gradients
            self._update_params(dw, db)

    def predict(self, X):
        y_pred = self._get_prediction(X)
        return y_pred


# define helper function to evaluate
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


if __name__ == '__main__':
    df = pd.read_csv('Flight_Price_Dataset_Q2.csv')

    departure_time_dict = {
        'Morning': 6,
        'Afternoon': 5,
        'Evening': 4,
        'Night': 3,
        'Late_Night': 2,
        'Early_Morning': 1,
    }

    stops_dict = {
        'zero': 3,
        'one': 2,
        'two_or_more': 1
    }

    arrival_time_dict = {

        'Morning': 6,
        'Afternoon': 5,
        'Evening': 4,
        'Early_Morning': 3,
        'Night': 2,
        'Late_Night': 1
    }

    class_dict = {
        'Business': 2,
        'Economy': 1
    }

    df['Departure_time_dict'] = df['departure_time'].map(departure_time_dict)
    df['stops_dict'] = df['stops'].map(stops_dict)
    df['arrival_time_dict'] = df['arrival_time'].map(arrival_time_dict)
    df['class_dict'] = df['class'].map(class_dict)

    df2 = df.copy()
    to_drop = ['departure_time', 'stops', 'arrival_time', 'class', 'price']
    df2.drop(to_drop, inplace=True, axis=1)
    X = df2.values

    print(X)
    y = np.array(df['price'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)

    start_time = time.time()

    linreg = LinearRegression(learning_rate=0.001, n_iters=10000)
    linreg.fit(X_train, y_train)

    end_time = time.time()

    predictions = linreg.predict(X_test)
    execution_time = end_time - start_time

    with open('nine-UIAI4021-PR1-Q2.txt', 'w', encoding='utf-8') as input_file:
        input_file.write(f"Training Time: {execution_time} s\n")
        input_file.write("logs: \n")
        input_file.write(f"MSE: {mean_squared_error(y_test, predictions)}\n")
        input_file.write(f"RMSE: {rmse(y_test, predictions)}\n")
        input_file.write(f"MAE: {mean_absolute_error(y_test, predictions)}\n")
        input_file.write(f"R2: {r2_score(y_test, predictions)}\n")

    input_file.close()








