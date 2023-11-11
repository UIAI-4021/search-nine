import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


class LinearRegression:
    def __init__(self, learning_rate, n_iters=1000):
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
        'Morning': 1,
        'Afternoon': 2,
        'Evening': 3,
        'Night': 4,
        'Late_Night': 5,
        'Early_Morning': 6,
    }

    stops_dict = {
        'zero': 1,
        'one': 2,
        'two_or_more': 3
    }

    arrival_time_dict = {

        'Morning': 1,
        'Afternoon': 2,
        'Evening': 3,
        'Early_Morning': 4,
        'Night': 5,
        'Late_Night': 6
    }

    class_dict = {
        'Business': 1,
        'Economy': 2
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

    linreg = LinearRegression(learning_rate=0.0001, n_iters=1000)
    linreg.fit(X_train, y_train)

    predictions = linreg.predict(X_test)
    print(f"RMSE: {rmse(y_test, predictions)}")
    print(mean_squared_error(y_test, predictions))
    print(mean_absolute_error(y_test, predictions))
    print(r2_score(y_test, predictions))