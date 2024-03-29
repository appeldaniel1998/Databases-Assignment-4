import random

import numpy as np


def splitData(x, y, trainPercent):
    random.seed(0)
    trainNum = int(np.round(len(x) * trainPercent))
    x_train = []
    y_train = []

    for i in range(trainNum):
        rand = int(round(random.random() * (len(x) - 1)))
        x_train.append(x[rand])
        np.delete(x, rand)
        y_train.append(y[rand])
        np.delete(y, rand)
    return np.array(x_train), x, np.array(y_train), y

    # x_train = x[:trainNum]
    # x_test = x[trainNum:]
    # y_train = y[:trainNum]
    # y_test = y[trainNum:]
    # return x_train, x_test, y_train, y_test


def calcMSE(y_test: np.ndarray, y_pred: np.ndarray):
    mse = 0
    for i in range(len(y_test)):
        mse += (y_test[i] - y_pred[i]) ** 2
    return mse * (1 / len(y_test))


def readData():
    my_file = open("prices.txt", "r")
    contentOfFile = my_file.read()
    lstOfLines = contentOfFile.split('\n')
    data = []
    for i in range(len(lstOfLines)):
        data.append(lstOfLines[i].split(','))

    arrData = np.zeros((len(data) - 1, len(data[0])))
    for i in range(arrData.shape[0]):
        for j in range(arrData.shape[1]):
            arrData[i][j] = float(data[i][j])
    my_file.close()
    return arrData


def linearRegressionFit(data: np.ndarray, results: np.ndarray):
    newData = np.zeros((data.shape[0], data.shape[1] + 1))
    for i in range(newData.shape[0]):
        newData[i][1:] = data[i]
        newData[i][0] = 1
    data = newData
    xTx = data.T @ data
    xTy = data.T @ results
    w = np.linalg.inv(xTx) @ xTy
    return w


def linearRegressionPredict(x_test: np.ndarray, w: np.ndarray):
    newData = np.zeros((x_test.shape[0], x_test.shape[1] + 1))
    for i in range(newData.shape[0]):  # adding one in the first index of the new 2D array
        newData[i][1:] = x_test[i]
        newData[i][0] = 1

    y = np.zeros((newData.shape[0]))  # Calculating the return y
    for i in range(len(y)):
        y[i] = np.sum(newData[i] * w)
    return y


def main():
    arrData = readData()

    # If the below function were used (from sklearn), due to the different randomizer, the MSE would be 28.9 (currently 38.7)
    # X_train, X_test, y_train, y_test = train_test_split(arrData[:, :-1], arrData[:, -1], train_size=0.75, random_state=0)

    X_train, X_test, y_train, y_test = splitData(arrData[:, :-1], arrData[:, -1], 0.75)
    w = linearRegressionFit(X_train, y_train)
    y_pred = linearRegressionPredict(X_test, w)
    print("MSE score: " + str(calcMSE(y_test, y_pred)))

    from sklearn.linear_model import LinearRegression
    linReg = LinearRegression()
    linReg.fit(X_train, y_train)
    y_pred_sk = linReg.predict(X_test)
    print("MSE score of sklearn model: " + str(calcMSE(y_test, y_pred)))


if __name__ == '__main__':
    main()
