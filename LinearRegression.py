import numpy as np
from sklearn.model_selection import train_test_split


# def split(data: np.ndarray):
#     df = pd.DataFrame(data)
#     XY_train = df.sample(frac=0.75)
#     XY_test = df.drop(XY_train.index)
#     X_train = np.array(XY_train.loc[:, XY_train.columns != 11])
#     X_test = np.array(XY_test.loc[:, XY_test.columns != 11])
#     y_train = np.array(XY_train.loc[:, XY_train.columns == 11])
#     y_test = np.array(XY_test.loc[:, XY_test.columns == 11])
#     return X_train, X_test, y_train, y_test
#
#     # trainPercent = int(x.shape[0] * 0.75)
#     # X_train, X_test, y_train, y_test = x[:trainPercent, :], x[trainPercent:, :], y[:trainPercent], y[trainPercent:]
#     # return X_train, X_test, y_train, y_test


def calcMSE(y_test: np.ndarray, y_pred: np.ndarray):
    mse = 0
    for i in range(len(y_test)):
        mse += (y_test[i] - y_pred[i]) ** 2
    return mse * (1 / len(y_test))


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
    for i in range(newData.shape[0]):
        newData[i][1:] = x_test[i]
        newData[i][0] = 1
    x_test = newData

    y = np.zeros((x_test.shape[0]))
    for i in range(len(y)):
        y[i] = np.sum(x_test[i] * w)
    return y


def main():
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

    X_train, X_test, y_train, y_test = train_test_split(arrData[:, :-1], arrData[:, -1], train_size=0.75, random_state=0)
    w = linearRegressionFit(X_train, y_train)
    y_pred = linearRegressionPredict(X_test, w)
    print(calcMSE(y_test, y_pred))

    from sklearn.linear_model import LinearRegression
    linReg = LinearRegression()
    linReg.fit(X_train, y_train)
    y_pred_sk = linReg.predict(X_test)
    print(calcMSE(y_test, y_pred_sk))



if __name__ == '__main__':
    main()
