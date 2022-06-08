import numpy as np

from LinearRegression import *


def confusionMatrix(y_true: np.ndarray, y_pred: np.ndarray):
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            truePos += 1
        elif y_true[i] == y_pred[i] == 0:
            trueNeg += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            falseNeg += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            falsePos += 1
    return truePos, trueNeg, falsePos, falseNeg


def precision(y_true: np.ndarray, y_pred: np.ndarray):
    truePos, trueNeg, falsePos, falseNeg = confusionMatrix(y_true, y_pred)
    return truePos / (truePos + falsePos)


def recall(y_true: np.ndarray, y_pred: np.ndarray):
    truePos, trueNeg, falsePos, falseNeg = confusionMatrix(y_true, y_pred)
    return truePos / (truePos + falseNeg)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    truePos, trueNeg, falsePos, falseNeg = confusionMatrix(y_true, y_pred)
    return (truePos + trueNeg) / len(y_true)


def fScore(y_true: np.ndarray, y_pred: np.ndarray):
    return 2 * (recall(y_true, y_pred) * precision(y_true, y_pred)) / (recall(y_true, y_pred) + precision(y_true, y_pred))


def logisticRegressionFit(data: np.ndarray, results: np.ndarray):
    return linearRegressionFit(data, results)


def logisticRegressionPredict(x_test: np.ndarray, w: np.ndarray):
    y = linearRegressionPredict(x_test, w)

    for i in range(len(y)):
        y[i] = 1 / (1 + np.exp(-y[i]))
    y = np.round(y)
    return y


def main():
    arrData = readData()

    y = arrData[:, 10]
    x = np.delete(arrData, 10, 1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=0)
    w = logisticRegressionFit(X_train, y_train)
    y_pred = logisticRegressionPredict(X_test, w)

    print("Accuracy: " + str(accuracy(y_test, y_pred)))
    print("Recall: " + str(recall(y_test, y_pred)))
    print("Precision: " + str(precision(y_test, y_pred)))
    print("F Score: " + str(fScore(y_test, y_pred)))


if __name__ == '__main__':
    main()
