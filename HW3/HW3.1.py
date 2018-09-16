import pandas as pd
from numpy import matrix


def main():
    Prediction = [1, 0, 1, 0, 1, 1, 1, 0, 1, 0]
    Actual = [1, 0, 1, 1, 0, 1, 1, 1, 1, 0]
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0
    for i in range(len(Prediction)):
        if (Prediction[i] == Actual[i]):
            if (Prediction[i] == 1):
                TP += 1
            else:
                TN += 1
        else:
            if (Prediction[i] == 1):
                FP += 1
            else:
                FN += 1
    ConMatrix = matrix([[TP, FP], [TN, FN]])
    Accurate = (TP + TN) / (len(Prediction))
    Recall = TP / (TP + FN)
    Fallout = FP / (FP + TN)
    print('Confusion Matrix: \n %s' % ConMatrix)
    print('Accurate = %s' % Accurate)
    print('Recall = %s' % Recall)
    print('Fallout = %s' % Fallout)


if __name__ == '__main__':
    main()
