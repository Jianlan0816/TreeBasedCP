import numpy as np
from sklearn.svm import OneClassSVM
from sklearn import svm
import copy
'''
:param xtrain: ndarray, training data
:param ytrain: ndarray, training label
:param xcalibration: ndarray, calibration data
:param ycalibration: ndarray, calibration label
:param xtest: ndarray, test data
:param labels: list, unique labels with customized order for ytrain
:return pval: ndarray, p values for test data among all labels'''
def ICAD(xtrain, ytrain, xcalibration, ycalibration, xtest, labels):
    pval_compare = np.empty([xtest.shape[0], len(labels)])
    for j in range(len(labels)):
        x = xtrain[ytrain == labels[j]]
        #print("labels[j]", x.shape)
        xcal = xcalibration[ycalibration == labels[j]]
        #print("labels[j]", xcal.shape)
        clf = OneClassSVM(gamma='scale', kernel='rbf').fit(x)
        cal_score = clf.decision_function(xcal)
        x_test_score = clf.decision_function(xtest)
        for i in range(len(x_test_score)):
            k = np.sum(cal_score <= x_test_score[i])
            k = k.item()
            p = (k + 1) / (len(cal_score) + 1)
            pval_compare[i][j] = p
    return pval_compare

'''
:param xtrain1, xtrain2: ndarray, two equal folds training data
:param ytrain1, ytrain2: ndarray, two equal folds training labels
:param x_test1, x_test2: ndarray, two equal folds test data
:param labels: list, unique labels with customized order for ytrain
:return pval: ndarray, p values for test data among all labels'''
def BCOPS(x_train1, x_train2, y_train1, y_train2, x_test1, x_test2, labels):
    xtest = np.concatenate((x_test1, x_test2), axis=0)
    pval = np.empty([xtest.shape[0], len(labels)])
    for j in range(len(labels)):
        x1 = x_train1[y_train1==labels[j]]
        x2 = x_train2[y_train2==labels[j]]

        clf1 = svm.SVC(probability=True)
        train1_x = np.concatenate((x1, x_test1), axis=0)
        train1_y = ['a'] * x1.shape[0] + ['b'] * x_test1.shape[0]
        clf1.fit(train1_x, train1_y)
        clf2 = svm.SVC(probability=True)
        train2_x = np.concatenate((x2, x_test2), axis=0)
        train2_y = ['a'] * x2.shape[0] + ['b'] * x_test2.shape[0]
        clf2.fit(train2_x, train2_y)

        conf_scores1 = clf2.predict_proba(x1)
        test_scores1 = clf2.predict_proba(x_test1)

        conf_scores2 = clf1.predict_proba(x2)
        test_scores2 = clf1.predict_proba(x_test2)

        for m in range(test_scores1.shape[0]):
            k = np.sum(test_scores1[m][0] >= conf_scores1[:, 0])
            p = (k.item() + 1) / (conf_scores1.shape[0] + 1)
            pval[m][j] = p
        for n in range(test_scores2.shape[0]):
            k = np.sum(test_scores2[n][0] >= conf_scores2[:, 0])
            p = (k.item() + 1) / (conf_scores2.shape[0] + 1)
            pval[n + x_test1.shape[0]][j] = p

    return pval

'''
:param: x_train: ndarray, proper training data
:param: y_train: ndarray, proper training labels
:param: x_calibration: ndarray, calibration data
:param: y_calibration: ndarray, calibration labels
:param: x_test: ndarray, test data
:param: labels: list, unique labels with customized order for y_train
:return pval: ndarray, p value for test data among all labels'''
def new_method(x_train, y_train, x_calibration, y_calibration, x_test, labels):
    pval = np.empty([x_test.shape[0], len(labels)])
    y_train2 = copy.deepcopy(y_train)
    y_calibration2 = copy.deepcopy(y_calibration)
    # initializing lists
    keys = copy.deepcopy(labels)
    values = list(range(len(labels)))
    res = {keys[i]: values[i] for i in range(len(keys))}
    for i in range(len(y_train2)):
        y_train2[i] = res[y_train2[i]]
    for i in range(len(y_calibration2)):
        y_calibration2[i] = res[y_calibration2[i]]
    clf = svm.SVC(probability=True)
    clf.fit(x_train, y_train2)
    conf_score = clf.predict_proba(x_calibration)
    test_score = clf.predict_proba(x_test)
    for k in range(len(labels)):
        conf_s = conf_score[y_calibration2 == str(k), k]
        for p in range(test_score.shape[0]):
            p_val = (np.sum(test_score[p][k] >= conf_s) + 1) / (conf_s.shape[0] + 1)
            pval[p][k] = p_val
    return pval
