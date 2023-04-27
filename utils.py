import random
import math
import numpy as np
from sklearn.model_selection import train_test_split
"""
:param t: newick tree
:param depth: integer representing i^th layer
:return: list of nodes in the same depth in newick format
"""
def search_by_depth(t, depth):
    matches = []
    for n1 in t.traverse():
        root = t.get_tree_root()
        d1 = root.get_distance(n1)
        if d1 == depth:
            matches.append(n1)
    return matches
"""
:param node: a newick tree node
:return: true or false if the node is a leaf
"""
def check_leaf(node):
    if node.is_leaf():
        return True
    else:
        return False
"""
:param data_x: a proper training data
:param data_y: a proper training data label
:param node: a tree node with names
:return: transformed data with one class
"""
def ICAD_transform(data_x, data_y, node):
    des = node.get_descendants(is_leaf_fn=check_leaf)
    name = list()
    name.append(node.name)
    # get name of each corresponding descendants with itself
    for descen in des:
        name.append(descen.name)
    # get train data from x_train
    index = [i for i, e in enumerate(data_y) if e in name]
    trainx = data_x[index]
    trainy = [1] * trainx.shape[0]
    return trainx, trainy


"""
:param data_x: a proper training data
:param data_y: a proper training data label
:param node: a tree node with names 
:return: transformed data with multiple classes as children numbers
"""
def NEW_transform(data_x, data_y, node):
    train_x = np.ones((0,data_x.shape[1]))
    train_y = np.array([])
    if (node is None):
        train_x = data_x
        train_y = [1] * train_x.shape[0]
    else:
        des = node.get_children()
        k = 1
        for child in des:
            x, y = ICAD_transform(data_x, data_y, child)
            train_x = np.concatenate((train_x, x), axis = 0)
            yy = [k] * x.shape[0]
            train_y = np.concatenate((train_y, yy), axis = 0)
            k = k + 1
    return train_x, train_y

"""
:param data_x: a training data set
:param data_y: a training label set
:param inlier: list of inlier labels
:param outlier: list of outlier labels
:param a: test size in inlier, default is 0.5
:param ratio: ratio of inlier test and outlier test samples, default is 1.0
:param b: calibration size in proper training set, default is 0.3
:return: data split into three, a proper training, a calibration, and a test data
"""
def ICAD_split(data_x, data_y, inlier, outlier, a=0.5, ratio = 1.0, b = 0.3):
    index1 = [i for i, e in enumerate(data_y) if e in inlier]
    x_inlier = data_x[index1]
    y_labels = data_y[index1]
    #num_list = random.sample(range(0, len(y_labels)), 600)
    #x_inlier = x_inlier[num_list]
    #y_labels = y_labels[num_list]
    x_train, x_test, y_train, y_test = train_test_split(x_inlier, y_labels,
                                                        test_size= a, random_state=52)
    c = math.floor(len(y_test)*ratio)
    index2 = [i for i,e in enumerate(data_y) if e in outlier]
    x_outlier = data_x[index2]
    y_outlier = ['outlier']*x_outlier.shape[0]
    num_list = random.sample(range(0, len(y_outlier)), c)
    x_outlier = x_outlier[num_list]
    y_outlier = ['outlier']*x_outlier.shape[0]

    x_testall = np.concatenate([x_test, x_outlier], axis=0)
    y_testall = np.concatenate([y_test, y_outlier], axis=0)

    x_trainset, x_calibration, y_trainset, y_calibration = train_test_split(x_train, y_train,
                                                                            test_size=b, random_state=52)
    return x_trainset, x_calibration, y_trainset, y_calibration, x_testall, y_testall

"""
:param data_x: a training data set
:param data_y: a training label set
:param inlier: list of inlier labels
:param outlier: list of outlier labels
:param a: test size in inliers, default is 0.5
:param ratio: ratio of inlier test to outlier test samples, default is 1.0
:return: two equal folds of training and two equal folds of test
"""
def BCOPS_split(data_x, data_y, inlier, outlier, a = 0.5, ratio = 1.0):
    index1 = [i for i, e in enumerate(data_y) if e in inlier]
    x_inlier = data_x[index1]
    y_labels = data_y[index1]
    x_train, x_test, y_train, y_test = train_test_split(x_inlier, y_labels,
                                                        test_size=a, random_state=52)
    c = math.floor(len(y_test) * ratio)
    index2 = [i for i, e in enumerate(data_y) if e in outlier]
    x_outlier = data_x[index2]
    y_outlier = ['outlier'] * x_outlier.shape[0]
    num_list = random.sample(range(0, len(y_outlier)), c)
    x_outlier = x_outlier[num_list]
    y_outlier = ['outlier'] * x_outlier.shape[0]

    x_testall = np.concatenate([x_test, x_outlier], axis=0)
    y_testall = np.concatenate([y_test, y_outlier], axis=0)

    x_test1, x_test2, y_test1, y_test2 = train_test_split(x_testall, y_testall,
                                                          test_size = 0.5, random_state=52)
    x_train1, x_train2, y_train1, y_train2 = train_test_split(x_train, y_train,
                                                              test_size = 0.5, random_state = 52)
    return x_train1, x_train2, y_train1, y_train2, x_test1, x_test2, y_test1, y_test2
"""
:param data_x: a training data set
:param data_y: a training label set
:param inlier: list of inlier labels
:param outlier: list of outlier labels
:param a: test size in inlier, default is 0.5
:param ratio: ratio of inlier test and outlier test samples, default is 1.0
:param b: calibration size in proper training set, default is 0.3
:return: training data split into two, a proper training and a calibration, test data split into two equal folds
"""
def NEW_split(data_x, data_y, inlier, outlier, a = 0.5, ratio = 1.0, b = 0.3):
    index1 = [i for i, e in enumerate(data_y) if e in inlier]
    x_inlier = data_x[index1]
    y_labels = data_y[index1]
    x_train, x_test, y_train, y_test = train_test_split(x_inlier, y_labels,
                                                        test_size = a, random_state=52)
    c = math.floor(len(y_test) * ratio)
    index2 = [i for i, e in enumerate(data_y) if e in outlier]
    x_outlier = data_x[index2]
    y_outlier = ['outlier'] * x_outlier.shape[0]
    num_list = random.sample(range(0, len(y_outlier)), c)
    x_outlier = x_outlier[num_list]
    y_outlier = ['outlier'] * x_outlier.shape[0]

    x_testall = np.concatenate([x_test, x_outlier], axis=0)
    y_testall = np.concatenate([y_test, y_outlier], axis=0)

    x_trainset, x_calibration, y_trainset, y_calibration = train_test_split(x_train, y_train,
                                                                            test_size=b, random_state=52)
    x_test1, x_test2, y_test1, y_test2 = train_test_split(x_testall, y_testall,
                                                          test_size = 0.5, random_state=52)
    return x_trainset, x_calibration, y_trainset, y_calibration, x_test1, x_test2, y_test1, y_test2
"""
:param list: list of elements
:return list of unique elements
"""
def unique(list):
    unique_list = []
    for x in list:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
"""
:param nodelist: list of nodes
:return a list of unique parent nodes
"""
def New_split_nodes(nodelist):
    parents = []
    for i in nodelist:
        p = i.up
        parents.append(p)
    pa = unique(parents)
    return pa
