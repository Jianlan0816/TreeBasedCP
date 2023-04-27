from sklearn.svm import OneClassSVM
import utils
import numpy as np
import math
from ete3 import Tree
import matplotlib.pyplot as plt

# read a csv file with label
data = np.loadtxt('/Users/Jianlan/Downloads/mousebrain_Data/mousebrain_simulate3_pcalarge.txt')
#data = np.loadtxt('/Users/Jianlan/Downloads/10X_pbmc_batch1_pca.txt')
#data = data.T
labels = np.loadtxt('/Users/Jianlan/Downloads/mousebrain_Data/mousebrain_simlabel3_large.txt', dtype='str')
#labels = np.loadtxt('/Users/Jianlan/Downloads/10X_pbmc_batch1_label_pca.txt')
#print(data.shape)

# data split into training, calibration and test
x_trainset, x_calibration, y_trainset, y_calibration, x_testall, y_testall = \
    utils.ICAD_split(data, labels, ['Neuron3', 'Neuron1-2', 'Neuron1-1', 'Neuron2'],
                     ['Microglia','Oligo2','Oligo1','Astrocytes'], 0.5, 1.0, 0.3)

# create a hierarchical tree and get the depth of it
t = Tree('(Neuron2, Neuron3, (Neuron1-1, Neuron1-2)Neuron1)Neurons;',format = 1)
node = t.get_tree_root()
_, depth = node.get_farthest_node()
# create a pval dict
pval = {}

for d in range(math.floor(depth + 1)):
    node_list = utils.search_by_depth(t, d)
    pval["pval{0}".format(d)] = np.empty([len(y_testall), len(node_list)])
    for j in range(len(node_list)):
        trainx, trainy = utils.ICAD_transform(x_trainset, y_trainset, node_list[j])
        calx, caly = utils.ICAD_transform(x_calibration, y_calibration, node_list[j])
        clf = OneClassSVM(gamma='scale', kernel='rbf').fit(trainx)
        cal_score = clf.decision_function(calx)
        x_test_score = clf.decision_function(x_testall)
        for i in range(len(y_testall)):
            k = np.sum(cal_score <= x_test_score[i])
            k = k.item()
            p = (k + 1) / (len(cal_score) + 1)
            pval["pval{0}".format(d)][i][j] = p

plt.hist(pval['pval0'][y_testall != "outlier"])
plt.show()