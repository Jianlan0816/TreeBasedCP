import numpy as np
import utils
import math
from ete3 import Tree
from sklearn import svm
import matplotlib.pyplot as plt
# read a csv file with label
data = np.loadtxt('/Users/Jianlan/Downloads/mousebrain_Data/mousebrain_simulate3_pcalarge.txt')
#data = np.loadtxt('/Users/Jianlan/Downloads/10X_pbmc_batch1_pca.txt')
#data = data.T
labels = np.loadtxt('/Users/Jianlan/Downloads/mousebrain_Data/mousebrain_simlabel3_large.txt', dtype='str')
#labels = np.loadtxt('/Users/Jianlan/Downloads/10X_pbmc_batch1_label_pca.txt')
#print(data.shape)

x_train1, x_train2, y_train1, y_train2, x_test1, x_test2, y_test1, y_test2 =\
    utils.BCOPS_split(data, labels, ['Neuron3', 'Neuron1-2', 'Neuron1-1', 'Neuron2'],
                     ['Microglia','Oligo2','Oligo1','Astrocytes'],0.5, 1.0)

# create a hierarchical tree and get the depth of it
t = Tree('(Neuron2, Neuron3, (Neuron1-1, Neuron1-2)Neuron1)Neurons;',format = 1)
node = t.get_tree_root()
_, depth = node.get_farthest_node()
# create a pval dict
pval = {}

for d in range(math.floor(depth + 1)):
    node_list = utils.search_by_depth(t, d)
    pval["pval{0}".format(d)] = np.empty([len(y_test1)+len(y_test2), len(node_list)])

    for j in range(len(node_list)):
        trainx1, trainy1 = utils.ICAD_transform(x_train1, y_train1, node_list[j])
        trainx2, trainy2 = utils.ICAD_transform(x_train2, y_train2, node_list[j])

        clf1 = svm.SVC(probability=True)
        train1_x = np.concatenate((trainx1, x_test1), axis=0)
        train1_y = ['a'] * trainx1.shape[0] + ['b'] * x_test1.shape[0]
        clf1.fit(train1_x, train1_y)
        clf2 = svm.SVC(probability=True)
        train2_x = np.concatenate((trainx2, x_test2), axis=0)
        train2_y = ['a'] * trainx2.shape[0] + ['b'] * x_test2.shape[0]
        clf2.fit(train2_x, train2_y)

        for m in range(x_test1.shape[0]):
            confset = np.vstack([trainx1, x_test1[m]])
            conf_scores = clf2.predict_proba(confset)
            p = np.sum(conf_scores[-1][0] >= conf_scores[:, 0]) / (trainx1.shape[0] + 1)
            pval["pval{0}".format(d)][m][j] = p
        for n in range(x_test2.shape[0]):
            confset = np.vstack([trainx2, x_test2[n]])
            conf_scores = clf1.predict_proba(confset)
            p = np.sum(conf_scores[-1][0] >= conf_scores[:, 0]) / (trainx2.shape[0] + 1)
            pval["pval{0}".format(d)][n + x_test1.shape[0]][j] = p

y_testall = np.concatenate([y_test1, y_test2], axis=0)
plt.hist(pval['pval0'][y_testall != "outlier"])
plt.show()