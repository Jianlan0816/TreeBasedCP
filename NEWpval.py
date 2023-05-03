import utils
import predset_const
import evaluation_metrics
import numpy as np
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
#print(x_calibration.shape)
# create a hierarchical tree and get the depth of it
t = Tree('(Neuron2, Neuron3, (Neuron1-1, Neuron1-2)Neuron1)Neurons;',format = 1)
node = t.get_tree_root()
_, depth = node.get_farthest_node()
# create a pval dict
pval = {}

powerlistall = list()
fdrlistall = list()
fcrlistall = list()
avg_sizelistall = list()
for alpha in np.arange(0.1, 0.31, 0.05):
    print("alpha", alpha)
    powerlist = list()
    fdrlist = list()
    fcrlist = list()
    avg_sizelist = list()
    for runtime in range(50):

        # data split into training, calibration and test
        x_trainset, x_calibration, y_trainset, y_calibration, x_test1, x_test2, y_test1, y_test2 = \
            utils.NEW_split(data, labels, ['Neuron3', 'Neuron1-2', 'Neuron1-1', 'Neuron2'],
                             ['Microglia','Oligo2','Oligo1','Astrocytes'], 0.5, 1.0, 0.3)

        for d in range(math.floor(depth + 1)):
            node_list = utils.search_by_depth(t, d)
            pval["pval{0}".format(d)] = np.empty([len(y_test1)+len(y_test2), len(node_list)])
            pa, _ = utils.New_split_nodes(node_list)
            c = 0
            for node in pa:

                train_x, train_y = utils.NEW_transform(x_trainset, y_trainset, node)
                cal_x, cal_y = utils.NEW_transform(x_calibration, y_calibration, node)

                train1_x = np.concatenate((train_x, x_test1), axis = 0)
                train1_y = np.concatenate((train_y, [max(train_y)+1]*len(y_test1)), axis = 0)
                train2_x = np.concatenate((train_x, x_test2), axis = 0)
                train2_y = np.concatenate((train_y, [max(train_y)+1]*len(y_test2)), axis = 0)
                clf1 = svm.SVC(probability=True)
                clf2 = svm.SVC(probability=True)
                clf1.fit(train1_x, train1_y)
                clf2.fit(train2_x, train2_y)

                conf_score1 = clf2.predict_proba(cal_x)
                conf_score2 = clf1.predict_proba(cal_x)

                t_score1 = clf2.predict_proba(x_test1)
                t_score2 = clf1.predict_proba(x_test2)

                for q in range(t_score1.shape[1]-1):
                    conf_s = conf_score1[cal_y == np.unique(cal_y)[q],:]
                    for p in range(t_score1.shape[0]):
                        p_val = (np.sum(t_score1[p][q] >= conf_s[:,q]) + 1) / (conf_s.shape[0] + 1)
                        pval["pval{0}".format(d)][p][q+c] = p_val
                for q in range(t_score2.shape[1]-1):
                    conf_s = conf_score2[cal_y == np.unique(cal_y)[q],:]
                    for p in range(t_score2.shape[0]):
                        p_val = (np.sum(t_score2[p][q] >= conf_s[:,q]) + 1) / (conf_s.shape[0] + 1)
                        pval["pval{0}".format(d)][p + t_score2.shape[0]][q+c] = p_val

                if node is None:
                    c = c
                else:
                    c = c + len(node.get_children())

        y_testall = np.concatenate([y_test1, y_test2], axis=0)
        #print(np.unique(y_testall))
        #print(pval['pval0'].shape)
        #plt.hist(pval['pval0'][y_testall != "outlier"])
        #plt.show()

        aux = predset_const.pred_const(pval, alpha)
        power, fdr, fcr, avg_size = evaluation_metrics.calculate_coverage(aux, y_testall)
        powerlist.append(power)
        fdrlist.append(fdr)
        fcrlist.append(fcr)
        avg_sizelist.append(avg_size)

    powerlistall.append(powerlist)
    fdrlistall.append(fdrlist)
    fcrlistall.append(fcrlist)
    avg_sizelistall.append(avg_sizelist)

np.random.seed(10)
fig, axs = plt.subplots(2,2)
axs[0,0].boxplot(powerlistall)
axs[0,0].set_title("Power")
axs[0,0].set_xticks([1,2,3,4,5])
axs[0,0].set_xticklabels([0.1,0.15,0.20,0.25,0.3])
axs[0,1].boxplot(fdrlistall)
axs[0,1].set_title("FDR")
axs[0,1].set_xticks([1,2,3,4,5])
axs[0,1].set_xticklabels([0.1,0.15,0.20,0.25,0.3])
axs[1,0].boxplot(fcrlistall)
axs[1,0].set_title("FCR")
axs[1,0].set_xticks([1,2,3,4,5])
axs[1,0].set_xticklabels([0.1,0.15,0.20,0.25,0.3])
axs[1,1].boxplot(avg_sizelistall)
axs[1,1].set_title("Avg_size")
axs[1,1].set_xticks([1,2,3,4,5])
axs[1,1].set_xticklabels([0.1,0.15,0.20,0.25,0.3])

plt.show()

