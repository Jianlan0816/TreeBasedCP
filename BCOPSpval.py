import numpy as np
import utils
import math
import predset_const
import evaluation_metrics
import p_adjustment
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
newfdrlistall = list()
newfdrlistall2 = list()
newfdrlistall3 = list()
newfdrlistall4 = list()
for alpha in np.arange(0.1, 0.31, 0.05):
    print("alpha", alpha)
    powerlist = list()
    fdrlist = list()
    fcrlist = list()
    avg_sizelist = list()
    newfdrlist = list()
    newfdrlist2 = list()
    newfdrlist3 = list()
    newfdrlist4 = list()

    for runtime in range(50):
        x_train1, x_train2, y_train1, y_train2, x_test1, x_test2, y_test1, y_test2 =\
            utils.BCOPS_split(data, labels, ['Neuron3', 'Neuron1-2', 'Neuron1-1', 'Neuron2'],
                             ['Microglia','Oligo2','Oligo1','Astrocytes'],0.5, 1.0)


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

                conf_scores1 = clf2.predict_proba(trainx1)
                test_scores1 = clf2.predict_proba(x_test1)

                conf_scores2 = clf1.predict_proba(trainx2)
                test_scores2 = clf1.predict_proba(x_test2)

                for m in range(test_scores1.shape[0]):
                    k = np.sum(test_scores1[m][0] >= conf_scores1[:, 0])
                    p = (k.item() + 1) / (conf_scores1.shape[0] + 1)
                    pval["pval{0}".format(d)][m][j] = p
                for n in range(test_scores2.shape[0]):
                    k = np.sum(test_scores2[n][0] >= conf_scores2[:, 0])
                    p = (k.item() + 1) / (conf_scores2.shape[0] + 1)
                    pval["pval{0}".format(d)][n + x_test1.shape[0]][j] = p

        y_testall = np.concatenate([y_test1, y_test2], axis=0)
        #plt.hist(pval['pval0'][y_testall != "outlier"])
        #plt.show()
        padj = p_adjustment.p_val_adjust(t, pval, len(y_testall))
        reject = p_adjustment.fdr_correction(padj, 0.2)
        newfdr = p_adjustment.fdr_calculation(reject, y_testall)
        newfdrlist.append(newfdr)

        reject = p_adjustment.fdr_correction(padj, 0.15)
        newfdr = p_adjustment.fdr_calculation(reject, y_testall)
        newfdrlist2.append(newfdr)

        reject = p_adjustment.fdr_correction(padj, 0.1)
        newfdr = p_adjustment.fdr_calculation(reject, y_testall)
        newfdrlist3.append(newfdr)

        reject = p_adjustment.fdr_correction(padj, 0.05)
        newfdr = p_adjustment.fdr_calculation(reject, y_testall)
        newfdrlist4.append(newfdr)

        aux = predset_const.pred_const(pval, alpha)
        power, fdr, fcr, avg_size = evaluation_metrics.calculate_coverage(aux, y_testall)
        powerlist.append(power)
        fdrlist.append(fdr)
        fcrlist.append(fcr)
        avg_sizelist.append(avg_size)

    #print(len(powerlist))
    powerlistall.append(powerlist)
    fdrlistall.append(fdrlist)
    fcrlistall.append(fcrlist)
    avg_sizelistall.append(avg_sizelist)
    #print(len(powerlistall))
    newfdrlistall.append(newfdrlist)
    newfdrlistall2.append(newfdrlist2)
    newfdrlistall3.append(newfdrlist3)
    newfdrlistall4.append(newfdrlist4)

np.random.seed(10)
fig, axs = plt.subplots(4, 2)
axs[0, 0].boxplot(powerlistall)
axs[0, 0].set_title("Power")
axs[0, 0].set_xticks([1, 2, 3, 4, 5])
axs[0, 0].set_xticklabels([0.1, 0.15, 0.20, 0.25, 0.3])
axs[0, 1].boxplot(fdrlistall)
axs[0, 1].set_title("FDR")
axs[0, 1].set_xticks([1, 2, 3, 4, 5])
axs[0, 1].set_xticklabels([0.1, 0.15, 0.20, 0.25, 0.3])
axs[1, 0].boxplot(fcrlistall)
axs[1, 0].set_title("FCR")
axs[1, 0].set_xticks([1, 2, 3, 4, 5])
axs[1, 0].set_xticklabels([0.1, 0.15, 0.20, 0.25, 0.3])
axs[1, 1].boxplot(avg_sizelistall)
axs[1, 1].set_title("Avg_size")
axs[1, 1].set_xticks([1, 2, 3, 4, 5])
axs[1, 1].set_xticklabels([0.1, 0.15, 0.20, 0.25, 0.3])
axs[2,0].boxplot(newfdrlistall)
axs[2,0].axhline(y = 0.2, color = 'r', linestyle = 'dashed')
axs[2,0].set_title("FDR adjustment")
axs[2,0].set_xticks([1,2,3,4,5])
axs[2,0].set_xticklabels([0.1,0.15,0.20,0.25,0.3])
axs[2,1].boxplot(newfdrlistall2)
axs[2,1].axhline(y = 0.15, color = 'r', linestyle = 'dashed')
axs[2,1].set_title("FDR adjustment")
axs[2,1].set_xticks([1,2,3,4,5])
axs[2,1].set_xticklabels([0.1,0.15,0.20,0.25,0.3])
axs[3,0].boxplot(newfdrlistall3)
axs[3,0].axhline(y = 0.1, color = 'r', linestyle = 'dashed')
axs[3,0].set_title("FDR adjustment")
axs[3,0].set_xticks([1,2,3,4,5])
axs[3,0].set_xticklabels([0.1,0.15,0.20,0.25,0.3])
axs[3,1].boxplot(newfdrlistall4)
axs[3,1].axhline(y = 0.05, color = 'r', linestyle = 'dashed')
axs[3,1].set_title("FDR adjustment")
axs[3,1].set_xticks([1,2,3,4,5])
axs[3,1].set_xticklabels([0.1,0.15,0.20,0.25,0.3])

plt.show()