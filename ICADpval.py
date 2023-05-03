from sklearn.svm import OneClassSVM
import utils
import p_adjustment
import predset_const
import evaluation_metrics
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

# create a hierarchical tree and get the depth of it
t = Tree('(Neuron2, Neuron3, (Neuron1-1, Neuron1-2)Neuron1)Neurons;',format = 1)
node = t.get_tree_root()
_, depth = node.get_farthest_node()
# create a pval dict

powerlistall = list()
fdrlistall = list()
fcrlistall = list()
avg_sizelistall = list()
newfdrlistall = list()
newfdrlistall2 = list()
newfdrlistall3 = list()
newfdrlistall4 = list()

for alpha in np.arange(0.1, 0.32, 0.05):
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
        # data split into training, calibration and test
        x_trainset, x_calibration, y_trainset, y_calibration, x_testall, y_testall = \
            utils.ICAD_split(data, labels, ['Neuron3', 'Neuron1-2', 'Neuron1-1', 'Neuron2'],
                             ['Microglia','Oligo2','Oligo1','Astrocytes'], 0.5, 1.0, 0.3)
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

        '''
        #plt.hist(pval['pval0'][y_testall != "outlier"])
        #plt.show()
        if (alpha == 0.1) and (runtime == 0):
            parent = pval["pval0"][:,0]
            child = pval["pval1"]
            utils.plot_scatter(parent, child,'Neuron',['Neuron2','Neuron3','Neuron1'], y_testall)
            #utils.plot_density(parent, child, 'Neuron', ['Neuron2','Neuron3','Neuron1'])
        '''
        padj = p_adjustment.p_val_adjust(t,pval,len(y_testall))
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

    newfdrlistall.append(newfdrlist)
    newfdrlistall2.append(newfdrlist2)
    newfdrlistall3.append(newfdrlist3)
    newfdrlistall4.append(newfdrlist4)

    powerlistall.append(powerlist)
    fdrlistall.append(fdrlist)
    fcrlistall.append(fcrlist)
    avg_sizelistall.append(avg_sizelist)

np.random.seed(10)
fig, axs = plt.subplots(4,2)
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
