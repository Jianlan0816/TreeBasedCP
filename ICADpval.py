from sklearn.svm import OneClassSVM
import utils
import p_adjustment
import predset_const
import evaluation_metrics
import numpy as np
import math
from ete3 import Tree
import matplotlib.pyplot as plt
import Compare_methods

# read a csv file with label
data = np.loadtxt('/Users/Jianlan/Downloads/mousebrain_Data/mousebrain_simulate3_pcalarge.txt')
#data = np.loadtxt('/Users/Jianlan/Downloads/sim_15class/sim_15class_pca.txt')
#data = data.T
#labels = np.loadtxt('/Users/Jianlan/Downloads/sim_15class/simlabel_15class_pca.txt', dtype='str')
labels = np.loadtxt('/Users/Jianlan/Downloads/mousebrain_Data/mousebrain_simlabel3_large.txt', dtype='str')
#print(data.shape)

# create a hierarchical tree and get the depth of it
t = Tree('(Neuron2, Neuron3, (Neuron1-1, Neuron1-2)Neuron1)Neurons;',format = 1)
#t = Tree('(((t4,t1)Node3,t11)Node2,((t6,(t14,(((t2,t9)Node9,t7)Node8,(t3,t8)Node10)Node7)Node6)Node5,((t13,t15)Node12,((t12,t10)Node14,t5)Node13)Node11)Node4)Node1;', format = 1)
node = t.get_tree_root()
_, depth = node.get_farthest_node()
# create a pval dict

powerlistall = list()
fdrlistall = list()
fcrlistall = list()
avg_sizelistall = list()

powerlistall2 = list()
fdrlistall2 = list()
fcrlistall2 = list()
avg_sizelistall2 = list()

newfdrlistall = list()

for alpha in np.arange(0.1, 0.32, 0.05):
    print("alpha", alpha)
    powerlist = list()
    fdrlist = list()
    fcrlist = list()
    avg_sizelist = list()

    powerlist2 = list()
    fdrlist2 = list()
    fcrlist2 = list()
    avg_sizelist2 = list()

    newfdrlist = list()

    for runtime in range(10):
        # data split into training, calibration and test
        x_trainset, x_calibration, y_trainset, y_calibration, x_testall, y_testall = \
            utils.ICAD_split(data, labels, ['Neuron3', 'Neuron1-2', 'Neuron1-1', 'Neuron2'],
                             ['Microglia','Oligo2','Oligo1','Astrocytes'], 0.5, 1.0, 0.3)


        pval2 = Compare_methods.ICAD(x_trainset, y_trainset, x_calibration, y_calibration, x_testall, ['Neuron3', 'Neuron1-2', 'Neuron1-1', 'Neuron2'])
        pval = {}
        for d in range(math.floor(depth + 1)):
            node_list = utils.search_by_depth(t, d)
            pval["pval{0}".format(d)] = np.empty([len(y_testall), len(node_list)])
            for j in range(len(node_list)):
                trainx, _ = utils.ICAD_transform(x_trainset, y_trainset, node_list[j])
                #print("tree", node_list[j].name, trainx.shape)
                calx, _ = utils.ICAD_transform(x_calibration, y_calibration, node_list[j])
                #print("tree", node_list[j].name, calx.shape)
                clf = OneClassSVM(gamma='scale', kernel='rbf').fit(trainx)
                cal_score = clf.decision_function(calx)
                x_test_score = clf.decision_function(x_testall)
                for i in range(len(y_testall)):
                    k = np.sum(cal_score <= x_test_score[i])
                    k = k.item()
                    p = (k + 1) / (len(cal_score) + 1)
                    pval["pval{0}".format(d)][i][j] = p


        #plt.hist(pval['pval0'][y_testall == "outlier"])
        #neuron2 = pval2[np.where(y_testall == "Neuron1-2")]
        #print(type(neuron2))
        #plt.hist(neuron2[:,1])
        #plt.show()

        '''
        if (alpha == 0.1) and (runtime == 0):
            parent = pval["pval0"][:,0]
            child = pval["pval1"]
            utils.plot_scatter(parent, child,'Neuron',['Neuron2','Neuron3','Neuron1'], y_testall)
            #utils.plot_density(parent, child, 'Neuron', ['Neuron2','Neuron3','Neuron1'])
        '''
        aux2 = predset_const.pred_const2(pval2, alpha, ['Neuron3', 'Neuron1-2', 'Neuron1-1', 'Neuron2'])
        #print(aux2[1:10])
        power2, fdr2, fcr2, avg_size2 = evaluation_metrics.calculate_coverage2(aux2, y_testall)
        #print("compare", power2)
        powerlist2.append(power2)
        fdrlist2.append(fdr2)
        fcrlist2.append(fcr2)
        avg_sizelist2.append(avg_size2)

        aux = predset_const.pred_const(pval, alpha)
        #print(aux[802])
        power, fdr, fcr, avg_size = evaluation_metrics.calculate_coverage(aux, y_testall)
        #print("tree", power)
        powerlist.append(power)
        fdrlist.append(fdr)
        fcrlist.append(fcr)
        avg_sizelist.append(avg_size)

        padj = p_adjustment.p_val_adjust(t, pval, len(y_testall), alpha)
        reject = p_adjustment.fdr_correction(padj, alpha)
        newfdr = p_adjustment.fdr_calculation(reject, y_testall)
        newfdrlist.append(newfdr)
        #print(pval['pval0'][802])

    newfdrlistall.append(newfdrlist)

    powerlistall.append(powerlist)
    fdrlistall.append(fdrlist)
    fcrlistall.append(fcrlist)
    avg_sizelistall.append(avg_sizelist)

    powerlistall2.append(powerlist2)
    fdrlistall2.append(fdrlist2)
    fcrlistall2.append(fcrlist2)
    avg_sizelistall2.append(avg_sizelist2)

np.random.seed(10)
fig, axs = plt.subplots(2,4)
axs[0,0].boxplot(powerlistall)
axs[0,0].set_title("Power_tree")
axs[0,0].set_xticks([1,2,3,4,5])
axs[0,0].set_xticklabels([0.1,0.15,0.20,0.25,0.3])
axs[0,1].boxplot(fdrlistall)
axs[0,1].set_title("FDR_tree")
axs[0,1].set_ylim([0, 1])
axs[0,1].set_xticks([1,2,3,4,5])
axs[0,1].set_xticklabels([0.1,0.15,0.20,0.25,0.3])
axs[0,1].axhline(y=0.1,linestyle ='--')
axs[0,1].axhline(y=0.15,linestyle ='--')
axs[0,1].axhline(y=0.2,linestyle ='--')
axs[0,1].axhline(y=0.25,linestyle ='--')
axs[0,1].axhline(y=0.3,linestyle ='--')
axs[1,0].boxplot(fcrlistall)
axs[1,0].set_title("FCR_tree")
axs[1,0].set_xticks([1,2,3,4,5])
axs[1,0].set_xticklabels([0.1,0.15,0.20,0.25,0.3])
axs[1,1].boxplot(avg_sizelistall)
axs[1,1].set_title("Avg_size_tree")
axs[1,1].set_xticks([1,2,3,4,5])
axs[1,1].set_xticklabels([0.1,0.15,0.20,0.25,0.3])
axs[0,2].boxplot(newfdrlistall)
#axs[2,0].axhline(y = 0.2, color = 'r', linestyle = 'dashed')
axs[0,2].set_ylim([0, 1])
axs[0,2].set_title("FDR adjustment tree")
axs[0,2].set_xticks([1,2,3,4,5])
axs[0,2].set_xticklabels([0.1,0.15,0.20,0.25,0.3])
axs[0,2].axhline(y=0.1,linestyle ='--')
axs[0,2].axhline(y=0.15,linestyle ='--')
axs[0,2].axhline(y=0.2,linestyle ='--')
axs[0,2].axhline(y=0.25,linestyle ='--')
axs[0,2].axhline(y=0.3,linestyle ='--')

axs[0,3].boxplot(powerlistall2)
axs[0,3].set_title("Power_ICAD")
axs[0,3].set_xticks([1,2,3,4,5])
axs[0,3].set_xticklabels([0.1,0.15,0.20,0.25,0.3])
axs[1,2].boxplot(fdrlistall2)
axs[1,2].set_title("FDR_ICAD")
axs[1,2].set_ylim([0, 1])
axs[1,2].set_xticks([1,2,3,4,5])
axs[1,2].set_xticklabels([0.1,0.15,0.20,0.25,0.3])
axs[1,2].axhline(y=0.1,linestyle ='--')
axs[1,2].axhline(y=0.15,linestyle ='--')
axs[1,2].axhline(y=0.2,linestyle ='--')
axs[1,2].axhline(y=0.25,linestyle ='--')
axs[1,2].axhline(y=0.3,linestyle ='--')
axs[1,3].boxplot(fcrlistall2)
axs[1,3].set_title("FCR_ICAD")
axs[1,3].set_xticks([1,2,3,4,5])
axs[1,3].set_xticklabels([0.1,0.15,0.20,0.25,0.3])

plt.show()
