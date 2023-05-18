import numpy as np
"""
:param pval: a dictionary of p_value tables for each layer of the tree
:param alpha: a threshold between 0 to 1
:param tree: a dictionary of tree node names, saved as dict of lists. For example, {pval0: ['Nueron', 'Neuron-1']}
:param d: int, depth of the tree
:return: prediction set
"""
def pred_const(pval, alpha):
    aux = list()
    d = 0
    alphanew = alpha
    for i in range(pval["pval{0}".format(d)].shape[0]):
        label = list()
        # print("sample",x)
        if (pval["pval0"][i] <= alphanew) or ((pval["pval1"][i,:] <= alphanew).all()) or ((pval["pval2"][i,:] <= alphanew).all()):
            label = ["outlier"]
        else:
            if pval["pval0"][i] > alphanew and pval["pval1"][i][0] > alphanew:
                label.append("Neuron2")

            if pval["pval0"][i] > alphanew and pval["pval1"][i][1] > alphanew:
                label.append("Neuron3")

            if pval["pval0"][i] > alphanew and pval["pval1"][i][2] > alphanew and pval["pval2"][i][0] > alphanew:
                label.append("Neuron1-1")

            if pval["pval0"][i] > alphanew and pval["pval1"][i][2] > alphanew and pval["pval2"][i][1] > alphanew:
                label.append("Neuron1-2")
        # print(label)
        aux.append(label)
    return aux

'''
:param pval: ndarray, p values for each labels
:param alpha: float, threshold between 0 to 1
:param labels: list, customized ordered label list
:return: prediction set'''
def pred_const2(pval2, alpha, labels):
    aux = list()
    for i in range(pval2.shape[0]):
        if ((pval2[i,:] <= alpha).all()):
            l = ['outlier']
        else:
            #for j in range(pval2.shape[1]):
            #    if pval2[i][j] > alpha:
            #        l.append(labels[j])
            l = np.array(labels)[pval2[i,:] > alpha]
            l = l.tolist()
        aux.append(l)
    return aux

