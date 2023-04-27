"""
:param pval: a dictionary of p_value tables for each layer of the tree
:param alpha: a threshold between 0 to 1
:return: prediction set
"""
def pred_const(pval, alpha):
    aux = list()
    for i in range(pval["pval0"].shape[0]):
        label = list()
        # print("sample",x)
        if pval["pval0"][i][0] > alpha:
            if pval["pval1"][i][0] > alpha:
                label.append("Neuron2")

            if pval["pval1"][i][1] > alpha:
                label.append("Neuron3")

            if pval["pval1"][i][2] > alpha:
                if pval["pval2"][i][0] > alpha:
                    label.append("Neuron1-1")

                if pval["pval2"][i][1] > alpha:
                    label.append("Neuron1-2")

        if (pval["pval0"][i] <= alpha) or ((pval["pval1"][i,:] <= alpha).all()) or ((pval["pval2"][i,:] <= alpha).all()):
            label = ["outlier"]
        # print(label)
        aux.append(label)
    return aux