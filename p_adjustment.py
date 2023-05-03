from ete3 import Tree
import math
import utils
import numpy as np
"""
:param Tree: ete3 tree structure
:param p_val: dictionary of p_values of test samples for each layer of the tree
:param n: number of test samples
:return: a list of adjusted p_values for all test samples
"""
def p_val_adjust(Tree, pval, n):
    node = Tree.get_tree_root()
    _, depth = node.get_farthest_node()
    p_adjlist = list()
    for i in range(n):
        p_test = list()
        for d in range(math.floor(depth + 1)):
            node_list = utils.search_by_depth(Tree, d)
            pa, parent = utils.New_split_nodes(node_list)
            p_layer = list()
            for j in range(len(node_list)):
                p = pval["pval{0}".format(d)][i][j]
                p = p/len(node_list)
                if pa == [None]:
                    p_parent = p
                else:
                    d_parent = d - 1
                    node_list_parent = utils.search_by_depth(Tree, d_parent)
                    pa_index = node_list_parent.index(parent[j])
                    p_parent = pval["pval{0}".format(d_parent)][i][pa_index]
                p_layer.append(min(p,p_parent))
            p_max = max(p_layer)
            p_test.append(p_max)
        p_adj = min(p_test)
        p_adjlist.append(p_adj)
    return p_adjlist

"""
:param pvals: array_like, 1d. Set of p-values of the individual tests
:param alpha: float, optional. Custom FDR. Defaults to '0.05'
:return rejected: ndarray, bool. True if a hypothesis is rejected, False if not
:return pvalue-corrected: ndarray. pvalues adjusted for multiple hypothesis testing to limit FDR
"""
def fdr_correction(pvals, alpha = 0.05):
    pvals = np.array(pvals)
    pvals_sortind = np.argsort(pvals)
    pvals_sorted = np.take(pvals, pvals_sortind)
    nobs = len(pvals_sorted)
    ecdffactor = np.arange(1, nobs+1)/float(nobs)
    reject = pvals_sorted <= ecdffactor*alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True
    reject_ = np.empty_like(reject)
    reject_[pvals_sortind] = reject
    return reject_

'''
:param reject: ndarray, bool. True if a hypothesis is rejected, False if not
:param y_testall: ndarray. Set of true labels for test samples
:return FDR
'''
def fdr_calculation(reject, y_testall):
    tp = 0
    fp = 0
    for i in range(len(y_testall)):
        if reject[i] == True and y_testall[i] == 'outlier':
            tp = tp + 1
        if y_testall[i] != 'outlier' and reject[i] == True:
            fp = fp + 1
    return fp/(fp+tp)