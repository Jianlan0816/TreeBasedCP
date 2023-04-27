"""
:param conf_pred: list of prediction sets
:param y_test: list of true labels
:return fdr, fcr and power
"""
def calculate_coverage(conf_pred, y_test):
    tp = 0
    fp = 0
    fc = 0
    tc = 0
    pred_inlier = 0
    true_outlier = 0
    size = list()
    for i in range(len(conf_pred)):
        if y_test[i] == 'outlier' and conf_pred[i] == ["outlier"]:
            tp = tp + 1
        if y_test[i] != 'outlier' and (y_test[i] in conf_pred[i]):
            tc = tc + 1
            size.append(len(conf_pred[i]))
        if y_test[i] != 'outlier' and (y_test[i] not in conf_pred[i]) and conf_pred[i] != ["outlier"]:
            fc = fc + 1
        if y_test[i] != 'outlier' and conf_pred[i] == ["outlier"]:
            fp = fp + 1
        if conf_pred[i] != ["outlier"]:
            pred_inlier = pred_inlier + 1
        if y_test[i] == 'outlier':
            true_outlier = true_outlier + 1
    if true_outlier == 0:
        print("true_outlier is zero")
    if pred_inlier == 0:
        print("pred_inlier is zero")
    if (fp + tp) == 0:
        print("fp+tp is zero")
    # power, fdr, fcr, avg_size
    return tp/true_outlier, fp/(fp+tp), fc/pred_inlier, sum(size)/len(size)