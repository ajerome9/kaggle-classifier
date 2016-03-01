import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from log import msg
#import seaborn as sns

def get_cv_auc_bestiter(bst, dmx):
    proba = bst.predict(dmx, ntree_limit=bst.best_iteration)
    auc = roc_auc_score(dmx.get_label(), proba)
    return auc, proba

def gen_roc_plot(y, proba, auc, plot_name):
    plt.clf()
    fpr, tpr, _ = roc_curve(y, proba)
    plt.plot(fpr, tpr, label='auc:{:.3f}'.format(auc), linewidth=3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig(plot_name)

def score_cv(bst, dtrain, dtest, genplots=True):
    msg('computing auc')
    msg('    train auc')
    auctrain, probatrain = get_cv_auc_bestiter(bst, dtrain)
    msg('    test auc')
    auctest, probatest = get_cv_auc_bestiter(bst, dtest)
    msg('    auc: train = {}, test = {}'.format(auctrain, auctest))

    if genplots:
        msg('generating roc plots')
        gen_roc_plot(dtrain.get_label(), probatrain, auctrain, 'output-plot-roc-train.png')
        gen_roc_plot(dtest.get_label(), probatest, auctest, 'output-plot-roc-test.png')

    return auctrain, probatrain, auctest, probatest
    
def score_real(bst, dtrain):
    auctrain = roc_auc_score(dtrain.get_label(), bst.predict(dtrain))
    msg('train auc score = {}'.format(auctrain))
    
def score_xgbcv(cvret):
    plt.clf()
    testauc, teststd = np.array([]), np.array([])
    trainauc, trainstd = np.array([]), np.array([])
    for ret in cvret:
        r = ret.split('\t')
        testauc = np.append(testauc, float(r[1].split(':')[1].split('+')[0]))
        teststd = np.append(teststd, float(r[1].split(':')[1].split('+')[1]))
        trainauc = np.append(trainauc, float(r[2].split(':')[1].split('+')[0]))
        trainstd = np.append(trainstd, float(r[2].split(':')[1].split('+')[1]))
    
    plt.plot(np.arange(len(testauc)), testauc, color='g', label='Test auc')
    plt.fill_between(np.arange(len(testauc)), testauc - teststd, testauc + teststd, alpha=0.1, color='g')
    plt.plot(np.arange(len(trainauc)), trainauc, color='r', label='Train auc')
    plt.fill_between(np.arange(len(trainauc)), trainauc - trainstd, trainauc + trainstd, alpha=0.1, color='r')
    plt.legend()
    plt.savefig('output-xgbcv-auc.png')
    