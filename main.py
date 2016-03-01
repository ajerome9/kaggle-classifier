import xgboost as xgb
import numpy as np
import operator
import os
import pickle
from collections import defaultdict
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
    
import conf
from data import get_cvdata
from run_xgboost import train_xgb_for_cv, num_round
from sklearn.metrics import roc_auc_score
from log import msg
from scoring import score_cv

def get_bw(s, start, end):
    return ((s.split(start))[1].split(end)[0])

def get_booster_feats(bst):
    trees = bst.get_dump(with_stats=True)
    bst.dump_model('output-xgbdump-'+datetime.now().strftime("%Y%m%d-%H%M%S-%f")+'.txt', with_stats=True)
    allgain = defaultdict(float)
    count = defaultdict(int)
    roots = {}
    mainroots = (0,1,2)
    
    for tree in trees:
        for line in tree.split('\n'):
            line = line.strip()
            if line:
                parts = line.split(':')
                pos, line = int(parts[0]), parts[1]
                if line.startswith('['):
                    fcomp = get_bw(line, '[', ']')
                    feat = int(get_bw(fcomp, 'f', '<' if '<' in fcomp else '>'))
                    gain = get_bw(line, 'gain=', ',')
                    #print(line, feat, gain)
                    allgain[feat] += float(gain)
                    count[feat] += 1
                    
                    # take a note of feats that were the mainroots of the first tree
                    if (len(roots) < len(mainroots)):
                        if pos in mainroots:
                            for tgt in mainroots:
                                if pos==tgt and tgt not in roots:
                                    roots[tgt] = feat
    
    return allgain, count, roots
            
def get_topgain_feats(bst):
    gain, count, roots = get_booster_feats(bst)
    ratings = defaultdict(float)
    for feat,g in gain.items():
        c = count[feat]
        ratings[feat] = g/np.power(c,.8)
    ratings = sorted(ratings.items(), key=operator.itemgetter(1), reverse=True)
    return ratings, roots

def do_xgb(withcols=None, removecols=None):
    cvdata = get_cvdata(withcols=withcols, removecols=removecols)
    msg('Loaded cvdata: {}'.format(cvdata))
    xtrain, xtest = cvdata.xtrain, cvdata.xtest
    ytrain, ytest = cvdata.ytrain, cvdata.ytest
    featnames = cvdata.featnames

    msg('building dtrain dmatrix')
    dtrain = xgb.DMatrix(xtrain, label=ytrain)
    msg('building dtest dmatrix')
    dtest = xgb.DMatrix(xtest, label=ytest)

    bst = train_xgb_for_cv(dtrain, dtest, treecount=num_round)

    ratings, roots = get_topgain_feats(bst)

    auctrain, probatrain, auctest, probatest = score_cv(bst, dtrain, dtest, genplots=False)
    print('AUC_RPT, {}, {}'.format(auctrain, auctest))
    return ratings, probatest, ytest, roots

def exp_tree_with_toppers():
    """Find out how the auc turns out when only the top feats from a run participate in xgboost"""
    ratings, _, _ = do_xgb()
    for i in range(3,50,2):
        print('---------------')
        print(i)
        topfeats = [r[0] for r in ratings[:i]]
        print(topfeats)
        ratings2, _, _ = do_xgb(withcols=topfeats)
        print(ratings2)

def exp_tree_by_removing_toppers():
    """Find out how the auc turns out when only the top feats from a run participate in xgboost"""
    ratings, _, _ = do_xgb()
    top3feats = [r[0] for r in ratings[:3]]

    for i in range(4):
        msg('Will remove feats: {}'.format(top3feats))
        ratings, _, _ = do_xgb(removecols=top3feats)
        top3feats.extend([r[0] for r in ratings[:3]])

def get_corr_cols_via_numpy(xtrain):
    msg('computing correlated cols')
    corr = np.corrcoef(xtrain.toarray().T)
    removedcols = set()
    for i, r in enumerate(corr):
        for j, c in enumerate(r):
            cij = corr[i, j]
            if (i != j) and (abs(cij) > 0.7):
                if (i not in removedcols) and (j not in removedcols):
                    removedcols.add(j)
    msg('    done computing correlated cols. count = {}'.format(len(removedcols)))
    return removedcols

def generate_colmap(n, colstoremove):
    """Generates a map between indices in the final array -vs- original indices before column filtering"""
    colmap = {}
    j = 0
    for i in range(n):
        if i not in set(colstoremove):
            colmap[j] = i
            j+=1
    return colmap

def get_correlated_cols():
    if os.path.exists(conf.cvcorrelatedcols):
        msg('reading cvcorrelatedcols: {}'.format(conf.cvcorrelatedcols))
        colstoremove, n = pickle.load(open(conf.cvcorrelatedcols, 'rb'))
        msg('    correlated cols count = {}'.format(len(colstoremove)))
    else:
        msg('no cvcorrelatedcols file {}. will compute correlated columns'.format(conf.cvcorrelatedcols))
        cvdata = get_cvdata()
        msg('Loaded cvdata for computing correlatedcols: {}'.format(cvdata))
        correlatedcols = get_corr_cols_via_numpy(cvdata.xtrain)
        colstoremove = list(correlatedcols)
        n = cvdata.xtrain.shape[1]
        msg('pickling correlatedcols for reuse')        
        pickle.dump((colstoremove, n), open(conf.cvcorrelatedcols, 'wb'))

    return colstoremove, n

def exp_tree_by_removing_tree_roots():
    # remove root, l, r
    # force tree construction to take a different route
    colstoremove, n = get_correlated_cols()
    colmap = generate_colmap(n, colstoremove)
    
    ratings, proba1, ytest, roots = do_xgb(removecols=colstoremove)
    print('Roots-Orig: ', roots)
    print(ratings[:5])
    
    newproba = [proba1]

    for i in range(2):
        print('\n==========================\nEnsemble-{}'.format(i+1))
        print('root to remove={}, index in original array={}'.format(roots[0], colmap[roots[0]]))
        colstoremove.append(colmap[roots[0]])
        colmap = generate_colmap(n, colstoremove)
        
        ratings, proba2, _, roots = do_xgb(removecols=colstoremove)
        newproba.append(proba2)

        proba3 = []
        for (p) in zip(proba1, proba2):
            proba3.append((np.mean(p)))

        print('Avg AUC@Ensemble#{} = {}'.format(i+1, roc_auc_score(ytest, np.array(proba3))))
        print('Roots-New: ', roots)
        print(ratings[:5])

def rf():
    cvdata = get_cvdata()
    msg('Loaded cvdata: {}'.format(cvdata))
    xtrain, xtest = cvdata.xtrain, cvdata.xtest
    ytrain, ytest = cvdata.ytrain, cvdata.ytest
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=1, n_jobs=40,
                                 random_state=0, max_features=int(np.sqrt(xtrain.shape[1])), verbose=1)
    clf.fit(xtrain, ytrain)
    proba = clf.predict_proba(xtest)[:,1]
    auc = roc_auc_score(ytest, proba)
    print(auc)
    

if __name__=='__main__':
    #exp_tree_with_toppers()
    #exp_tree_by_removing_toppers()
    #exp_tree_by_removing_tree_roots()
    rf()