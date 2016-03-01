import sys
import os
import csv
from datetime import datetime
import operator
import pickle
import itertools

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack
import xgboost as xgb

from scoring import score_cv, score_real, score_xgbcv, get_cv_auc_bestiter
from const import DATECOLS, BOOLCOLS, CATENCCOLS, DROPCOLS
from feat import gen_fancy_feats
import conf
from log import msg
from conf import dtypesf as dtypespkl
from textmining import gen_text_feats_from_title_columns

# XGB Params
max_depth = 13
eta = 0.001
silent = 1
objective = 'binary:logistic'
gamma = 0
min_child_weight = 1
max_delta_step = 0
subsample = 0.95
colsample_bytree = 0.95
num_round = 20000
alpha = 0
lambdaparam = 1
scale_pos_weight=1

def get_dtypes():
    return pickle.load(open(dtypespkl, 'rb'))

def build_date_feats(df):
    """Converts each data column in the DATECOLS list into the corresponding
    day,mon,year,ordinal and drops the original date column
    """
    for c in DATECOLS:
        df = pd.merge(df.loc[:, c].apply(lambda x: pd.Series(
                    {c+'_ORD':x.toordinal() if x.year<2100 else -1, 
                     c+'_YEAR':x.year if x.year<2100 else -1, 
                     c+'_MON':x.month if x.year<2100 else -1, 
                     c+'_DAY':x.day if x.year<2100 else -1,
                     c+'_WKDAY':x.weekday() if x.year<2100 else -1,
                    })), df, left_index=True, right_index=True)
    df.drop(DATECOLS, axis=1, inplace=True)
    return df

def update_df_with_pvi_pkldicts(df):
    msg('    update_df_with_pvi_pkldicts')
    pklcols = os.listdir(conf.basepath+'pdicts/')
    for c in pklcols:
        msg('    * loading pkl for column {}'.format(c))
        d = pickle.load(open(conf.basepath+'pdicts/'+c, 'rb'))
        msg('        updating df column {}'.format(c))
        df.loc[:, c].update(pd.Series(d))
    msg('    completed update_df_with_pvi_pkldicts')

def read_csv_or_pickle(pklfile, csvfile):
    if os.path.exists(pklfile):
        msg('    reading from pickle file {}'.format(pklfile))
        df = pd.read_pickle(pklfile)
    else:
        msg('    no pickle file {}. reading from csv {}'.format(pklfile, csvfile))
        nandate = datetime(2100, 1, 1)
        dateparser = lambda dates: [nandate if pd.isnull(d) else pd.datetime.strptime(d, '%d%b%y:%H:%M:%S') for d in dates]
        df = pd.read_csv(csvfile, parse_dates=DATECOLS, date_parser=dateparser, dtype=get_dtypes())
        msg('    building date feats')
        df = build_date_feats(df)
        msg('    transforming boolean strings to ints')
        for c in BOOLCOLS:
            df.loc[:, c] = df.loc[:, c].apply(lambda x: 1 if x=='true' else -1 if x=='false' else 0)
        msg('    pickling dataframe to {}'.format(pklfile))
        df.to_pickle(pklfile)
        
    df.set_index('ID', inplace=True)
    update_df_with_pvi_pkldicts(df)
    df.reset_index(inplace=True)
    return df

def load_real_data():
    # read train/test files
    train = read_csv_or_pickle(conf.trainpkl, conf.trainf)
    train.drop('ID', axis=1, inplace=True)
    msg('    done reading train')
    
    test = read_csv_or_pickle(conf.testpkl, conf.testf)
    msg('    done reading test')

    # slice to get the X/y portions. idtest is a column of test ids.
    xtrain = train.iloc[:, :-1]
    ytrain = train.iloc[:, -1]
    xtest = test.iloc[:, 1:]
    idtest = test.iloc[:, 0]

    # handle missing values
    xtrain.fillna(-1, inplace=True)
    xtest.fillna(-1, inplace=True)    
    
    return xtrain, ytrain, xtest, idtest


def load_cv_data():
    """Reads the train data alone, optionally extracting n random rows from it.
    """
    n=None #conf.cvrows

    train = read_csv_or_pickle(conf.trainpkl, conf.trainf)
    train.drop('ID', axis=1, inplace=True)
    
    if n:
        perm = np.random.RandomState(0).permutation(len(train))[:n]
        X = train.iloc[perm, :-1]
        y = train.iloc[perm, -1]
    else:
        X = train.iloc[:, :-1]
        y = train.iloc[:, -1]

    # handle missing values
    X.fillna(-1, inplace=True)
    
    return X, y

def encode_categorical(dftrain, dftest, sep='_'):
    """Encodes the list of columns provided using a OneHotEncoding scheme.
    Returns a csr sparse matrix.
    """
    cols = CATENCCOLS
    vec = DictVectorizer(separator=sep)

    # OneHotEncoded version of columns in 'cols'
    cols_to_dict = lambda df, cols: df[cols].to_dict(orient='records')
    enctrain = vec.fit_transform(cols_to_dict(dftrain, cols))
    enctest = vec.transform(cols_to_dict(dftest, cols))
    
    dftrain = dftrain.drop(cols, axis=1)
    dftest = dftest.drop(cols, axis=1)

    # keeping track of feature names
    featnames = list(dftrain.columns)
    featnames.extend(vec.get_feature_names())

    # stack the original array with encoded columns
    return hstack([csr_matrix(dftrain.values), enctrain]), \
           hstack([csr_matrix(dftest.values), enctest]), featnames

def merge_text_feats(xtrain, xtest, txttrain, txttest, featnames, textfeatnames):
    train = hstack([xtrain, txttrain])
    test = hstack([xtest, txttest])
    featnames.extend(['TITLE_'+t for t in textfeatnames])
    return train, test, featnames

def remove_low_var_features(xtrain, xtest):
    selector = VarianceThreshold()
    xtrain = selector.fit_transform(xtrain)
    xtest = selector.transform(xtest)
    return xtrain, xtest, selector.get_support(indices=True)

def log_featimp_and_save_xgb(bst, featnames, featindices):
    scoremap = bst.get_fscore()
    featscores = sorted(scoremap.items(), key=operator.itemgetter(1), reverse=True)

    fdict = {}
    for i, featidx in enumerate(featindices):
        fdict[i] = featnames[featidx]

    impfeats = []
    for f, imp in featscores:
        fidx = int(f[1:])
        impfeats.append((fdict[fidx], imp, f))

    with open('output-impfeats.csv', 'w') as f:
        a = csv.writer(f)
        a.writerow(['feature', 'importance', 'xgbfeatid'])
        a.writerows(impfeats)

    bst.save_model('output-xgb.model')
    bst.dump_model('output-xgbdump.txt', with_stats=True)

def gen_features_cvreal(xtrain, xtest):
    msg('dropping {} low relevance cols'.format(len(DROPCOLS)))
    xtrain = xtrain.drop(DROPCOLS, axis=1)
    xtest = xtest.drop(DROPCOLS, axis=1)
    
    msg('gen_text_feats_from_title_columns (2 cols will be dropped, text feats will be merged later)')
    xtrain, xtest, txttrain, txttest, textfeatnames = gen_text_feats_from_title_columns(xtrain, xtest)

    msg('gen_fancy_feats')
    xtrain, xtest = gen_fancy_feats(xtrain, xtest)

    msg('encode_categorical')
    msg('    size before encoding = {}, {}'.format(xtrain.shape, xtest.shape))
    xtrain, xtest, featnames = encode_categorical(xtrain, xtest)
    msg('    size after encoding = {}, {}'.format(xtrain.shape, xtest.shape))

    msg('merge_text_feats')
    xtrain, xtest, featnames = merge_text_feats(xtrain, xtest, txttrain, txttest, featnames, textfeatnames)
    msg('    after merge_text_feats: xtrain: {}, xtest: {}'.format(xtrain.shape, xtest.shape))
    
    msg('feat selection: removing low variance features')
    xtrain, xtest, featindices = remove_low_var_features(xtrain, xtest)
    return xtrain, xtest, featnames, featindices

def gen_and_write_preds(bst, dtest, idtest):
    proba = bst.predict(dtest)

    msg('    writing predictions to file')
    with open('output-pred.csv', 'w') as f:
        f.write('ID,target\n')
        for (rowid, p) in zip(idtest, proba):
            f.write('{},{}\n'.format(rowid, p))

def get_xgbcv_dmatrix():
    if os.path.exists(conf.xgbcvdmatrix):
        msg('reading from xgbcvdmatrix: {}'.format(conf.xgbcvdmatrix))
        dmx = xgb.DMatrix(conf.xgbcvdmatrix)
    else:
        msg('no dmx file {}. will load data and gen feats'.format(conf.xgbcvdmatrix))
        msg('loading data')
        X, y = load_cv_data()
    
        msg('train_test_split')
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.75, random_state=0)
        xtrain, xtest = xtrain.copy(), xtest.copy()
        msg('xtrain: {}, xtest: {}, ytrain: {}, ytest: {}'.format(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))
    
        xtrain, xtest, featnames, featindices = gen_features_cvreal(xtrain, xtest)
    
        x = vstack([xtrain, xtest])
        y = np.concatenate((ytrain.values, ytest.values))
        msg('building dmx')
        dmx = xgb.DMatrix(x, label=y)
        msg('saving dmx')
        dmx.save_binary(conf.xgbcvdmatrix)
    return dmx

def get_cv_dmatrix():
    if os.path.exists(conf.cvdtrain):
        msg('reading from cvdtrain: {}'.format(conf.cvdtrain))
        dtrain = xgb.DMatrix(conf.cvdtrain)
        msg('reading from cvdtest: {}'.format(conf.cvdtest))
        dtest = xgb.DMatrix(conf.cvdtest)
        msg('reading cvfeats')
        featnames, featindices = pickle.load(open(conf.cvfeats, 'rb'))
    else:
        msg('no dmx file {}. will load data and gen feats'.format(conf.cvdtrain))
        msg('loading data')
        X, y = load_cv_data()
    
        msg('train_test_split')
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.75, random_state=0)
        xtrain, xtest = xtrain.copy(), xtest.copy()
        msg('xtrain: {}, xtest: {}, ytrain: {}, ytest: {}'.format(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))
    
        xtrain, xtest, featnames, featindices = gen_features_cvreal(xtrain, xtest)
        msg('after feat gen: xtrain: {}, xtest: {}, ytrain: {}, ytest: {}'.format(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))
        
        msg('building dtrain dmatrix')
        dtrain = xgb.DMatrix(xtrain, label=ytrain)
        msg('building dtest dmatrix')
        dtest = xgb.DMatrix(xtest, label=ytest)

        msg('saving dmx')
        dtrain.save_binary(conf.cvdtrain)
        dtest.save_binary(conf.cvdtest)
        msg('saving cvfeats')
        pickle.dump((featnames, featindices), open(conf.cvfeats, 'wb'))
    return dtrain, dtest, featnames, featindices


def get_real_dmatrix():
    if os.path.exists(conf.realdtrain):
        msg('reading from realdtrain: {}'.format(conf.realdtrain))
        dtrain = xgb.DMatrix(conf.realdtrain)
        msg('reading from realdtest: {}'.format(conf.realdtest))
        dtest = xgb.DMatrix(conf.realdtest)
        msg('reading realfeats/idtest')
        featnames, featindices, idtest = pickle.load(open(conf.realfeats, 'rb'))
    else:
        msg('no dmx file {}. will load data and gen feats'.format(conf.realdtrain))
        msg('loading data')
        xtrain, ytrain, xtest, idtest = load_real_data()
        msg('xtrain: {}, xtest: {}, ytrain: {}, idtest: {}'.format(xtrain.shape, xtest.shape, ytrain.shape, len(idtest)))
    
        xtrain, xtest, featnames, featindices = gen_features_cvreal(xtrain, xtest)
        msg('after feat gen: xtrain: {}, xtest: {}, ytrain: {}'.format(xtrain.shape, xtest.shape, ytrain.shape))
        
        msg('building dtrain dmatrix')
        dtrain = xgb.DMatrix(xtrain, label=ytrain)
        msg('building dtest dmatrix')
        dtest = xgb.DMatrix(xtest)

        msg('saving dmx')
        dtrain.save_binary(conf.realdtrain)
        dtest.save_binary(conf.realdtest)
        msg('saving realfeats/idtest')
        pickle.dump((featnames, featindices, idtest), open(conf.realfeats, 'wb'))
    return dtrain, dtest, featnames, featindices, idtest

def do_xgbcv():
    """Note: This method merges train/test data so that it can be handed off to xgb easily. 
    This might be introducing issues, especially there is a possibility of data leakage.
    This should not be the preferred method. Use do_cv instead"""
    
    dtrain = get_xgbcv_dmatrix()
    
    param = {'max_depth':max_depth, 'eta':eta, 'silent':silent, 'objective':objective, 
             'gamma':gamma, 'min_child_weight':min_child_weight, 'max_delta_step':max_delta_step, 
             'subsample':subsample, 'colsample_bytree':colsample_bytree,
             'alpha':alpha, 'lambda':lambdaparam, 'scale_pos_weight':scale_pos_weight}
    
    nfold = 4
    msg('starting xgbcv. {}-fold, rounds={}, params: {}'.format(nfold, num_round, param))
    
    cvret = xgb.cv(param, dtrain, num_round, nfold=nfold,
           metrics={'auc'}, seed=0)
    
    score_xgbcv(cvret)
    
    msg('completed do_xgbcv')

def do_gridsearch_cv():
    p = (('max_depth', [8,9,10]),
         ('eta', [0.005, 0.015, 0.1]),
         ('gamma', [0]),
         ('min_child_weight', [1]),
         ('max_delta_step', [0]),
         ('subsample', [0.6, 0.95]),
         ('colsample_bytree', [0.6, 0.95]),
         ('alpha', [0]),
         ('lambda', [1]),
         ('scale_pos_weight', [1, 2, 3.347]),
         ('objective', ['binary:logistic']),
         ('eval_metric', ['auc'])
         )

    pvalues = [e[1] for e in p]
    pnames = [e[0] for e in p]

    treecount = 1000
    
    numcombs = 1
    for values in pvalues:
        numcombs *= len(values)

    bestcomb, bestauctest, bestauctrain, bestiter = 0, 0, 0, 0

    dtrain, dtest, featnames, featindices = get_cv_dmatrix()

    for i, comb in enumerate(itertools.product(*pvalues)):
        param = {k:v for k, v in zip(pnames, comb)}

        print('\n\n-------------------------------------------')
        param['silent'] = silent
        msg('{}/{} with params: {}'.format(i+1, numcombs, param))
        watchlist  = [(dtrain,'train'), (dtest,'eval')]
        bst = xgb.train(param, dtrain, treecount, watchlist, early_stopping_rounds=10)

        auctrain, _ = get_cv_auc_bestiter(bst, dtrain)
        auctest, _ = get_cv_auc_bestiter(bst, dtest)
        
        if(auctest > bestauctest):
            bestauctest = auctest
            bestauctrain = auctrain
            bestcomb = i+1
            bestiter = bst.best_iteration
        
        print('GS_AUC_TRAINTEST,{},{},{},{},{},{}'.format(i+1, auctest, auctrain, auctrain-auctest, bst.best_iteration, 
               ','.join([str(elem) for elem in comb])))
        print('Best so far: Comb={}, Test={}, Train={}, Iter={}'.format(bestcomb, bestauctest, bestauctrain, bestiter))

def train_xgb_for_cv(dtrain, dtest, treecount):
    param = {'max_depth':max_depth, 'eta':eta, 'silent':silent, 'objective':objective, 
             'gamma':gamma, 'min_child_weight':min_child_weight, 'max_delta_step':max_delta_step, 
             'subsample':subsample, 'colsample_bytree':colsample_bytree,
             'alpha':alpha, 'lambda':lambdaparam, 'scale_pos_weight':scale_pos_weight, 'seed':0, 'eval_metric':'auc'}
    
    msg('starting cv. maxrounds={}, params: {}'.format(treecount, param))
    watchlist  = [(dtrain,'train'), (dtest,'eval')]
    bst = xgb.train(param, dtrain, treecount, watchlist, early_stopping_rounds=10)
    return bst

def do_cv():
    dtrain, dtest, featnames, featindices = get_cv_dmatrix()

    bst = train_xgb_for_cv(dtrain, dtest, treecount=num_round)

    msg('logging feature importances and saving xgb model')
    log_featimp_and_save_xgb(bst, featnames, featindices)
    
    msg('scoring')
    score_cv(bst, dtrain, dtest)
    
    msg('completed do_cv')
    
def do_real():
    dtrain, dtest, featnames, featindices, idtest = get_real_dmatrix()
    
    param = {'max_depth':max_depth, 'eta':eta, 'silent':silent, 'objective':objective, 
             'gamma':gamma, 'min_child_weight':min_child_weight, 'max_delta_step':max_delta_step, 
             'subsample':subsample, 'colsample_bytree':colsample_bytree,
             'alpha':alpha, 'lambda':lambdaparam, 'scale_pos_weight':scale_pos_weight, 'seed':0, 'eval_metric':'auc'}
    treecountreal = 20000

    msg('starting train. maxrounds={}, params: {}'.format(treecountreal, param))
    watchlist  = [(dtrain,'train')]
    bst = xgb.train(param, dtrain, treecountreal, watchlist)

    msg('generating and writing predictions')
    gen_and_write_preds(bst, dtest, idtest)

    msg('logging feature importances and saving xgb model')
    log_featimp_and_save_xgb(bst, featnames, featindices)

    msg('scoring')
    score_real(bst, dtrain)
    
    msg('completed do_real')
    
    
if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'cv':
            do_cv()
        elif sys.argv[1] == 'real':
            do_real()
        elif sys.argv[1] == 'xgbcv':
            do_xgbcv()
        elif sys.argv[1] == 'gs':
            do_gridsearch_cv()