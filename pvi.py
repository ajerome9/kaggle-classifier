import numpy as np
import pandas as pd
import pickle
import os
import xgboost as xgb
from const import DATECOLS, BOOLCOLS, CATENCCOLS, DROPCOLS
from log import msg
import conf
import nandict

def cleancols(X):
    X = X.drop(DATECOLS, axis=1, errors='ignore')
    X = X.drop(BOOLCOLS, axis=1, errors='ignore')
    X = X.drop(CATENCCOLS, axis=1, errors='ignore')
    X = X.drop(DROPCOLS, axis=1, errors='ignore')
    X = X.drop(['VAR_0404', 'VAR_0493'], axis=1, errors='ignore')
    return X
    
def read_and_merge_test_train(trainf, testf):
    msg('    read_and_merge_test_train')
    msg('        reading train')
    strain = pd.read_csv(conf.basepath+trainf, dtype=pickle.load(open(conf.dtypesf, 'rb')), index_col='ID')
    msg('        reading test')
    stest = pd.read_csv(conf.basepath+testf, dtype=pickle.load(open(conf.dtypesf, 'rb')), index_col='ID')
    msg('        train={}, test={}'.format(len(strain), len(strain)))
    msg('        dropping target column from train')
    strain = strain.drop('target', axis=1)
    msg('        appending test to train')
    scom = strain.append(stest, verify_integrity=True)
    msg('        dropping non-numerical columns')
    scom = cleancols(scom)
    msg('        size of merged array = {}'.format(scom.shape))
    return scom

def make_pred_dict(df, c, desc):
    """desc[0]
       - 'c': XGBClf
       - 'r': XGBReg
       desc[1]
       - Null-like values (eg: -99999)
    """
    msg('make_pred_dict')
    # columns other than the one for which we are trying to predict
    othercols = list(df.columns)
    othercols.remove(c)
    
    nv = desc[1]
    
    msg('    generating criteria for row selection')
    # either the value is one of the null-like-values or the column is really null
    if nv:
        criteria = ((df.loc[:, c].isin(nv)) | df.loc[:, c].isnull())
    else:
        criteria = (df.loc[:, c].isnull())
    
    msg('    selecting train/test rows')
    xtrain = df.loc[~criteria, othercols].values
    xtestwithidx = df.loc[criteria, othercols]
    xtest = xtestwithidx.values
    ytrain = df.loc[~criteria, c].values
    
    #model = xgb.XGBClassifier(seed=0) if desc[0]=='c' else xgb.XGBRegressor(seed=0)
    model = xgb.XGBRegressor(seed=0)
    msg('    fitting model : {}'.format(model))
    model.fit(xtrain, ytrain)
    msg('    doing pred')
    ypred = model.predict(xtest)

    # round the preds
    ypred = np.round(ypred)
    
    msg('    preparing return dict')
    return {k:v for k,v in zip(xtestwithidx.index, ypred)}

def do_pvi(df, c, desc):
    """Do Predicted Value Imputation of the given column in 
    the df. Values to be imputed are those in desc[1]
    as well as real nulls.
    """
    msg('* PVI for column = {}, values = {}'.format(c, desc))
    pdict = make_pred_dict(df, c, desc)
    msg('    pickling results')
    pickle.dump(pdict, open(conf.basepath+'pdicts/'+c, 'wb'))
    msg('==================================================')

def do_pvi_for_all(nandict):
    df = read_and_merge_test_train('train.csv', 'test.csv')
    os.makedirs(conf.basepath+'pdicts/', exist_ok=True)
    for c, desc in nandict.items():
        do_pvi(df, c, desc)
    msg('Completed PVI')
    

if __name__ == '__main__':
    do_pvi_for_all(nandict.nandict)