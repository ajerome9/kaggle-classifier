import itertools

import numpy as np
import pandas as pd

import conf
from log import msg
from const import DATECOLS, BOOLCOLS, CATENCCOLS, DROPCOLS

def gen_zipcode_feats(xtrain, xtest):
    def gen_zip_prefixes(df):
        df.loc[:, 'ZIP1'] = (df.loc[:, 'VAR_0241']/10000).astype('int').astype('str')
        df.loc[:, 'ZIP123'] = (df.loc[:, 'VAR_0241']/100).astype('int').astype('str')
        df.loc[:, 'ZIP45'] = (df.loc[:, 'VAR_0241']%100).astype('int')
        return df

    return gen_zip_prefixes(xtrain), gen_zip_prefixes(xtest)
    
def gen_fancy_feats(xtrain, xtest):
    msg('    adding feat: zipcode feats')
    xtrain, xtest = gen_zipcode_feats(xtrain, xtest)
    
    # feat: are both states the same (guessing that states might be perm residence, temp residence)
    msg('    adding feat: same state')
    xtrain.loc[:, 'SAME_STATE'] = (xtrain.VAR_0237==xtrain.VAR_0274).astype(int)
    xtest.loc[:, 'SAME_STATE'] = (xtest.VAR_0237==xtest.VAR_0274).astype(int)

    # diff between all date ordinals
    msg('    adding feat: date ordinal diffs')
    dateords = [d+'_ORD' for d in DATECOLS]
    for comb in itertools.combinations(dateords, 2):
        newcol = 'DIFF_'+comb[0][4:8]+'_'+comb[1][4:8]
        xtrain.loc[:, newcol] = xtrain.loc[:, comb[0]] - xtrain.loc[:, comb[1]]
        xtest.loc[:, newcol] = xtest.loc[:, comb[0]] - xtest.loc[:, comb[1]]

    return xtrain, xtest
    