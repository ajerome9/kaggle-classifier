import os
import pickle

import numpy as np
from sklearn.cross_validation import train_test_split

import conf
from log import msg
from run_xgboost import load_cv_data, gen_features_cvreal

class Data:
    """Holds data for springleaf"""
    def __init__(self, xtrain, xtest, ytrain, ytest, allfeatnames, featindices):
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
        
        fn = []
        for i, featidx in enumerate(featindices):
            fn.append(allfeatnames[featidx])
        self.featnames = fn
        
    def __repr__(self):
        #self.ytest = None
        return 'xtrain: {}, xtest: {}, ytrain: {}, ytest: {}, featnames: {}'.format(self.xtrain.shape, self.xtest.shape, 
                    self.ytrain.shape, self.ytest.shape if self.ytest is not None else None, len(self.featnames))

    def filter(self, withcols, removecols):
        xtrain, xtest, featnames = self.xtrain, self.xtest, self.featnames
            
        if removecols is not None:
            withcols = np.delete(np.arange(xtrain.shape[1]), removecols)
            
        if withcols is not None:
            self.xtrain = xtrain[:,withcols]
            self.xtest = xtest[:,withcols]
            self.featnames = [featnames[i] for i in withcols]

        return self

def get_cvdata(withcols=None, removecols=None):
    if os.path.exists(conf.cvdata):
        msg('reading cvdata: {}'.format(conf.cvdata))
        cvdata = pickle.load(open(conf.cvdata, 'rb'))
    else:
        msg('no cvdata file {}. will load data and gen feats'.format(conf.cvdata))
        msg('loading data')
        X, y = load_cv_data()
        
        msg('train_test_split')
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.75, random_state=0)
        xtrain, xtest = xtrain.copy(), xtest.copy()
        msg('xtrain: {}, xtest: {}, ytrain: {}, ytest: {}'.format(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))
    
        xtrain, xtest, allfeatnames, featindices = gen_features_cvreal(xtrain, xtest)
        msg('after feat gen: xtrain: {}, xtest: {}, ytrain: {}, ytest: {}'.format(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))
        
        cvdata = Data(xtrain, xtest, ytrain, ytest, allfeatnames, featindices)
        
        msg('pickling cvdata for reuse')        
        pickle.dump(cvdata, open(conf.cvdata, 'wb'))
                
    return cvdata.filter(withcols, removecols)


if __name__ == '__main__1':
    removecols = list(range(5, 2590))
    removecols.append(0)
    cvdata = get_cvdata(removecols=removecols)
    print(cvdata, cvdata.featnames)
    
    cvdata = get_cvdata(withcols=[0,2])
    print(cvdata, cvdata.featnames)
    
    cvdata = get_cvdata()
    print(cvdata)