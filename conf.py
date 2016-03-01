import os

IS_AWS = False
AWS_PATH = '/awsdata/'
LOCAL_PATH = '/Users/aa/lab/kaggle/springleaf/data/'

if os.path.exists(AWS_PATH):
    basepath = AWS_PATH
    IS_AWS = True
else:
    basepath = LOCAL_PATH

trainf = basepath + 'train.csv'
testf = basepath + 'test.csv'
dtypesf = basepath + 'dtypes_dict.pkl'
trainpkl = basepath + 'train.pkl'
testpkl = basepath + 'test.pkl'

xgbcvdmatrix = basepath + 'xgbcvdmatrix.dmx'
cvdtrain = basepath + 'cvdtrain.dmx'
cvdtest = basepath + 'cvdtest.dmx'
cvfeats = basepath + 'cvfeats.pkl'

realdtrain = basepath + 'realdtrain.dmx'
realdtest = basepath + 'realdtest.dmx'
realfeats = basepath + 'realfeats.pkl'

cvdata = basepath + 'cvdata.pkl'
cvcorrelatedcols = basepath + 'cvcorrelatedcols.pkl'

realsnowdata = basepath + 'realsnowdata.pkl'
cvsnowdata = basepath + 'cvsnowdata.pkl'

# how many rows to perform CV on
cvrows = None if IS_AWS else 1000