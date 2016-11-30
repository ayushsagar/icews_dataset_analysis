#%matplotlib inline
from glob import glob
from random import shuffle
import warnings
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cross_validation, metrics
dtype=theano.config.floatX='float64'
    
def transformTarget(val):
    return 1 if val>0 else 0

countryPairFilenameList = glob("./dataset/*.csv")

# Get: no. of files << no. of rows in each file
shuffle(countryPairFilenameList)
countryPairFilenameList = countryPairFilenameList[:10]

# Concatenate one-by-one
X = pd.DataFrame()
for countryPairFilename in countryPairFilenameList:
    x = pd.read_csv(countryPairFilename)
    X = X.append(x,ignore_index=True)
    print 'reading', countryPairFilename

# Assuming time-invariance of pattern being learnt
X = X.drop(['year'],axis=1) # drop redundant date attr

# Split input and target
y = X['mat_conflict']
X = X.drop(['mat_conflict'],axis=1)

# Transform target
y = y.apply(transformTarget)

# shift target columns up by 1 row, and input columns down by 1 row for prediction task 
X = X.ix[:(len(X)-2),:]
y = y.ix[1:,]

#X = X/X.max().astype(dtype)
X = X.as_matrix()
y = y.as_matrix()

# Randomly sample and split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)

print 'Dataset loaded'