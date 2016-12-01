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

NO_OF_DATASET_FILES_TO_READ = 10
    
def transformTarget(val):
    return 1 if val>0 else 0

countryPairFilenameList = glob("./dataset/*.csv")

# Get: no. of files << no. of rows in each file
shuffle(countryPairFilenameList)
countryPairFilenameList = countryPairFilenameList[:NO_OF_DATASET_FILES_TO_READ]

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
y = y.reshape(y.shape[0],1)

# Randomly sample and split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)


print 'Dataset loaded'

# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------

warnings.filterwarnings("ignore") # will hide deprecation warning

sigma = lambda x: 1 / (1 + T.exp(-x))

act = T.tanh

# sequences: x_t
# prior results: h_tm1, c_tm1
# non-sequences: W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xy, W_hy, W_cy, b_y
def one_lstm_step(x_t, h_tm1, c_tm1, W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xy, W_ho, W_cy, b_o, W_hy, b_y):
    i_t = sigma(theano.dot(x_t, W_xi) + theano.dot(h_tm1, W_hi) + theano.dot(c_tm1, W_ci) + b_i)
    f_t = sigma(theano.dot(x_t, W_xf) + theano.dot(h_tm1, W_hf) + theano.dot(c_tm1, W_cf) + b_f)
    c_t = f_t * c_tm1 + i_t * act(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_c) 
    o_t = sigma(theano.dot(x_t, W_xo)+ theano.dot(h_tm1, W_ho) + theano.dot(c_t, W_co)  + b_o)
    h_t = o_t * act(c_t)
    y_t = sigma(theano.dot(h_t, W_hy) + b_y) 
    return [h_t, c_t, y_t]
    

def sample_weights(sizeX, sizeY):
    values = np.ndarray([sizeX, sizeY], dtype=dtype)
    for dx in xrange(sizeX):
        vals = np.random.uniform(low=-1., high=1.,  size=(sizeY,))
        #vals_norm = np.sqrt((vals**2).sum())
        #vals = vals / vals_norm
        values[dx,:] = vals
    _,svs,_ = np.linalg.svd(values)
    #svs[0] is the largest singular value                      
    values = values / svs[0]
    return values  
    

n_in = 4 # input vector size
n_hidden = n_i = n_c = n_o = n_f = 20
n_y = 1 # output vector size

# initialize weights
# i_t and o_t should be "open" or "closed"
# f_t should be "open" (don't forget at the beginning of training)
# we try to archive this by appropriate initialization of the corresponding biases 

W_xi = theano.shared(sample_weights(n_in, n_i))  
W_hi = theano.shared(sample_weights(n_hidden, n_i))  
W_ci = theano.shared(sample_weights(n_c, n_i))  
b_i = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_i)))
W_xf = theano.shared(sample_weights(n_in, n_f)) 
W_hf = theano.shared(sample_weights(n_hidden, n_f))
W_cf = theano.shared(sample_weights(n_c, n_f))
b_f = theano.shared(np.cast[dtype](np.random.uniform(0, 1.,size = n_f)))
W_xc = theano.shared(sample_weights(n_in, n_c))  
W_hc = theano.shared(sample_weights(n_hidden, n_c))
b_c = theano.shared(np.zeros(n_c, dtype=dtype))
W_xo = theano.shared(sample_weights(n_in, n_o))
W_ho = theano.shared(sample_weights(n_hidden, n_o))
W_co = theano.shared(sample_weights(n_c, n_o))
b_o = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_o)))
W_hy = theano.shared(sample_weights(n_hidden, n_y))
b_y = theano.shared(np.zeros(n_y, dtype=dtype))

c0 = theano.shared(np.zeros(n_hidden, dtype=dtype))
h0 = T.tanh(c0)

params = [W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y, c0]

#first dimension is time

#input 
v = T.matrix(dtype=dtype)

# target
target = T.matrix(dtype=dtype)

# hidden and outputs of the entire sequence
[h_vals, _, y_vals], _ = theano.scan(fn=one_lstm_step, 
                                  sequences = dict(input=v, taps=[0]), 
                                  outputs_info = [h0, c0, None ], # corresponds to return type of fn
                                  non_sequences = [W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y] )

# Criss-entropy cost function chosen for multiclass classification
#cost =  T.mean((target - y_vals) ** 2) #-T.mean(target * T.log(y_vals)+ (1.- target) * T.log(1. - y_vals)) * T.abs_(target-y_vals)
cost = -T.mean(target * T.log(y_vals) + (1.- target) * T.log(1. - y_vals))

# learning rate
lr = np.cast[dtype](.1)
learning_rate = theano.shared(lr)

gparams = []
for param in params:
  gparam = T.grad(cost, param)
  gparams.append(gparam)

updates=[]
for param, gparam in zip(params, gparams):
    updates.append((param, param - gparam * learning_rate))
    
learn_rnn_fn = theano.function(inputs = [v, target],
                               outputs = cost,
                               updates = updates)
                               
predictions = theano.function(inputs = [v], outputs = y_vals)


#--------------------------------


nb_epochs=100
train_errors = []
  
def train_rnn():
    print "Started training"
    for x in range(nb_epochs):
        error = 0.
        for j in range(len(X_train)):
            index = np.random.randint(0, len(X_train))
            i = np.matrix(X_train[index,:])
            o = np.matrix([y_train[index]])
            train_cost = learn_rnn_fn(i, o)
            error += train_cost
        train_errors.append(error)
        print "epoch:%d/%d loss:%.4f" %(x+1,nb_epochs, error)
        
train_rnn()

plt.plot(range(len(train_errors)), train_errors, 'b-')
plt.title('Training Curve')
plt.xlabel('Epoch(s)')
plt.ylabel('Error')
plt.show()

round_vec = np.vectorize(round)
y_pred = round_vec(predictions(X_test))

acc = metrics.accuracy_score(y_test, y_pred)
prec = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1score = metrics.f1_score(y_test, y_pred)

confmat = metrics.confusion_matrix(y_test, y_pred)

print 'Confusion matrix:'
print confmat
print '\nAccuracy: %.2f%%' %(acc * 100)
print 'Precision score: %.2f%%'%(prec * 100)
print 'Recall score %.2f%%'%(recall * 100)
print 'F1 Score: %.2f%%'%(f1score * 100)