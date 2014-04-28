'''
CS688 HW05: Restricted Boltzmann Machine for predicting missing data

@author: Emma Strubell
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import load_data as load
import plotter
import rbm

train_size = load.max_train_size
test_size = load.max_test_size
train_split = 60
 
train_full_model = True

print "Loading %d/%d training instances" % (train_size, load.max_train_size)
train_instances = load.load_data('train', train_size)
train_portion, _ = load.get_split(train_instances, train_split)

t = 100  # number of training iterations
k = 10 # number of hidden units
b = 100 # number of batches of data cases
c = 100 # number of Gibbs chains
alpha = 0.5  # step size
lam = 0.0001 # regularization param

print "Training model on %d%% of data for validation..." % (train_split)
w_c, w_b, w_p, results = rbm.train_rbm(train_portion, t, k, b, c, alpha, lam)
# print "w_c:", w_c
# print "w_b:", w_b
# print "w_p:", w_p
load.write_params(w_c, "Wc-t%d-k%d-b%d-c%d-test" % (t, k, b, c))
load.write_params(w_b, "Wb-t%d-k%d-b%d-c%d-test" % (t, k, b, c))
load.write_params(w_p, "Wp-t%d-k%d-b%d-c%d-test" % (t, k, b, c))

if train_full_model:
    print "Training full model t=%d, k=%d, b=%d, c=%d, alpha=%g, lambda=%g" % (t, k, b, c, alpha, lam)
    w_c, w_b, w_p, results = rbm.train_rbm(train_instances, t, k, b, c, alpha, lam)
    load.write_params(w_c, "Wc-t%d-k%d-b%d-c%d" % (t, k, b, c))
    load.write_params(w_b, "Wb-t%d-k%d-b%d-c%d" % (t, k, b, c))
    load.write_params(w_p, "Wp-t%d-k%d-b%d-c%d" % (t, k, b, c))