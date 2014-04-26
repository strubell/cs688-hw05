'''
CS688 HW05: Restricted Boltzmann Machine for binary classification

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
 
print "Loading %d/%d training instances" % (train_size, load.max_train_size)
train_instances = load.load_data('train', train_size)

t = 10  # number of training iterations
k = 10 # number of hidden units
b = 100 # number of batches of data cases
c = 100 # number of Gibbs chains
alpha = 0.1  # step size
lam = 0.0001 # regularization param

w_c, w_b, w_p, results = rbm.train_rbm(train_instances, t, k, b, c, alpha, lam)

# write model to files
load.write_params(w_c, "Wc-t%d-k%d-b%d-c%d" % (t, k, b, c))
load.write_params(w_b, "Wb-t%d-k%d-b%d-c%d" % (t, k, b, c))
load.write_params(w_p, "Wp-t%d-k%d-b%d-c%d" % (t, k, b, c))

