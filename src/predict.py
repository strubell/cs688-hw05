'''
CS688 HW05: Restricted Boltzmann Machine for predicting missing data

@author: Emma Strubell
'''

import numpy as np
import load_data as load
import rbm

test_size = load.max_test_size

print "Loading %d/%d testing instances" % (test_size, load.max_test_size)
test_instances = load.load_data('test', test_size)

print "Loading model parameters"
t = 10  # number of training iterations
k = 10 # number of hidden units
b = 100 # number of batches of data cases
c = 100 # number of Gibbs chains
w_c = load.load_params("Wc-t%d-k%d-b%d-c%d" % (t, k, b, c))
w_b = load.load_params("Wb-t%d-k%d-b%d-c%d" % (t, k, b, c))
w_p = load.load_params("Wp-t%d-k%d-b%d-c%d" % (t, k, b, c))

# compute embeddings for train and test data
print "Predicting missing values"
marginals = np.log(rbm.predict(w_c, w_b, w_p, test_instances))
print "Log marginals:"
rows, cols = marginals.shape
for i in range(rows):
    for j in range(cols):
        if(test_instances[i,j] == -1):
            print "%g " % (marginals[i,j]),
    print
    
