'''
CS688 HW05: Restricted Boltzmann Machine for binary classification

@author: Emma Strubell
'''

import load_data as load
import rbm

train_size = load.max_train_size
test_size = load.max_test_size

print "Loading %d/%d training instances" % (train_size, load.max_train_size)
train_instances = load.load_data('train', train_size)

print "Loading %d/%d training labels" % (train_size, load.max_train_size)
train_labels = load.load_labels('train', train_size)

print "Loading %d/%d testing instances" % (test_size, load.max_test_size)
test_instances = load.load_data('test', test_size)

print "Loading %d/%d testing labels" % (test_size, load.max_test_size)
test_labels = load.load_labels('test', test_size)

print "Loading model parameters"
t = 10  # number of training iterations
k = 10 # number of hidden units
b = 100 # number of batches of data cases
c = 100 # number of Gibbs chains
w_c = load.load_params("Wc-t%d-k%d-b%d-c%d" % (t, k, b, c))
w_b = load.load_params("Wb-t%d-k%d-b%d-c%d" % (t, k, b, c))
w_p = load.load_params("Wp-t%d-k%d-b%d-c%d" % (t, k, b, c))

# compute embeddings for train and test data
print "Computing embeddings"
train_embeddings = rbm.compute_embeddings(w_c, w_b, w_p, train_instances)
test_embeddings = rbm.compute_embeddings(w_c, w_b, w_p, test_instances)
    
# write train and test embeddings as labeled feature vectors for SVMLight
print "Writing labeled feature vectors (embeddings) for SVMLight"
load.write_svmlight(train_embeddings, train_labels, "train-embeddings")
load.write_svmlight(test_embeddings, test_labels, "test-embeddings")

# write train and test binary vectors as labeled feature vectors for SVMLight
print "Writing labeled feature vectors (raw) for SVMLight"
load.write_svmlight(train_instances, train_labels, "train-raw")
load.write_svmlight(test_instances, test_labels, "test-raw")
