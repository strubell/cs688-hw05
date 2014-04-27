'''
CS688 HW05: Restricted Boltzmann Machine for predicting missing data

@author: Emma Strubell
'''

import load_data as load
import rbm

train_size = load.max_train_size
train_split = 0.6
missing = 0.2

print "Loading %d/%d training instances" % (train_size, load.max_train_size)
train_instances = load.load_data('train', train_size)

print "Performing random %d/%d split" % (train_split*100, 100-train_split*100)
_, test_portion = load.get_split(train_instances, train_split)
print "Using %d test instances" % (len(test_portion))

print "Loading model parameters"
t = 100  # number of training iterations
k = 10 # number of hidden units
b = 100 # number of batches of data cases
c = 100 # number of Gibbs chains
w_c = load.load_params("Wc-t%d-k%d-b%d-c%d-test" % (t, k, b, c))
w_b = load.load_params("Wb-t%d-k%d-b%d-c%d-test" % (t, k, b, c))
w_p = load.load_params("Wp-t%d-k%d-b%d-c%d-test" % (t, k, b, c))

# compute embeddings for train and test data
print "Predicting missing values"
accuracy = rbm.test(w_c, w_b, w_p, test_portion, missing)
print "Accuracy:", accuracy
