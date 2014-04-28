'''
CS688 HW05: Restricted Boltzmann Machine for predicting missing data

Implementation of block Gibbs sampling / RBM training using mini-batch
stochastic gradient ascent

@author: Emma Strubell
'''
from __future__ import division
import numpy as np

# perform block Gibbs sampling given learned parameters for the given number
# of iterations
def block_gibbs_sample(w_p, w_b, w_c, d, k, iters, message="", verbose=False):
    print "Block Gibbs sampling %s" % (message)
    xs = np.empty((iters, d))
    hiddens = np.empty((iters, k))
    energies = np.empty(iters)
    # initialize hidden layer to random binary values
    hiddens[-1] = np.random.binomial(1, 0.5, k)
    for iter in range(iters):
        if(verbose): print "Sampling iteration %d" % (iter)

        # sample new visible units
        p_xs = sigmoid(w_c+np.dot(w_p,hiddens[iter-1]))
        xs[iter] = np.where(np.random.rand(d) < p_xs, 1.0, 0.0)
        
        # sample new hidden units
        p_hs = sigmoid(w_b+np.dot(w_p.transpose(),xs[iter]))
        hiddens[iter] = np.where(np.random.rand(k) < p_hs, 1.0, 0.0)
        
        # compute negative energy of this setting of variables
        energies[iter] = -np.sum(w_p*np.outer(xs[iter],hiddens[iter])) - np.dot(w_b,hiddens[iter]) - np.dot(w_c,xs[iter])
        
    return xs, energies

# train parameters for the given number of iteration using the given
# learning rate and regularization parameter
def train_rbm(data, t, k, b, c, alpha, lam):
    n_b, d = data.shape
    n_b //= b
    
    # initialize Gibbs chains to random binary values
    chains = np.random.binomial(1, 0.5, (c,k))
    xs = np.empty((c,d))
    
    # initialize params w_b, w_c, w_p
    w_b = np.random.normal(0.0, 0.01, k)
    w_c = np.random.normal(0.0, 0.01, d)
    w_p = np.random.normal(0.0, 0.01, (k,d))
    
    for iter in range(t):
        print "Training iteration %d/%d" % (iter+1, t)
        for batch in range(b):
            # compute positive gradient contribution for each instance in batch
            data_batch = data[batch*n_b:(batch+1)*n_b]
            g_wc_pos = np.sum(data_batch,axis=0)
            p_k = sigmoid(w_b+np.transpose(np.array(np.matrix(w_p)*data_batch.transpose())))
            g_wb_pos = np.sum(p_k,axis=0)
            g_wp_pos = np.array(np.matrix(p_k.transpose())*data_batch)
            
            # compute negative gradient contribution from each chain, sample states
            p_xs = sigmoid(w_c+np.transpose(np.array(np.matrix(w_p.transpose())*chains.transpose())))
            xs = np.where(np.random.rand(c,d) < p_xs, 1.0, 0.0)
            p_hs = sigmoid(w_b+np.transpose(np.array(np.matrix(w_p)*xs.transpose())))
            chains = np.where(np.random.rand(c,k) < p_hs, 1.0, 0.0)
            g_wc_neg = np.sum(xs,axis=0)
            p_k = sigmoid(w_b+np.transpose(np.array(np.matrix(w_p)*xs.transpose())))
            g_wb_neg = np.sum(p_k,axis=0)
            g_wp_neg = np.transpose(np.matrix(xs.transpose())*p_k)
            
            # take a gradient step for each parameter in the model
            w_c += alpha*(g_wc_pos/n_b - g_wc_neg/c - lam*w_c)
            w_b += alpha*(g_wb_pos/n_b - g_wb_neg/c - lam*w_b)
            w_p += alpha*(g_wp_pos/n_b - g_wp_neg/c - lam*w_p)
    return w_c, w_b, w_p, xs

def compute_marginals(w_c, w_b, w_p, data, missing_indices):
    xs = data
    for i in range(4):
        new_hs = sigmoid(w_b+np.transpose(np.array(np.matrix(w_p)*xs.transpose())))
        hiddens = np.where(new_hs > 0.5, 1.0, 0.0)
        new_xs = sigmoid(w_c+np.array(np.matrix(hiddens)*w_p))
        xs[missing_indices] = np.where(new_xs > 0.5, 1.0, 0.0)[missing_indices]
    
    # one last iteration, then return marginals
    new_hs = sigmoid(w_b+np.transpose(np.array(np.matrix(w_p)*xs.transpose())))
    hiddens = np.where(new_hs > 0.5, 1.0, 0.0)
    return sigmoid(w_c+np.array(np.matrix(hiddens)*w_p))

# Return log marginals for missing values in data, signified by -1s
def predict(w_c, w_b, w_p, data, missing_indices=None):
    # get missing indices
    if missing_indices == None:
        missing_indices = np.where(data == -1)
    
    # set missing values to 0
    data_masked = np.copy(data)
    data_masked[missing_indices] = 0
    
    # return log marginals of missing values
    return compute_marginals(w_c, w_b, w_p, data_masked, missing_indices)
    
# this is complete data; remove missing% of values, predict them
# and return prediction accuracy
def test(w_c, w_b, w_p, data, missing_proportion):
    # generate random indices to remove
    rows, cols = data.shape
    np.random.seed(0)
    
    # remove random values from data
    data_missing = np.copy(data)
    rands = np.random.rand(rows, cols)
    mask = np.where(rands < missing_proportion, 0.0, 1.0)
    data_missing *= mask
    
    # compute marginals
    missing_indices = np.where(mask == 0)
    marginals = predict(w_c, w_b, w_p, data_missing, missing_indices)
    prediction = np.where(marginals > 0.5, 1.0, 0.0)
    
    count_1s = 0
    total_1s = 0
    count_0s = 0
    total_0s = 0
    total = 0
    flat_mask = mask.flatten()
    flat_data = data.flatten()
    for i,p in enumerate(prediction.flatten()):
        if(flat_mask[i] == 0):
            if(flat_data[i] == 1):
                total_1s += 1
                if(p == 1):
                    count_1s += 1
            elif(flat_data[i] == 0):
                total_0s += 1
                if(p == 0):
                    count_0s += 1
            total += 1
    accuracy = (count_1s + count_0s)/total
    print "0s: %d / %d = %g" % (count_0s, total_0s, count_0s/total_0s)
    print "1s: %d / %d = %g" % (count_1s, total_1s, count_1s/total_1s)

    return accuracy

# sigmoid/logistic function: f(x) = 1/(1+exp(-x))
def sigmoid(x): return 1/(1+np.exp(-x))