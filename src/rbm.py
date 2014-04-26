'''
CS688 HW05: Restricted Boltzmann Machine for binary classification

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

# embed the given data instances into the lower-dimensional space defined
# by the given model parameters
def compute_embeddings(w_c, w_b, w_p, data):
    return sigmoid(w_b+np.transpose(np.array(np.matrix(w_p)*data.transpose())))

# sigmoid/logistic function: f(x) = 1/(1+exp(-x))
def sigmoid(x): return 1/(1+np.exp(-x))