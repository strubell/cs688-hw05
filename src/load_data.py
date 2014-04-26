'''
CS688 HW05: Restricted Boltzmann Machine for binary classification

Utilities for loading data

@author: Emma Strubell
'''

import numpy as np

max_train_size = 10000
max_test_size = 5000

data_dir = "../data/"
model_dir = "../models/"
word_file = model_dir+"words.txt"

# load trained parameter (model) files
def load_params(type): return np.loadtxt("%s%s.txt" % (model_dir, type))

# load data into numpy array, ignoring missing values (test set)
def load_data(type, num_lines=0): 
    raw_data = np.genfromtxt("%s%s.txt" % (data_dir, type), usecols=range(101), skip_footer=(max_train_size if type == "train" else max_test_size)-num_lines)
    return np.array(filter(lambda x: x[100] != -1, raw_data))[:,:-1]

# load data labels into numpy array, ignoring missing values (test set)
def load_labels(type, num_lines=0):
    raw_labels = np.genfromtxt("%s%s.txt" % (data_dir, type), usecols=(100,101,102,103), skip_footer=(max_train_size if type == "train" else max_test_size)-num_lines)
    labels = [np.where(raw_labels[i] == 1) for i in range(len(raw_labels))]
    labels = map(lambda x: x[0][0]+1, filter(lambda x: x[0] != -1, labels))
    return labels

# load feature word labels into Python list
def load_words(): return readlines(word_file)

# write trained parameter (model) files
def write_params(vec, name): np.savetxt(model_dir+name+".txt", vec)

# write labeled feature vectors in SVMlight sparse format
def write_svmlight(features, labels, fname):
    file = open(data_dir+fname, 'w')
    for feats,label in zip(features,labels):
        file.write("%d " % (label))
        for idx,feat in enumerate(feats):
            if(not feat == 0.0):
                file.write("%d:%g " % (idx+1,feat))
        file.write("\n")
    file.close()