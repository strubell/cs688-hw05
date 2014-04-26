#!/bin/bash

./svm_multiclass_learn -c 1000000 -v 2 -e 1 data/train-embeddings models/svm-embeddings-model
