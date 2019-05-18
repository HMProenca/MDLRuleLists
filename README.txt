#######Supplement for:
#
# Hugo M. Proenca, Matthijs van Leeuewen (2019) 
#"Interpretable multiclass classification by MDL-based rule lists"
#"Interpretable classifiers using rules and Bayesian analysis: Building a better 
# https://arxiv.org/abs/1905.00328
#
#Version 1.0, May 14, 2019
# 
#
####README
#
# This code implements the Interpretable multiclass classification by 
# MDL-based rule lists algorithm as described in the paper and all its
# experiments.
#
# This code requires the external frequent itemset mining package "PyFIM," 
# available at http://www.borgelt.net/pyfim.html
#
# It follows a short description of each file:
######################################################################################
# filename :  mdl_rulelists.py  (more details included inside the file)
# 
# The algorithm performs multiclass classification with binary variables.
#
# ## TO RUN THIS CODE IN A DIFFERENT DATASET:
#
# The user needs to define the "filename", and the "minsupp" and "maxlen" of the rules mined
#
# ## EXAMPLE: 
# 
# import mdl_rulelists as mdlrl
# filename = "./datasetexample/breast.csv" 
# minsupp = 5 # it is a 5% minimum support threshold
# maxlen = 4
# data_it, cl,item2class = mdlrl.binary2itemsets(filename) 
# data_train, data_test = divide "data_it" according to your intereststs
# modelprob,modelraw,lfinal,lorig,nfreqp = mdlrl.mdl_rulelist(data_train, cl,minsupp,maxlen)
# pred, probvector, probmatrix,RULEactivated = mdlrl.prediction(data_test,modelprob,cl,item2class)
# ## in case a prediction needs to be done one can use mdlrl.prediction
# ## where "datatest" is a dataset without class labels 
# ## (it is actually indiferent if it has class labels or not)
#
######################################################################################
# filename :  runpaperexperiments.py  (more details included inside the file)
# 
# This file executes all the experiments described in the paper and returns all
# the results to the xps folder
#
# This file uses mdl_rulelists.py to run its experiments
