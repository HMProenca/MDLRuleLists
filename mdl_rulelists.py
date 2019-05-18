#######Supplement for:
#
# Hugo M. Proenca, Matthijs van Leeuewen (2019) 
# "Interpretable multiclass classification by MDL-based rule lists"
# "Interpretable classifiers using rules and Bayesian analysis: Building a better 
# https://arxiv.org/abs/1905.00328
#
# Version 1.0, May 14, 2019
# 
#
####README
#
#This code implements the Interpretable multiclass classification by 
# MDL-based rule lists algorithm as described in the 
#paper. All the datasets used in the paper are included with this file.
#
# This code requires the external frequent itemset mining package "PyFIM," 
# available at http://www.borgelt.net/pyfim.html
#
# The algorithm performs multiclass classification with binary variables.
#
# IMPORTANT NOTE: In the paper we only deal with rules of the form x2=1 
# where variable 2 takes the value 1. We never deal with negative conditions such
# as x2=0. We deal with the negative conditions by creating another variable, 
# for example x3 = negative (x2), in the case of binary variables. In the case
# of categorical variables we only create a dummy separation as this makes sure
# that all the cases are represented in the form xi=1. 
# Also note that in terms of encoding this is exactly the same as taking into account
# for the encoding of the negative condition, thus we selected this way as it is more clean.
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
####################################################################################	
#                                  ## FUNCTIONS ##
####################################################################################
#
# - data_it, cl, item2class = mdlrl.binary2itemsets(filename)
# INPUT:
# filename - the file that contains the .csv file with the data in binary format,i.e.,
#            the variables can either be 0 or 1 and the class label (last column) can
#            take a range of integers
# OUTPUT:
# data_it    - (a list of sets) the data in itemset format, i.e., it has the number 
#              of the column from 1 until Number_Variables. 
#              Example: having 4 in the 3rd set means that variable4=1 in the 3rd instance
#              [{1,2,3,4,10},{1,4,11}...]
# cl         - the class labels in itemset format in a list of frozensets
#              Example cl = [frozenset([10]),frozenset([11])]
# item2class - a dictionary with the equivalence between the original class labels
#              the itemset class labels
#              Example from "breast" dataset: 
#              item2class = {frozenset({20}): 1, frozenset({19}): 0}
#
####################################################################################
#
# - modelprob,modelraw,lfinal,lorig,nfreqp = mdlrl.mdl_rulelist(data_train, cl,minsupp,maxlen)
#
# INPUT:
# data_train - dataset in itemset format - a list of sets
#              Example: [{1,2,3,4,10},{1,4,11}...]
# cl         - the class labels in itemset format in a list of frozensets
#              Example cl = [frozenset([10]),frozenset([11])]
# minsupp    - integer value of the minimum support threshold applied
#              example: minsupp = 5 -> it means a 5% threshold will be applied
# maxlen     - integer value of the maximum number of items in a rule
#            - example: maxlen = 4 -> maximum rule size no bigger than 4
# OUTPUT:
# modelprob  - model in probability format -  on the antecedent the model has the 
#              variables/items used and on the consequent it has probabilities per class.
#              Technically : a dictionary with keys as integer number of the rules. 
#              the rules are in the order in which they are activated endind in the default rule
#              Example from "breast" dataset:
#              defaultdict(<class 'dict'>, 
#              {0: {frozenset({20}): 0.8536585365853658,        # probability for class 20
#                   frozenset({19}): 0.14634146341463414,       # probability for class 19
#                   'cl': frozenset({20}),                      # majority class
#                   'p': frozenset({4, 12, 6, 14})}             # pattern with vars 4,12,6 and 14 =1
#               1: {frozenset({20}): 0.023255813953488372, 
#					frozenset({19}): 0.9767441860465116, 
#                   'cl': frozenset({19}), 
#                   'p': frozenset()}})
# modelraw    - model in the raw format, with the support per rule, length of the enconding at the 
#               moment the rule was added, and other constants used during the program
#               Example for "breast" dataset: 
#               defaultdict(<class 'dict'>, 
#              {0: {'lm': 21.266554420960116, 
#                   frozenset({19, 20}): 244, 
#                   frozenset({20}): 209, 
#                   frozenset({19}): 35, 
#                   'const1': -0.0, 
#                   'const2': 61.897354823361184, 
#                   'ld': 210.78872105113896, 
#                   'p': frozenset({4, 12, 6, 14})}, 
#              1: {'lm': 0, 
#                  frozenset({19, 20}): 385, 
#                  frozenset({20}): 8, 
#                  frozenset({19}): 377, 
#                  'const1': -0.0, 
#                  'ld': 589.0626969633095, 
#                  'const2': 589.0626969633095, 
#                  'p': frozenset()}})
# lfinal      - length of the final encoding
# lorig       - length of the dataset using just one default rule (equal to the priors per class)
# nfreqo      - number of frequent itemsets or patterns
# 
####################################################################################
#
# - pred, probvector, probmatrix,RULEactivated = mdlrl.prediction(data_test,modelprob,cl,item2class)
#
# INPUT:
# data_test - dataset in itemset format - a list of sets
#              Example: [{1,2,3,4,10},{1,4,11}...]
# model     - model in the probability format mentioned before
# cl         - the class labels in itemset format in a list of frozensets
#              Example cl = [frozenset([10]),frozenset([11])]
# OUTPUT: 
# pred      - the class label predictided by the rule list
# probvector- the probability with which the class label was predicted
# probmatrix- the probabilities for each class label for the respective instance
# RULEactivated- the rule activated and all its information
#
####################################################################################
#
#
# CODE:
import gc
import time
import numpy as np
from collections  import defaultdict
from math  import log, ceil, floor
from fim import fpgrowth
from scipy.special  import gammaln
import copy
from sklearn.preprocessing import label_binarize
from sklearn.metrics  import roc_auc_score as aucscore
from sklearn.metrics import accuracy_score
# the code to make the rule lists
def mdl_rulelist(dataset, cl,minsupp,maxlen):
    # load the data
    up_st = 0
    ep = 1
    rows = len(dataset)
    allcl = frozenset().union(*cl)
    l_universal = {key: universal_int(key) for key in range(0,500)}
    #print(rows)
    l2 = log(2)
    l_gammaln = {key: gammaln(key)/l2 for key in range(0,rows+len(allcl)+1)}
    #l_gammaln = [gammaln(key) for key in range(0,200)]
    # Make singleton Code table
    singletons = singleton_table(dataset) 
    single_input = [s for s in singletons if not(s <= allcl)]# without classes
    model = empty_model(dataset,cl,allcl,ep)
    itemsets = getfrequentitems(dataset,cl,minsupp,maxlen)
    supp, t_id,score,unif,newpat,bestscore = countocurrences(dataset,cl,allcl,\
                                                single_input,itemsets,model,ep,minsupp,l_universal,l_gammaln)
    nfreqp = len(single_input)+len(itemsets)
    # Make dictionary of 10 first logarithms of the universal code for integers
    # initial length
    #Initial score
    lmodl,nr = 0,1
    delta=1
    oldl = model[0]["ld"]
    count = 0
    while delta > 0:
        # find max score! 
        if bool(score): # returns true if score is non empty
            if bestscore < 0: break
        else:
            break
        model = addrule(newpat,supp,cl,allcl,model,ep,unif,l_universal)
        nr = len(model)
        auxl = model[nr-1]["lm"] + model[nr-1]["ld"]
        delta = oldl -auxl
        newpat,bestscore = update_dataset(newpat,supp,t_id,score,cl,allcl,model,ep,unif,l_universal,l_gammaln)
        oldl = auxl
        count +=1 
    
    del supp, t_id,score, dataset,l_universal
    # import gc
    gc.collect()
    lorig = model[0]["ld"]
    lfinal = oldl
    model = modelfinalform(model,cl)
    modelprob = model_prob(model,cl,allcl,ep)
    return modelprob,model,lfinal,lorig,nfreqp
# transform the model in its final form with the default rule at the end instead of begining
def modelfinalform(model,cl):
    modelnew = defaultdict(dict)
    nr = len(model)
    nc = len(cl)
    for r in range(nr):
        if r != nr-1:
            modelnew[r]=model[r+1]
        else:
            modelnew[r]=model[0]
    return modelnew   
# compute the universal code for integers	
def universal_int(value):
    # Computes the universal code for integers with recursive log : BRUTE FORCE
    const =  2.865064
    logsum = log(const,2)
    cond = True # condition
    if value == 0:
        logsum = 0
    else:
        while cond: # Recursive log
            value = log(value,2)
            cond = value > 0.000001
            if value < 0.000001: 
                break
            logsum += value
    return logsum	
# makes a lists with the frozensets of the singletons (incuding class labels)	
def singleton_table(data):
    items = set.union(*data)
    # ST = [set([i]) for i in items if i not in cl]
    singleton = [frozenset([i]) for i in items]
    return singleton	
# returns the support of each item plus the instances where they appear
# STANDARD COVER 
def std_cover(CT, data, cl):
    # the standard cover function returns the support and the instance ids of the items contained in CT
    # CT = [{1,2},...,{1}]
    # data = [[0,1...],...,[0,1...]]
    # flag = 2 -> complex standard cover  with "usage" and t_usage
    allcl = frozenset().union(*cl)
    supp = defaultdict(lambda: defaultdict(int))
    # import collections
    supp = defaultdict(lambda: defaultdict(int))
    t_id = defaultdict(lambda: defaultdict(set))
    t_support = [[[] for j in range(len(CT))] for i in range(len(cl)+1)]  # transactions in which the items are used
    for itr, tr in enumerate(data):
        for icode, code in enumerate(CT):
            if code <= tr:  # if c is in t
                t_support[0][icode].append(itr)
                supp[allcl][code] += 1
                for ic, c in enumerate(cl):
                    if c <= tr:
                        supp[c][code]+= 1
                        t_support[ic + 1][icode].append(itr)
                            
    for irow,row in enumerate(t_support):
        for icol,col in enumerate(row):
            if irow == 0:
                t_id[allcl][CT[icol]] = set(col)    
            else:
                t_id[cl[irow-1]][CT[icol]] = set(col)  # I should change the row to the right class and col to the right column
                                            # Except the first (zero) that represents the complete cover!
    return supp, t_id  # supp = [699,699,...,0]
### functions to compute the length or gain of the encoding	regarding data
def length_data(model,cl,allcl,epsilon):
    l2 = log(2) # natural logarithm of 2
    nr = len(model) # number of rules
    nc = len(cl) # number of classes
    ld = nr*nc*gammaln(epsilon)/l2 - nr*gammaln(epsilon*nc)/l2    
    for r in range(nr):
        ld += -sum([gammaln(model[r][c]+epsilon)/l2 for c in cl])
        ld += gammaln(model[r][allcl]+epsilon*nc)/l2    
    return ld
def delta_data(model,pat,supp,cl,allcl,epsilon,l_gammaln):
    # computes the absolute data gain of adding "pat"
    l2 = log(2)
    nc= len(cl)
    dld = model[len(model)-1]['const1'] +model[len(model)-1]['const2']
    # new pattern contribution
    dld += +sum([l_gammaln[supp[c][pat]+epsilon]  for c in cl])
    dld += -(l_gammaln[supp[allcl][pat]+nc*epsilon])
    # new empty rule contribution
    dld += +sum([l_gammaln[model[0][c]-supp[c][pat]+epsilon] for c in cl])
    dld += -(l_gammaln[model[0][allcl]-supp[allcl][pat]+nc*epsilon])            

    return dld	
def delta_data_const(model,cl,allcl,epsilon):
    # computation of a constant that appears in all 
    # the absolute gain comptutations, namely regarding the epsilons
    # This is K1
    l2 = log(2) # natural logarithm of 2
    nc= len(cl)
    const1 = -(nc*gammaln(epsilon)/l2 - gammaln(epsilon*nc)/l2)
    # This is K2
    const2 = -sum([gammaln(model[0][c]+epsilon)/l2 for c in cl])
    const2 += gammaln(model[0][allcl]+nc*epsilon)/l2
    return const1, const2	
### functions to compute the length of the encoding regarding the model 
def length_model(model,cl,allcl,unif,l_universal):
    # computes the length of the model encoding
    # l_universal -> dictionary of universal codes
    n_rules = len(model)
    # Length of the number of rules
    l_rules = l_universal[n_rules-1] # we do not count the empty rule!
    # Length of the number of patterns and of the items
    l_n_p = 0 # number of patterns
    l_s = 0 # singleton encoding
    l_c = 0
    for j in range(1,n_rules):
        pat = model[j]['p']
        l_n_p +=  l_universal[len(pat)]
        l_s   += -len(pat)*log(unif,2)
    lm = l_rules + l_n_p + l_s +l_c
    return lm
def delta_model(nr,pat,cl,allcl,unif,l_universal):
    # computes the absolute gain of the model encoding by adding pat
    # l_universal -> dictionary of universal codes
    # st -> singleton code table
    # pat -> pattern in the following form (item1,item2,...,itemN)
    # nr - number of rules
    # currently it has nr-1 (without the empty one) and it will have nr!
    l_rules = l_universal[nr-1] - l_universal[nr] # we do not count the empty rule! 
    # Length of the number of patterns and of the items
    l_c = 0
    l_n_p =  - l_universal[len(pat)]
    l_s   = len(pat) *log(unif,2) 
    dlm = l_rules + l_n_p + l_s + l_c
    return dlm	
def deltascore_compute(nrules,pattern,cl,allcl,model,supp,epsilon,usage,unif,l_universal,l_gammaln):
    # computes the normalized gain of adding "pattern" to "model"
    score =(delta_model(nrules,pattern,cl,allcl,unif,l_universal) \
            + delta_data(model,pattern,supp,cl,allcl,epsilon,l_gammaln))/usage
    return score	
# create an empty model with just a default rule	
def empty_model(dataset,cl,allcl,epsilon):
    # creates the empty model that has the following format
    # model[0]['p'] = set(pattern)
    # model[0][0] = support(emptyrule) = size of dataset
    # model[0][cl[0]] = support(emptyrule class cl[0]) 
    # etc. for model[0][cl[i]] 
    # Variables: 
    # size_data = len(dataset)
    # supp = dictionary of supports!
    counts_cl = [sum([1 for t in dataset  if c <= t]) for c in cl]
    allcl = frozenset().union(*cl)
    model = defaultdict(dict)
  
    model[0]['p'] = frozenset(list())
    model[0][allcl] = len(dataset)
    for ic,c in enumerate(cl): 
        model[0][c]= counts_cl[ic]
            
    model[0]['lm'] = 0 # length of model
    model[0]['ld'] =  length_data(model,cl,allcl,epsilon) # length of data 
    #model[0]['ld'] =  length_data2(model,cl,allcl,epsilon) # length of data 
    model[0]['const1'],model[0]['const2']  =  delta_data_const(model,cl,allcl,epsilon)
    #list of tuples
    #model = [tuple(-log(1/len(ratioclass),2) for rt in ratioclass)]
    return model	
# transform the model based on support into a model of probabilities
def model_prob(model,cl,allcl,ep):
    modelnew = defaultdict(dict)
    nr = len(model)
    nc = len(cl)
    for r in range(nr):
        for c in cl:
            modelnew[r][c] = (model[r][c]+ep)/(model[r][allcl]+nc*ep)
        modelnew[r]['cl'] = max(modelnew[r], key=modelnew[r].get) 
        modelnew[r]['p'] = model[r]['p']
    return modelnew  	
# add a new rule to the rule list
def addrule(pat,supp,cl,allcl,model,epsilon,unif,l_universal):
    # adds rule with antecedent "pat" and respective usages at the end
    nmodel = copy.deepcopy(model)
    # pat - pattern in tuple form
    # vsupp (N0, Nclass1, Nclass2, ...)
    nr = len(model)
    # New rule
    nmodel[nr]['p']= pat
    nmodel[nr][allcl] = supp[allcl][pat]
    nmodel[0][allcl] = nmodel[0][allcl] - supp[allcl][pat] # update empty rule
    for c in cl:
        saux = supp[c][pat]
        nmodel[nr][c]=  saux
        # update empty rule
        nmodel[0][c] = nmodel[0][c] - saux
        
    nmodel[nr]['lm'] = length_model(nmodel,cl,allcl,unif,l_universal) # length of model
    nmodel[nr]['ld'] =  length_data(nmodel,cl,allcl,epsilon) # length of data 
    nmodel[nr]['const1'],nmodel[nr]['const2'] =  delta_data_const(nmodel,cl,allcl,epsilon)    
    return nmodel
# mine all the frequent items
def getfrequentitems(data,cl,minsupp,maxlhs):
    # returns all patterns/frequent itemsets given a certain minsupp and max length (maxlhs)    
    itemsets = []
    for c in cl:
        #start_time = time.time()
        data_aux = [t.difference(c) for t in data if c <= t]
        #print("Time for set difference: " +str(time.time()-start_time))
        start_time = time.time()
        itemsets.extend([r[0] for r in fpgrowth(data_aux,supp=minsupp,zmin= 2,zmax=maxlhs)])
        #print("Time for set fpgrowth: " +str(time.time()-start_time))

    #remove repeated sets
    #start_time = time.time()
    itemsets = list(set(itemsets))
    #print("Time for list set transform: " +str(time.time()-start_time))

    #start_time = time.time()
    itemsets.sort(key=len)
    #print("Time for sorting list: " +str(time.time()-start_time))
    return itemsets	
# count occurrences of items 
def countocurrences(data,cl,allcl,single_input,itemsets,model,epsilon,minsupp,l_universal,l_gammaln):
    # count sngletons first
    rmlist = []
    supp, t_id = std_cover(single_input, data, cl)
    unif = 1/len(single_input)
    score = defaultdict(float)
    bestscore = -100000
    for pat in supp[allcl].keys():
        auxscore= deltascore_compute(1,pat,cl,allcl,\
                     model,supp,epsilon,supp[allcl][pat],unif,l_universal,l_gammaln)
        score[pat] =auxscore
        if auxscore > bestscore:
            bestscore = auxscore
            newpat = pat
            
    # non-singletons
    for it in itemsets:
        breakoff = True
        pat = frozenset(it)
        #decompose itemset in singletons
        sing_act = [frozenset([x]) for x in it]
        # check for each classs which singleton has less counts
        for c in [allcl] + cl:
            aux_tid = t_id[c][pat.difference(sing_act[-1])].intersection(t_id[c][pat.difference(sing_act[0])])
            aux_supp = len(aux_tid)
            if c == allcl and aux_supp < minsupp:
                breakoff = False
                break
   
            supp[c][pat] = aux_supp
            t_id[c][pat] = aux_tid
        if breakoff:
            if aux_supp == min([supp[allcl][dpat] for dpat in [pat.difference(s) for s in sing_act]]):
                rmlist.append(pat) 
            auxscore= deltascore_compute(1,pat,cl,allcl,\
                                           model,supp,epsilon,supp[allcl][pat],unif,l_universal,l_gammaln)
            score[pat] =auxscore
            if auxscore >bestscore:
                bestscore = auxscore
                newpat = pat
            
    #print("Size remove list: " +str(len(rmlist)))
    for rm in rmlist:    
        for c in cl:
            t_id[c].pop(rm)
            supp[c].pop(rm)
        score.pop(rm)
        supp[allcl].pop(rm)        
    return supp, t_id, score,unif,newpat,bestscore
# update counts in the dataset
def update_dataset(pat,supp,t_id,score,cl,allcl,model,epsilon,unif,l_universal,l_gammaln):
    remtid = [t_id[c][pat] for c in cl] # transactions ids to remove!
    rmlist = []
    nr = len(model)
    bestscore = -10000000
    newpat = []
    
    #keysscore = sorted(score, key=score.__getitem__, reverse=True)
    for cand in supp[allcl].keys():
        auxs = 0
        for ic,c in enumerate(cl):
            auxtid = t_id[c][cand] - remtid[ic] # the difference!
            laux = len(auxtid)
            auxs += laux
            t_id[c][cand] = auxtid
            supp[c][cand] = laux
        
        # THIS if will drop the singletons if they do not have a minimum support
        if auxs > 0 and cand not in cl:    
            supp[allcl][cand] = auxs
            auxscore = deltascore_compute(nr,cand,cl,allcl,model,supp,epsilon,auxs,unif,l_universal,l_gammaln) 
            score[cand] =auxscore
            if auxscore >bestscore:
                bestscore = auxscore
                newpat = cand
        elif cand in cl:
            pass
        else:
            rmlist.append(cand)
    # remove unwanted patterns        
    for rm in rmlist:    
        for c in cl:
            t_id[c].pop(rm)
            supp[c].pop(rm)
        score.pop(rm)
        supp[allcl].pop(rm)
        
    return newpat, bestscore
# transform the data from a normal boolean dataset to a itemset dataset
def binary2itemsets(data_bin):
    # The dataset format has to have the data comma separated
    # with the last column as the class
    #data_bin = np.genfromtxt(filename, delimiter=',',dtype=int)
    classes = np.unique(data_bin[:,-1])
    nclasses = len(classes)
    # number of variables in the dataset excluding classes
    nvars = (data_bin.shape[1]-1)
    itemclasses = [nvars+i for i in range(1,nclasses+1)] 
    item2class = {frozenset([it]): classes[i] for i,it in enumerate(itemclasses)}

    nitems = nvars+nclasses
    data_it=[]
    for row in data_bin:
        aux = np.nonzero(row[:-1])[0]+1
        # check index and then put the classe the nvectorclass
        idxclass = np.where(np.in1d(classes, row[-1]))[0][0]
        data_it.append(set(aux.tolist()+[itemclasses[idxclass]]))

    cl = [frozenset([it]) for it in itemclasses]    
    # return a data set with the class in a frozenset and the rest in a set as the usual inputs for my code
    
    return data_it,cl,item2class
# make a prediction given a unlabeled dataset and a model
def prediction(undata,model,cl,item2class):
    # undata = unlabeled data
    # Initialization of the model
    #[(0->Pattern[s],
    #  1->class[s],
    #  2->L(cl|P)[f],
    #  3->L(not cl|P)[s],
    #  4->Pattern and class[s],
    #  5-> Pr(cl|P)[f],
    #  (...),...]
    # Empty model
    #probClass =[[0]*len(cl)  for i in range(len(undata))]
    pred = np.zeros(len(undata),dtype=int)
    probvector = np.zeros(len(undata))
    probmatrix = np.zeros([len(undata),len(cl)])
    RULEactivated = []
    # Find majority class
    nr = len(model)
    for it,t in enumerate(undata):
        for r in range(nr):
            if model[r]['p'] <= t:
                pred[it] = item2class[model[r]['cl']]
                probvector[it] = model[r][model[r]['cl']]
                probmatrix[it] =np.array([model[r][c] for c in cl])
                RULEactivated.append(r) 
                break
    return pred, probvector, probmatrix,RULEactivated                     
# Example to run in case mdl_rulelists is called alone!	
if __name__ == '__main__':
    datasetname= "wine"
    fileload = "./datasets/wine.csv"
    data_orig = np.genfromtxt(fileload, delimiter=',',dtype=int)
    data, cl,item2class = binary2itemsets(data_orig)
    minsuppvals = [5]
    minsupp = 5
    maxlen = 4
    print("Dataset: " + str(datasetname) +" ,min supp: " + str(minsupp)+" ,max length: " + str(maxlen))
    modelprob,modelraw,lfinal,lorig,nfreqp = mdl_rulelist(data, cl,minsupp,maxlen)
    pred, probvector, probmatrix,RULEactivated = prediction(data,modelprob,cl,item2class)
    print("Accuracy score on training data: " +str(accuracy_score(data_orig[:,-1],pred)))
    y_aux = label_binarize(data_orig[:,-1], classes = np.unique(data_orig[:,-1])) 
    aucweighted = aucscore(y_aux,probmatrix,average ="weighted")
    print("AUC score on training data: " +str(aucweighted))


