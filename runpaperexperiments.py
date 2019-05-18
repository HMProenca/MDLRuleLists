#######Supplement for:
#
#Hugo M. Proenca, Matthijs van Leeuewen (2019) 
#"Interpretable multiclass classification by MDL-based rule lists"
#"Interpretable classifiers using rules and Bayesian analysis: Building a better 
# https://arxiv.org/abs/1905.00328
#
#Version 1.0, May 14, 2019
# 
#
####README
# This code implements the experiments described in the paper
# 
# #This code requires the external frequent itemset mining package "PyFIM," 
#  available at http://www.borgelt.net/pyfim.html
#
# and the script "mdl_rulelists.py"
# 
# to run the experiments please use: python runexperiments.py [datasetname] [typeofexperiment]
#
# Types of experiments
# - typeofexperiment = 0 - minsupp = 5% and max length = 4
# - typeofexperiment = 1 - minsupp = [25,20,15,10,5,2,1,0.5,0.1]% and max length = 4
#
# dataset name: 
# datasets = ["breast","pima","iris","heart","hepatitis","iris","led7","pageblocs","tictactoe","wine",
#             "adult","chessbig","cylbands","horsecolic","pendigits","waveform","ionosphere","mushroom"]
#
# example (run in command line with pyfim and python3 installed): 
#             python runexperiments.py breast 0 
# 
# this will return the crossvalitation results using the indexes of the folds in the folder "datasets"
# and return all results to the folder "xps"
#
################################################################################
# OUTPUT files (example with "breast" dataset)
# 
#  a file in xps/ is created with the name "breastsupp5_len4_results"
# 
# - a_summarybreast.txt: contains a summary of all the results performed including the
#                    rule lists of the last folder
#
# - datatest0.txt       : dataset used for testing in fold 0 in itemset format (the first folder)
# 
# - datatrain0.txt      : dataset used for training in fold 0 in itemset format (the first folder)
#
# - model0.txt          : the model with probabilities as consequent
#
# - modelraw.txt        : the model with the respective support as consequents
# 
# - predictiontrain0.txt: the predictions made for datatrain0.txt in terms of itemsets
#
# - predictionprobtest0.txt: the probability of the prediction made for datatest0.txt
#
# - predictionprobtrain0.txt: the probability of the prediction made for datasettrain0.txt
#
# - ytest0             : class labels in itemsets of datasettest0.txt
#
# - ytrain0             : class labels in itemsets of datasettrain0.txt
#
import mdl_rulelists
import os
import sys
import gc
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from collections  import defaultdict
from math  import log, ceil, floor
from fim import eclat
from fim import fpgrowth
from scipy.special  import gammaln
from math import lgamma
import operator
import copy
from scipy.misc  import comb
from sklearn.metrics  import roc_auc_score as aucscore
from sklearn.model_selection  import StratifiedKFold
# crossvalidation loop using the indexes given so as to replicate the experiments
def crossvalidation(data,cl,nfolds,name,maxlen,minsuppclass,writeon,datasetdir):
    debugging_file = "xps/" + name + "supp" + str(minsuppclass) + \
    "_len" + str(int(maxlen)) + "_results"
    if os.path.isdir(debugging_file) == False:
        os.makedirs(debugging_file)
    auxname = os.path.join(debugging_file,"a_summary"+name+'.txt')
    f = open(auxname, 'w')
    print("Dataset: " + name,file=open(auxname, "a"))
    print("Note: Length of dataset uses rounded values",file=open(auxname, "a"))
    print("Parameters| nfolds: %d | Min Supp: %d | max length: %d " \
    %(nfolds,minsuppclass,maxlen),file=open(auxname, "a"))
    nclasses = len(cl)
    #skf = StratifiedKFold(n_splits=nfolds, random_state=1, shuffle=True)
    Nrules = [0]*nfolds
    Nitems,Ndiffitems, Lfinal, Lorig,nfreqp =[0]*nfolds, [0]*nfolds,[0]*nfolds,[0]*nfolds,[0]*nfolds
    acctr,aucmicrotr,aucmacrotr,aucweightedtr = [0]*nfolds,[0]*nfolds,[0]*nfolds,[0]*nfolds
    acctest,aucmicrotest,aucmacrotest,aucweightedtest = [0]*nfolds,[0]*nfolds,[0]*nfolds,[0]*nfolds
    auxpredtrain,auxrealtrain,auxpredtest,auxrealtest = [],[],[],[]
    time_elapsed = [0]*nfolds
    i = 0
    clvecaux = [ic for t in data for ic,c in enumerate(cl) if c <=  t]    
    #for train_idx, test_idx in skf.split(data, clvecaux):
    for auxidx in range(1,nfolds+1):
        print("Running Fold " +str(auxidx))
        filetoload = datasetdir + "testindex/" + "testindex_" + name + "_fold"+str(auxidx)+".txt"
        test_idx = np.loadtxt(filetoload, dtype = int)
        test_idx = [aa-1 for aa in test_idx]
        train_idx =[aa  for aa in range(len(clvecaux)) if aa not in test_idx]
        # Assign train and test
        XY_train = [data[idx] for idx in train_idx]
        write_file(debugging_file,"datatrain"+str(i)+".txt",XY_train,cl,writeon)
        XY_test = [data[idx] for idx in test_idx]
        write_file(debugging_file,"datatest"+str(i)+".txt",XY_test,cl,writeon)
        # Training 
        start_time = time.time()
        model, modelraw, Lfinal[i], Lorig[i],nfreqp[i] = \
            mdl_rulelists.mdl_rulelist(XY_train,cl,minsuppclass,maxlen)
        time_elapsed[i] = time.time()-start_time
        y_train = [c for t in XY_train for ic,c in enumerate(cl) if c <=  t]
        write_file(debugging_file,"ytrain"+str(i)+".txt",y_train,cl,writeon)
        y_test = [c for t in XY_test for ic,c in enumerate(cl) if c <=  t]
        write_file(debugging_file,"ytest"+str(i)+".txt",y_test,cl,writeon)
       
        predtr, probtr, RULEactivatedtr = prediction_itemset(XY_train,model,cl)
        acctr[i],aucmicrotr[i],aucmacrotr[i],aucweightedtr[i] \
        = performancemetrics(y_train,cl,model,RULEactivatedtr,predtr,probtr)
        write_file(debugging_file,"predictiontrain"+str(i)+".txt",predtr,cl,writeon)
        write_file(debugging_file,"model"+str(i)+".txt",model,cl,writeon)
        write_file(debugging_file,"modelraw"+str(i)+".txt",modelraw,cl,writeon)
        output = open(os.path.join(debugging_file, "predprobtrain"+str(i)+".csv"), 'w')
        for auxxxxx in probtr:
            output.write("%s \n" % auxxxxx)
        output.close()          
        Nrules[i] = len(model)-1
        auxpredtrain.extend(predtr)
        auxrealtrain.extend(y_train) 
        Nitems[i]= sum([len(model[r]['p'])/(len(model)-1) for r in range(1,len(model))])
        Ndiffitems[i] = len(frozenset().union(*[model[key]['p'] for key in model]))
        ## Test
        predtest, probtest, RULEactivatedtest = prediction_itemset(XY_test,model,cl)
        acctest[i],aucmicrotest[i],aucmacrotest[i],aucweightedtest[i] \
        = performancemetrics(y_test,cl,model,RULEactivatedtest,predtest,probtest)
        output = open(os.path.join(debugging_file, "predprobtest"+str(i)+".csv"), 'w')
        for auxxxxx in probtest:
            output.write("%s \n" % auxxxxx)
        output.close() 
        
        #fscoretest[i] = f1_score(y_test, predtest, average='binary', sample_weight=None)
        print("ACCtrain: " +str(round(acctr[i],3)) + " | ACCtest: " +str(round(acctest[i],3))+\
              " | AUCMicrotrain: " +str(round(aucmicrotr[i],3)) + " | AUCMicrotest: " +str(round(aucmicrotest[i],3)) +\
              " | AUCMacrotrain: " +str(round(aucmacrotr[i],3)) + " | AUCMacrotest: " +str(round(aucmacrotest[i],3)) +\
              " | AUCWeighttrain: " +str(round(aucweightedtr[i],3)) + " | AUCWeigthtest: " +str(round(aucweightedtest[i],3)) +\
              " | Lorig: " +str(Lorig[i]) + " | Lfinal: " +str(Lfinal[i]) + " | N_freq_it: " +str(nfreqp[i]) + " | time: " +str(time_elapsed[i]/60)+
              " | nrules: " +str(Nrules[i]) + " | ntiems: " +str(Nitems[i])\
              ,file=open(os.path.join(debugging_file,"a_summary"+name+'.txt'), "a"))
        #print("ACCtrain: " +str(acctr[i]) + "| ACCtest: " +str(acctest[i])+ " | F1train: " +str(fscoretr[i]) + "| F1test: " +str(fscoretest[i]))

        auxpredtest.extend(predtest) 
        auxrealtest.extend(y_test) 
        i+=1
    
    items =set.union(*data)
    ST = [set([i]) for i in items if i not in cl]
    compress = [a/b for a, b in zip(Lfinal,Lorig)]
    print("AcctrainMean : AcctrainSTD : AcctestMEAN : AcctestSTD :"+\
    " AUCtrainMicroMEAN : AUCtrainMicroSTD : AUCtestMicroMEAN : AUCtestMicroSTD :" + \
    " AUCtrainMacroMEAN : AUCtrainMacrotSTD : AUCtestMacroMEAN : AUCtestMacroSTD :" +\
    " AUCtrainWeightMEAN : AUCtrainWeightSTD : AUCtestWeightMEAN : AUCtestWeightSTD :" +\
    " LengthFinalMEAN : LengthFinalSTD :" + \
    " LengthRatioMEAN : LengthRatioSTD :" + \
    " Nrules : Nitems : Nitemsused: Ninstances :"+ \
    " Nvariables : Nclasses : avg_time : avg_freq_it :\n"+ \
    "%.3f : %.3f : %.3f : %.3f :%.3f : %.3f : %.3f : %.3f :%.3f : %.3f :\
    %.3f : %.3f : %.3f :%.3f : %.3f : %.3f : %.3f : %.3f : %.3f : %.3f :\
    %.1f : %.1f : %.1f : %.0f : %.0f : %.0f: %.3f: %.0f"
          %(np.mean(acctr), np.var(acctr)** (0.5),round(np.mean(acctest),3),np.var(acctest)** (0.5),
            np.mean(aucmicrotr), np.var(aucmicrotr)** (0.5), \
            np.mean(aucmicrotest),np.var(aucmicrotest)** (0.5),\
            np.mean(aucmacrotr), np.var(aucmacrotr)** (0.5), \
            np.mean(aucmacrotest),np.var(aucmacrotest)** (0.5),\
            np.mean(aucweightedtr), np.var(aucweightedtr)** (0.5), \
            np.mean(aucweightedtest),np.var(aucweightedtest)** (0.5),\
            np.mean(Lfinal),np.var(Lfinal),\
            np.mean(compress),np.var(compress),\
            sum(Nrules)/len(Nrules),sum(Nitems)/len(Nitems),\
            sum(Ndiffitems)/len(Ndiffitems),
            len(data),len(ST),len(cl),np.mean(time_elapsed)/60,np.mean(nfreqp)),\
          file=open(auxname, "a"))

    # Trasnform a list of sets into a list of ints
    auxrealtrain = sum([list(aux) for aux in auxrealtrain],[])
    auxpredtrain = sum([list(aux) for aux in auxpredtrain],[])
    auxrealtest = sum([list(aux) for aux in auxrealtest],[])
    auxpredtest = sum([list(aux) for aux in auxpredtest],[])
                       
    print("Confusion train matrix: \n",file=open(auxname, "a"))
    confTrain = confusion_matrix(auxrealtrain, auxpredtrain)
    for item in confTrain:
          print(item[0], ', '.join(map(str, item[1:])),file=open(auxname, "a"))
    print("Confusion normalized train matrix: \n",file=open(auxname, "a"))
    confTrainNorm = [[j/sum(row)  for j in row]   for row in confTrain]
    for item in confTrainNorm:
          print(item[0], ', '.join(map(str, item[1:])),file=open(auxname, "a"))
    #print(confusion_matrix(auxrealtrain, auxpredtrain))
    print("Confusion test matrix: \n",file=open(auxname, "a"))
    confTest = confusion_matrix(auxrealtest, auxpredtest)
    for item in confTest:
          print(item[0], ', '.join(map(str, item[1:])),file=open(auxname, "a"))  
            
    print("Confusion normalized test matrix: \n",file=open(auxname, "a"))
    confTestNorm = [[j/sum(row)  for j in row]   for row in confTest]
    for item in confTestNorm:
          print(item[0], ', '.join(map(str, item[1:])),file=open(auxname, "a")) 
            
    rule = "The last model : \n"
    for r in range(len(model)):
        rule += " pattern: " + str(model[r]['p']) + " | class: " +str(model[r]['cl']) +  " \n"
        for c in cl: 
            #rule += " prob" + str(list(c)) + ": " + str(rule[r][c])
            rule += " prob" + str(list(c))  + ": "+ str(model[r][c]) +" \n"    
    print(rule,file=open(auxname, "a"))        
    
    rule = "The last raw model : \n"
    for r in range(len(modelraw)):
        rule += " pattern: " + str(modelraw[r]['p']) +  " \n"
        for c in cl: 
            #rule += " prob" + str(list(c)) + ": " + str(rule[r][c])
            rule += " supp" + str(list(c))  + ": "+ str(modelraw[r][c]) +" \n"    
    print(rule,file=open(auxname, "a"))
    # Deleting objects
    del ST, items,XY_train, XY_test, data, auxrealtrain,auxpredtrain,auxrealtest,auxpredtest
    f.close()
    gc.collect()
# write file with results
def write_file(path,name,toprint,cl,writeon = "off"):
     # from collections # import defaultdict
    # import os
    if writeon == "on":
        output = open(os.path.join(path, name), 'w')
        if type(toprint) == list:
            for transaction in toprint:
                for item in list(transaction):
                      output.write("%s," % item)
                output.write("\n" )
        elif type(toprint) == defaultdict: 
            rule = "Model : \n"
            for r in range(len(toprint)):
                rule += "RULE N. : " +str(r) +" pattern: " + str(toprint[r]['p']) +  " \n"
                for c in cl: 
                    rule += " prob" + str(list(c))  + ": "+ str(toprint[r][c]) +" \n"    
            output.write(rule)
        output.close()
    else:
        pass
# prediction using itemsets
def prediction_itemset(undata,model,cl):
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
    pred = []
    prob = []
    RULEactivated = []
    cumprob = [[1 for c in cl]   for r in model.keys()]
    # Find majority class
    nr = len(model)
    for t in undata:
        for r in range(nr):
            if model[r]['p'] <= t:
                pred.append(model[r]['cl'])
                prob.append(model[r][model[r]['cl']])
                RULEactivated.append(r) 
                break
    return pred, prob, RULEactivated            
    #return predtr, probClass
# Performance meeasures
def performancemetrics(y,cl,model,RULEactivated,pred,probclass):
    accu = accuracy(y, pred)
    if len(cl)== 2: # majority is 1, minority is 0
        indexcl = [ic for t in y for ic,c in enumerate(cl) if c <=  t] 
        predprob = [1-p if model[RULEactivated[ip]]["cl"] == cl[0]\
                    else p for ip,p in enumerate(probclass)]
        aucmacro = aucscore(indexcl,predprob)  
        aucmicro = aucmacro
        aucweighted = aucmacro
        #print("Weighted AUC: " +str(aucmacro))
    else:
        # Micro AUC
        indexcl = [ic for t in y for ic,c in enumerate(cl) if c <=  t] 
        classesaux = [c for c in range(len(cl))]            
        y_aux = label_binarize(indexcl, classes=classesaux) 
        auxauc = np.array([[model[RULEactivated[it]][c] for c in cl] for it,t in enumerate(y)])
        aucmicro = aucscore(y_aux,auxauc,average ="micro")  
        #print("Micro AUC: " +str(aucmicro))
        # Macro AUC
        aucmacro = aucscore(y_aux,auxauc,average ="macro")    
        #print("Macro AUC: " +str(aucmacro))
        #Weighted
        aucweighted = aucscore(y_aux,auxauc,average ="weighted")
        #print("Weighted AUC: " +str(aucweighted))
    return accu,aucmicro,aucmacro,aucweighted
# accuracy
def accuracy(trueval, pred):  
    counttrue = 0
    for it,cl in enumerate(trueval):
        if pred[it] == cl:
            counttrue+= 1
    acc = counttrue/len(trueval)
    return acc
print("dataset")

datasetname = sys.argv[1]
print(datasetname)
typeofexperiment = int(sys.argv[2])
nfolds = 10
datasetfile = "./datasets/" 
#print(typeofexperiment)
# Tests that can be activated
debug = "on"
fileload = datasetfile + datasetname + ".csv" 
data_orig = np.genfromtxt(fileload, delimiter=',',dtype=int)
data, cl,item2class = mdl_rulelists.binary2itemsets(data_orig)
if typeofexperiment == 0:
	minsuppvals = [5]
elif typeofexperiment == 1: 
	minsuppvals = [25,20,15,10,5,2,1,0.5,0.1]
else: 
	print("Type of experiment wrongly selected")
maxlen = 4    
for minsuppclass in minsuppvals:
	print("Dataset: " + str(datasetname) + \
	" ,min supp: " + str(minsuppclass)+ \
	" ,max length: " + str(maxlen))
	crossvalidation(data,cl,nfolds,datasetname,maxlen,minsuppclass,debug,datasetfile)