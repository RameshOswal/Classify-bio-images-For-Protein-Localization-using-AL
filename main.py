# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 05:26:56 2016

@author: ramesho
"""
import DataAccessModel
import time
import numpy as np
from matplotlib import pyplot as plt
from BaseLearner import Base_Learner
from sklearn import ensemble, svm , naive_bayes
from sklearn.neighbors import KNeighborsClassifier
import random
import math
from sklearn.feature_selection import SelectKBest,chi2,f_classif

base_learner_algos = [[ensemble.RandomForestClassifier(),'RANDOMFOREST'],[svm.SVC(probability =True),'SVM'],[naive_bayes.GaussianNB(),'Gaussian_NB'],[KNeighborsClassifier(n_neighbors=3),'KNN']]
dataset_list = ['EASY','MODERATE','DIFFICULT']
poolsize_list = [50,100,150,200,500]
pool_size_start_list = [500,1000]   
budget=2500
iterations = 1
# loops to check all combinations of algos, poolsize and dataset
for classifier,algoName in [base_learner_algos[1]]:
    for dataset in dataset_list[1:2]:
        for poolsize in [poolsize_list[0]]:
            for pool_size_start in [pool_size_start_list[0]]:  
                for i in range(iterations):
                    # for one combination following algorithm works
                    #Loading Datasets
                    raw_data = np.loadtxt('Data/Data/'+dataset+'_TRAIN.csv', delimiter=',', dtype=bytes).astype(str)
                    trainX = raw_data[:, :-1].astype(np.float64)
                    trainY = raw_data[:, -1]
                    raw_data = np.loadtxt('Data/Data/'+dataset+'_TEST.csv', delimiter=',', dtype=bytes).astype(str)
                    testX = raw_data[:, :-1].astype(np.float64)
                    testY = raw_data[:, -1]
                    #Initializing cost and error matrix
                    array_length = int(math.ceil((budget-pool_size_start)/float(poolsize))+1)
                    active_learner_error = np.zeros(shape=(array_length))
                    active_learner_cost = np.zeros(shape=(array_length))
                    random_learner_error = np.zeros(shape=(array_length))
                    random_learner_cost = np.zeros(shape=(array_length))
                    #Running for  Active Learner Algo
                    #first random poolsize
                    indexes = range(len(trainX))
                    random.shuffle(indexes)
                    indexes = indexes[:pool_size_start]
                    randomIndexes = np.array(indexes)
                    #Feature Selection
                    feature_selection_clf = SelectKBest(f_classif,k=26)
                    feature_selection_clf.fit(trainX[indexes,:],trainY[indexes])
                    trainX = feature_selection_clf.transform(trainX)
                    testX = feature_selection_clf.transform(testX)
                    #setting the base learner Data Access and Query Selection Strategy
                    baseLearner = Base_Learner(classifier)
                    baseLearner.loadTest(testX,testY)
                    dataAccess = DataAccessModel.DataAccess(trainX,trainY,poolsize)
                    i=0
                    while (dataAccess.cost < budget):
                        dataAccess.poolBasedQuerySelection(indexes )
                        baseLearner.loadTrain(dataAccess.labeledFeatures,dataAccess.labels)
                        baseLearner.fit()
                        error = 1 - baseLearner.score()
                        indexes = dataAccess.UncertainitySampling(baseLearner.predictProb(dataAccess.pool))
                        #print(s,'cost: ',dataAccess.cost,'len of indexes',len(indexes))
                        active_learner_error[i] = active_learner_error[i] + error
                        active_learner_cost[i] = active_learner_cost[i] + dataAccess.cost                       
                        i = i +1
                        
                    #making predictions for blinded dataset                    
                    blinded_predictions_features = np.loadtxt('Data/Data/'+dataset+'_BLINDED.csv', delimiter=',')
                    indexFeatures = blinded_predictions_features[:,0]
                    blinded_predictions_features = blinded_predictions_features[:,1:].astype(np.float64)
                    blinded_predictions_features = feature_selection_clf.transform(blinded_predictions_features)
                    #blinded_predictions = np.zeros([len(blinded_predictions_features),2])
                    blinded_predictions = baseLearner.predict(blinded_predictions_features)
                    #print(blinded_predictions)                    
                    np.savetxt('Predictions/'+dataset+'_Predictions_' +time.asctime( time.localtime(time.time()) ).replace(':','_')+'.csv',blinded_predictions,delimiter = ',',fmt='%s')
                                  
                    pred = np.column_stack([indexFeatures,blinded_predictions])
                    print(blinded_predictions)
                    np.savetxt('Predictions/'+dataset+'_Predictions.csv',pred,delimiter = ',',fmt="%s")
                    
                    #Running for Random Learner
                    baseLearner = Base_Learner(classifier)
                    baseLearner.loadTest(testX,testY)
                    dataAccess = DataAccessModel.DataAccess(trainX,trainY,poolsize)
                    indexes = randomIndexes
                    i=0
                    while (dataAccess.cost < budget):
                        dataAccess.poolBasedQuerySelection(indexes )
                        baseLearner.loadTrain(dataAccess.labeledFeatures,dataAccess.labels)
                        baseLearner.fit()
                        error =1- baseLearner.score()
                        poolRange = range(len(dataAccess.pool))
                        random.shuffle(poolRange)
                        indexes = np.array(poolRange[:poolsize]) if (dataAccess.cost +dataAccess.poolsize) <=2500 else range(2500-dataAccess.cost)
                        #print(s,'cost: ',dataAccess.cost,'len of indexes',len(indexes))
                        random_learner_error[i] = random_learner_error[i] + error
                        random_learner_cost[i] = random_learner_cost[i] + dataAccess.cost
                        i=i +1
                #Averaging out the errors
                active_learner_average_error = active_learner_error/iterations
                active_learner_average_cost = active_learner_cost/iterations
                random_learner_average_error = random_learner_error/iterations
                random_learner_average_cost = random_learner_cost/iterations
                #Plot graph for error and cost
                plt.plot(active_learner_average_cost,active_learner_average_error, 'r', label = 'Active Learner')
                plt.plot(random_learner_average_cost,random_learner_average_error,'b--',label = 'Random Learner')
                plt.legend(loc='best')
                plt.title('Labeled Data V/s Test Error (Base Learner -'+algoName+')')
                plt.xlabel('Cost')
                plt.ylabel('Test Error')
                localtime = time.asctime( time.localtime(time.time()) )
                localtime = localtime.replace(':','_') #using local time to save all graphs
                filename = 'Graphs/'+localtime + '_' + algoName + '_POOL_SIZE_'+str(poolsize)+ '_POOL_STARTSIZE_'+str(pool_size_start)+'_'+dataset+'_iterations_'+str(iterations)
                plt.savefig(filename)
                plt.show()                
                print(filename)
                plt.close()
                import csv   
                fields=[localtime,algoName,dataset,pool_size_start,poolsize,min(active_learner_average_error),active_learner_average_cost[np.argmin(active_learner_average_error)],active_learner_average_error[-1],min(random_learner_average_error),random_learner_average_cost[np.argmin(random_learner_average_error)],random_learner_average_error[-1]]
                             
                with open(r'Analysis.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)