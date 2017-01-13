# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#This file is used to compare our active learning algorithm with the passive learning algorithm - We will use the same base learner
import numpy as np
from matplotlib import pyplot as plt
from BaseLearner import Base_Learner
from sklearn import ensemble, svm , naive_bayes
from sklearn.feature_selection import SelectKBest,f_classif


#classifier,algoName = svm.SVC(probability =True),'SVM'
base_learner_algos = [[ensemble.RandomForestClassifier(),'RANDOMFOREST'],[svm.SVC(probability =True),'SVM'],[naive_bayes.GaussianNB(),'Gaussian_NB']]
classifier,algoName = base_learner_algos[1]
#classifier,algoName = naive_bayes.GaussianNB(),'Gaussian_NB'
dataset_list = ['EASY','MODERATE','DIFFICULT']
dataset = dataset_list[0]
poolsize = 50
pool_size_start = 500
budget=2500
iterations = 1
#Loading Datasets
raw_data = np.loadtxt('Data/Data/'+dataset+'_TRAIN.csv', delimiter=',', dtype=bytes).astype(str)
trainX = raw_data[:, :-1].astype(np.float64)
trainY = raw_data[:, -1]

baseLearner = Base_Learner(classifier)
 #Feature Selection
feature_selection_clf = SelectKBest(f_classif,k=26)
feature_selection_clf.fit(trainX,trainY)
trainX = feature_selection_clf.transform(trainX)
baseLearner.loadTrain(trainX,trainY)
baseLearner.fit()
testX = np.loadtxt('Data/Data/'+dataset+'_BLINDED.csv', delimiter=',')[:,1:]
testX = feature_selection_clf.transform(testX)
testY = np.loadtxt('Predictions/'+dataset+'_Predictions.csv', delimiter=',', dtype=bytes).astype(str)[:,1]
baseLearner.loadTest(testX,testY)
print( baseLearner.score())