# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 09:20:33 2016

@author: ramesho
"""

class Base_Learner():
    def __init__(self,clf):
        self.trainX = 0
        self.trainY = 0
        self.testX = 0
        self.testY = 0
        self.clf = clf

    def loadTrain(self,trainX,trainY):
        self.trainX = trainX
        self.trainY = trainY
    def loadTest(self,testX,testY):
        self.testX = testX
        self.testY = testY        
    def fit(self):        
        
        self.clf.fit(self.trainX,self.trainY)
    def score(self):
        return self.clf.score(self.testX,self.testY)
    def predictProb(self,unlabeled):
        return self.clf.predict_proba(unlabeled)
    def predict(self,features):
        return self.clf.predict(features)