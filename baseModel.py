import pickle
import pandas as pd
import numpy as np

#reading data set
from sklearn import ensemble
class BaseLearner():
    def __init__(self):
        self.trainX = 0
        self.trainY = 0
        self.testX = 0
        self.testY = 0
        self.clf = ensemble.RandomForestClassifier()
    def load(self,trainX,trainY,testX,testY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY        
    def fit(self):        
        self.clf.fit(self.trainX,self.trainY)
    def score(self):
        return self.clf.score(self.testX,self.testY)