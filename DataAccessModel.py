##Pool based Data Access Model

import numpy as np

import math
class DataAccess:
    def __init__(self,features,labels,poolsize):
        self.labeledFeatures = np.empty(shape=(0,len(features[0])))
        self.labels = np.empty(shape=(0))
        self.oracle = labels
        self.pool = features
        self.cost = 0        
        self.poolsize = poolsize
    def poolBasedQuerySelection(self,indexes):
        #get the labels for particular indexexs from the oracle        
        labels = self.oracle[indexes]
        self.labels = np.append(self.labels,labels,axis=0)
        #update the cost
        self.cost = self.cost + len(indexes)
        #move these features to labeledFeatures set and remove from pool
        labeledFeatures = self.pool[indexes,:]
        self.labeledFeatures = np.append(self.labeledFeatures,labeledFeatures,axis = 0)
        
        #deleting the features and labels for whom the call has been made to oracle
        self.oracle = np.delete(self.oracle,indexes,0)
        self.pool = np.delete(self.pool,indexes,0)
    def UncertainitySampling(self,predictProb):
        #calculate entropy for each point
        entropy = np.array([-1*sum([i*math.log(i) for i in rowValues if i != 0]) for rowValues in predictProb   ])
        #args orts sometimes asc and sometimes desc
        #indexes = entropy.argsort()[-1*self.poolsize: ] if (self.cost + self.poolsize) <=2500 else range(2500-self.cost)
        indexes = entropy.argsort()
        if (entropy[indexes[0]] > entropy[indexes[-1]] ): #desc order sorted
            indexes = indexes[:self.poolsize: ] if (self.cost + self.poolsize) <=2500 else range(2500-self.cost)
        else: #asc order sorted
            indexes = indexes[-1*self.poolsize: ] if (self.cost + self.poolsize) <=2500 else range(2500-self.cost)
        return indexes