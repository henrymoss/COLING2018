import matplotlib
matplotlib.use('Agg')
import pandas as pd
import os
import sys
import math
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import reuters
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC



import warnings
warnings.filterwarnings("ignore")
################################
#script to perform CV on RF for Movie and select max_features 500 for 500 partition choices
#output: results (1000 by #parameter options pickle)
##############################

numberoffolds=5
#set seeds for reproducibility
random.seed(1234)
rs=1234


#choose number of folds
numberoffolds=5
#data preparation

#select only articles about wheat and/or corn
wheat_docs = reuters.fileids("wheat")
corn_docs = reuters.fileids("corn")
wheat_train = list(filter(lambda doc: doc.startswith("train"),wheat_docs))
wheat_test = list(filter(lambda doc: doc.startswith("test"),wheat_docs))
corn_train = list(filter(lambda doc: doc.startswith("train"),corn_docs))
corn_test = list(filter(lambda doc: doc.startswith("test"),corn_docs))
training_index = wheat_train +wheat_test+ corn_train+corn_test

#prepare data for wheat vs not wheat case
text=[]
clas = []
classname = ["pos", "neg"]
for i in training_index:
        text.append(reuters.raw(i))
        #check categorisation to make response
        if "wheat" in reuters.categories(i):
            clas.append(1)
        else:
            clas.append(0)


#store in dataframe
data = pd.DataFrame(clas, columns=['class'])
data["text"] = text
data = shuffle(data)
print("We have "+str(len(text))+" classified examples")


Y = data["class"]
X = data["text"]


count_vect = CountVectorizer(min_df = 1, ngram_range = (1, 3),analyzer='word',max_features=300)
X_counts = count_vect.fit_transform(X)
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

#set up possible parameters

parameter_candidates = [{'C':[1,5,10,50,100,500,1000,5000,10000],'gamma':[0.05*x for x in range(1,10)]}]

#create a classifier object with the classifier and parameter candidates
# this fixes partition aross evaluations
def crossvalidate(randomseed):
	classif=SVC()
	search = GridSearchCV(estimator=classif,n_jobs=15, param_grid=parameter_candidates,cv=KFold(n_splits=numberoffolds,shuffle=True,random_state=randomseed))
	# Train the classifier on data1's feature and target data
	search.fit(X_tfidf,Y)   
	return [search.grid_scores_,search.best_estimator_]
#run script
results=[]
# different partitions for each round of tuning
for r in range(1,1000):
	experiments=crossvalidate(r)
	y=[]
	for i in range(0,len(experiments[0])):
		y.append(experiments[0][i][1])
	results.append(y)
import pickle

with open('REUTERSSVM_differentpartitions_1000'+str(numberoffolds)+'_folds', 'wb') as fp:
	pickle.dump(results, fp)

