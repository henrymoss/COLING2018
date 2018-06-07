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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



import warnings
warnings.filterwarnings("ignore")
################################
#script to perform CV on RF for Movie and select max_features 1000 partition choices
#output: prediction scores for each parameter value on grid for each of 1000 partition choices
##############################


#set number of folds
numberoffolds=5

#set seed for reproducibility
rs=42
np.random.seed(1234)
random.seed(1234)
text=[]
clas = []
classname = ["pos", "neg"]
#load training examples
for item in classname:
	for file in os.listdir("../../../Data/aclImdb/test/" +item ):
			filename = "../../../Data/aclImdb/test/" +item + "//"+file
			fl = open(filename, "r", encoding="utf8").read()
			fl = re.sub("\n", " ", fl)
			text.append(fl)
			clas.append(item)
#load testing examples
for item in classname:
	for file in os.listdir("../../../Data/aclImdb/train/" +item ):
			filename = "../../../Data/aclImdb/train/" +item + "//"+file
			fl = open(filename, "r", encoding="utf8").read()
			fl = re.sub("\n", " ", fl)
			text.append(fl)
			clas.append(item)


data = pd.DataFrame(clas, columns=['class'])
data["text"] = text
data = shuffle(data)
#prepare data 
Y = data["class"].tolist()
X = data["text"].tolist()


#choose 1000 to be training set
#leave remaing 49000 to be test set/represent the population
Y = data["class"]
X = data["text"]
X_fixed=X[:1000]
Y_fixed=Y[:1000]
X_pop=X[1000:]
Y_pop=Y[1000:]

count_vect = CountVectorizer(min_df = 1, ngram_range = (1, 3),analyzer='word',max_features=300)
X_fixed_counts = count_vect.fit_transform(X_fixed)
tfidf_transformer = TfidfTransformer()
X_fixed_tfidf = tfidf_transformer.fit_transform(X_fixed_counts)
X_pop_counts=count_vect.transform(X_pop)
X_pop_tfidf=tfidf_transformer.transform(X_pop_counts)

#set up possible parameters

parameter_candidates = [{'max_features':[x/100 for x in range(1,50)]}]

#create a classifier object with the classifier and parameter candidates
# this fixes partition aross evaluations
def crossvalidate(randomseed):
	clf=RandomForestClassifier(random_state=42,n_estimators=100,n_jobs=20,max_depth=4)
	search = GridSearchCV(estimator=clf,n_jobs=20, param_grid=parameter_candidates,cv=KFold(n_splits=numberoffolds,shuffle=True,random_state=randomseed))
	# Train the classifier on data1's feature and target data
	search.fit(X_fixed_tfidf,Y_fixed)   
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
#save data
with open('MoviesRF_differentpartions_1000_'+str(numberoffolds)+'_folds', 'wb') as fp:
	pickle.dump(results, fp)

