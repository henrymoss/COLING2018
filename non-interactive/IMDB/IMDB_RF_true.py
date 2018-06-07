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
from sklearn.metrics import accuracy_score


import warnings
warnings.filterwarnings("ignore")
################################
#script to calculate global scores for IMDB Random Forest
#input: none
#output: global performance scores for each parameter choice in grid
##############################


numberoffolds=5
rs=1234
np.random.seed(1234)
random.seed(1234)
text=[]
clas = []
classname = ["pos", "neg"]
#load training examples
for item in classname:
	for file in os.listdir("./Data/aclImdb/test/" +item ):
			filename = "./Data/aclImdb/test/" +item + "//"+file
			fl = open(filename, "r", encoding="utf8").read()
			fl = re.sub("\n", " ", fl)
			text.append(fl)
			clas.append(item)
#load testing examples
for item in classname:
	for file in os.listdir("./Data/aclImdb/train/" +item ):
			filename = "./Data/aclImdb/train/" +item + "//"+file
			fl = open(filename, "r", encoding="utf8").read()
			fl = re.sub("\n", " ", fl)
			text.append(fl)
			clas.append(item)


data = pd.DataFrame(clas, columns=['class'])
data["text"] = text
data = shuffle(data)
#prepare data 
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

#calculate performance for each parameter value
scores=[]
for i in [x/100 for x in range(1,50)]:
    clf=RandomForestClassifier(random_state=42,n_estimators=100,n_jobs=20,max_depth=4,max_features=i)
    clf.fit(X_fixed_tfidf,Y_fixed)
    scores.append(accuracy_score(Y_pop, clf.predict(X_pop_tfidf)))
import pickle
#save data
with open('MoviesRF_Truth', 'wb') as fp:
	pickle.dump(scores, fp)

