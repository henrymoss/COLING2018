import matplotlib as plt
plt.use('Agg')
import re
import os
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import *
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import *
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from random import randint

################################
#script to produce figure 1(b)
#input: none
#output: figure showing variation in K-fold CV
#        prediction error estimates for different K
##############################

#set seeds for reproducibility
random.seed(1234)
rs=1234


numberoffolds=5
#data preparation
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


#store in dataframe
data = pd.DataFrame(clas, columns=['class'])
data["text"] = text
data = shuffle(data)
print("We have "+str(len(text))+" classified examples")



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

#repeat CV procedures
def repeatCV(numberofreplications,numberoffolds):
	out=[]
	for j in range(0,numberofreplications):
		print(j)
		cv=KFold(n_splits=numberoffolds,shuffle=True,random_state=rs+j*numberoffolds)
		scores=cross_val_score(classif, X_fixed_tfidf, Y_fixed, cv=cv,n_jobs=20)
		out.append(scores.mean())

	return out
#run experiment 

classif =RandomForestClassifier(random_state=42,n_jobs=20,n_estimators=100,max_depth=4)


numberofrepetitions=1000
out=[]
for i in range(2,15):
	out.append(repeatCV(numberofrepetitions,i))
	print(i)

import matplotlib.pyplot as plt
fig = plt.figure(1)
ax = fig.add_subplot(111)
bp = ax.boxplot(out)
ax.set_xticklabels([x for x in range(2,15)])
plt.xlabel('Number of Folds')
plt.ylabel('Accuracy')
plt.ylim(0.55,0.85)
plt.axhline(y=0.7285714285714285,color="r")
plt.show()
plt.savefig('MOVIES_VAR_K_2.pdf')

