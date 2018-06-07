import matplotlib
matplotlib.use('Agg')
import pandas as pd
import os
import sys
import math
import re
import random
import numpy as np
import nltk
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
from sklearn.feature_extraction import DictVectorizer



import warnings
warnings.filterwarnings("ignore")
################################
#script to perform CV on RF for Movie and select max_features 500 for 500 partition choices
#output: results (1000 by #parameter options pickle)
##############################

#set seeds for reproducibility
random.seed(1234)
np.random.seed(1234)
rs=1234

#choose number of folds
numberoffolds=10

# prepare training data (brown corpus)
tagged_sentences=[] 
for i in nltk.corpus.treebank.tagged_sents():
	tagged_sentences.append(i)

random.shuffle(tagged_sentences)
#check loaded
print(str(len(tagged_sentences))+" tagged sentences and "+str(len(nltk.corpus.brown.tagged_words()))+" words")



#need some feature extraction

def features(sentence, index):
	""" sentence: [w1, w2, ...], index: the index of the word """
	return {
		'word': sentence[index],
		'is_first': index == 0,
		'is_last': index == len(sentence) - 1,
		'is_capitalized': sentence[index][0].upper() == sentence[index][0],
		'is_all_caps': sentence[index].upper() == sentence[index],
		'is_all_lower': sentence[index].lower() == sentence[index],
		'prefix-1': sentence[index][0],
		'prefix-2': sentence[index][:2],
		'prefix-3': sentence[index][:3],
		'suffix-1': sentence[index][-1],
		'suffix-2': sentence[index][-2:],
		'suffix-3': sentence[index][-3:],
		'prev_word': '' if index == 0 else sentence[index - 1],
		'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
		'has_hyphen': '-' in sentence[index],
		'is_numeric': sentence[index].isdigit(),
		'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
	}

# helper function to strip tags from corpus to make training 
def untag(tagged_sentence):
	return [w for w, t in tagged_sentence]


def transform_to_dataset(tagged_sentences):
	X, y = [], []
 
	for tagged in tagged_sentences:
		for index in range(len(tagged)):
			X.append(features(untag(tagged), index))
			y.append(tagged[index][1])
 
	return X, y

X,Y = transform_to_dataset(tagged_sentences)
#have 100 000 words training in total
#insetead look at using 10 000 
X=X[:10000]
Y=Y[:10000]


dict_vect = DictVectorizer()
X_counts = dict_vect.fit_transform(X)



#set up possible parameters

parameter_candidates = [{'C':[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100]}]

#create a classifier object with the classifier and parameter candidates
# this fixes partition aross evaluations
def crossvalidate(randomseed):
	classif=LogisticRegression(n_jobs=10)
	search = GridSearchCV(estimator=classif,n_jobs=10, param_grid=parameter_candidates,cv=KFold(n_splits=numberoffolds,shuffle=True,random_state=randomseed))
	# Train the classifier on data1's feature and target data
	search.fit(X_counts,Y)   
	return [search.grid_scores_,search.best_estimator_]
#run script
results=[]
# different partitions for each round of tuning
for r in range(1,1000):
	experiments=crossvalidate(r)
	print(r)
	y=[]
	for i in range(0,len(experiments[0])):
		y.append(experiments[0][i][1])
	results.append(y)
import pickle

with open('POSLR_differentpartitions_1000_'+str(numberoffolds), 'wb') as fp:
	pickle.dump(results, fp)

