import matplotlib as plt
plt.use('Agg')
import re
import os
import random
import numpy as np
import nltk
from nltk.corpus import reuters
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
from sklearn.feature_extraction import DictVectorizer
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
#script to produce data for figure 4(a)
#input: none
#output: prediction error estimates for 1000 different radom partitions
#        for different K
##############################


#set seeds for reproducibility
random.seed(1234)
np.random.seed(1234)
rs=1234



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




#repeat CV procedures
def repeatCV(numberofreplications,numberoffolds):
	out=[]
	for j in range(0,numberofreplications):
		print(j)
		cv=KFold(n_splits=numberoffolds,shuffle=True,random_state=rs+j*numberoffolds)
		scores=cross_val_score(classif, X_counts, Y, cv=cv,n_jobs=40)
		out.append(scores.mean())

	return out
#run experiment 

classif=LogisticRegression()


numberofrepetitions=1000


for i in range(2,15):
	out=repeatCV(numberofrepetitions,i)
	with open("POS_K_VAR", 'a+') as file:
		file.write(str(out)+'\n')
	file.close
