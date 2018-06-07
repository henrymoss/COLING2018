import matplotlib as plt
plt.use('Agg')
import re
import os
import random
import numpy as np
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
#script to produce data for figure 4(b)
#input: none
#output: prediction error estimates for 1000 different radom partitions
#        for different K
##############################






#set seeds for reproducibility
random.seed(1234)
rs=1234



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

#repeat CV procedures
def repeatCV(numberofreplications,numberoffolds):
	out=[]
	for j in range(0,numberofreplications):
		print(j)
		cv=KFold(n_splits=numberoffolds,shuffle=True,random_state=rs+j*numberoffolds)
		scores=cross_val_score(classif, X_tfidf, Y, cv=cv,n_jobs=15)
		out.append(scores.mean())

	return out
#run experiment 

classif=svm.SVC(C=10,gamma=0.25)


numberofrepetitions=1000


for i in range(2,15):
	out=repeatCV(numberofrepetitions,i)
	with open("REUTERS_K_VAR", 'a+') as file:
		file.write(str(out)+'\n')
	file.close
