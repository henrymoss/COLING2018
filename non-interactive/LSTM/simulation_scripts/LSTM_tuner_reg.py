import os
import sys

#script to calc performance for same params but different seeds (seed*100 to seed*100+100)
os.chdir(sys.path[0])

sys.path.append(os.path.abspath(os.pardir))

number_of_folds=5

seed=int(sys.argv[1])



import keras
import numpy as np
import h5py

# Metrics
from sklearn.metrics import accuracy_score

# Models
from tdparse.models.tdlstm import TLSTM
# Tokenisers
from tdparse.tokenisers import whitespace, ark_twokenize
# Word Vectors
from tdparse.word_vectors import PreTrained, GloveTwitterVectors
# Get the data
from tdparse.helper import read_config, full_path
from tdparse.parsers import dong

# Load the datasets
dong_train = dong(full_path(read_config('dong_twit_train_data')))
dong_test = dong(full_path(read_config('dong_twit_test_data')))
# Load the word vectors
sswe_path = full_path(read_config('sswe_files')['vo_zhang'])
sswe = PreTrained(sswe_path, name='sswe')
#glove_50 = GloveTwitterVectors(50)
#glove_100 = GloveTwitterVectors(100)
#glove_200 = GloveTwitterVectors(200)


params=[[0.00001,0.00001],[0.00001,0.001],[0.00001,0.1],[0.001,0.00001],[0.001,0.001],[0.001,0.1],[0.1,0.00001],[0.1,0.001],[0.1,0.001],]
def fitter(rand):
	values=[]
	for reg in params:
		lstm_model = TLSTM(whitespace, sswe, epochs=100,patience=5, lower=True, optimiser='adam',lstm_dimension=50)
		predictions, scores = TLSTM. cross_val(dong_train.data_dict(), dong_train.sentiment_data(), 
									   lstm_model, cv=number_of_folds, scorer=accuracy_score,
									   reproducible=True,seed=seed*1000+rand+50,validation_size=0.2,bias_reg=reg[0],input_reg=reg[1])
		values.append(np.mean(scores))
	with open("output_5_both_reg.txt", 'a+') as file:
		file.write(str(values)+'\n')
	file.close
	return values

for i in range(0,20):
	fitter(i)