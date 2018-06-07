import random as rn
import os
import tempfile

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import preprocessing, models, optimizers, initializers
from keras.layers import Dense, LSTM, Input, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold, ShuffleSplit

class TLSTM():
	def __init__(self, tokeniser, embeddings, pad_size=-1, lower=False,
				 lstm_dimension=None, optimiser=None, patience=None,
				 batch_size=32, epochs=100):
		'''
		:param tokeniser: The tokeniser function to tokenise the text.
		:param embeddings: The embeddings to use
		:param pad_size: The max number of tokens to use per sequence. If -1 \
		use the text sequence in the training data that has the most tokens as \
		the pad size.
		:param lower: Whether to lower case the words being processed.
		:param lstm_dimension: Output of the LSTM layer. If None it is the \
		which is the default then the dimension will be the same as the \
		embedding vector.
		:param optimiser: Optimiser to for the LSTM default is SGD. Accepts any \
		`keras optimiser <https://keras.io/optimizers/>`_.
		:param patience: Wether or not to use EarlyStopping default is not \
		stated by the None value. If so this is the patience value e.g. 5.
		:param batch_size: Number of samples per gradient update
		:param epochs: Number of epochs to train the model.
		:type tokeniser: function
		:type embeddings: :py:class:`tdparse.word_vectors.WordVectors` instance
		:type pad_size: int. Default -1
		:type lower: bool. Default False
		:type lstm_dimension: int. Default None
		:type optimiser: Keras optimiser. Default None which uses SDG.
		:type patience: int. Default None.
		:type batch_size: int. Default 32
		:type epochs: int. Default 100.
		:returns: The instance of LSTM
		:rtype: :py:class:`tdparse.models.tdlstm.LSTM`
		'''

		self.tokeniser = tokeniser
		self.embeddings = embeddings
		self.pad_size = pad_size
		self.test_pad_size = 0
		self.lower = lower
		self.model = None
		self.lstm_dimension = lstm_dimension
		if self.lstm_dimension is None:
			self.lstm_dimension = self.embeddings.embedding_matrix.shape[1]
		self.optimiser = optimizers.SGD(lr=0.01)
		if optimiser is not None:
			self.optimiser = optimiser
		self.patience = patience
		self.batch_size = batch_size
		self.epochs = epochs

	def process_text(self, texts, max_length, padding='pre', truncate='pre'):
		'''
		Given a list of Strings, tokenised the text and lower case if set and then
		convert the tokens into a integers representing the tokens in the
		embeddings. Lastly it pads the data based on the max_length param.

		If the max_length is smaller than the sentences size it truncates the
		sentence. If max_length = -1 then the max_length is that of the longest
		sentence in the texts.

		:params texts: list of Strings
		:params max_length: How many tokens a sentence can contain. If it is \
		-1 then it uses the sentence with the most tokens as the max_length param.
		:params padding: which side of the sentence to pad: `pre` beginning, \
		`post` end.
		:params truncate: which side of the sentence to truncate: `pre` beginning \
		`post` end.
		:type texts: list
		:type max_length: int
		:type padding: String. Either `pre` or `post` default `pre`
		:type truncate: String. Either `pre` or `post` default `pre`
		:returns: A tuple of length 2 containg: 1. The max_length parameter, 2. \
		A list of a list of integers that have been padded.
		:rtype: tuple
		'''

		if max_length == 0:
			raise ValueError('The max length of a sequence cannot be zero')
		elif max_length < -1:
			raise ValueError('The max length has to be either -1 or above '\
							 'zero not {}'.format(max_length))

		# Process the text into integers based on the embeddings given
		all_sequence_data = []
		max_sequence = 0
		for text in texts:
			sequence_data = []
			tokens = self.tokeniser(text)
			for token in tokens:
				if self.lower:
					token = token.lower()
				sequence_data.append(self.embeddings.word2index[token])
			sequence_length = len(sequence_data)
			if sequence_length > max_sequence:
				max_sequence = sequence_length
			all_sequence_data.append(sequence_data)
		if max_sequence == 0:
			raise ValueError('The max sequence length is 0 suggesting no '\
							 'data was provided for training or testing')
		# Pad the sequences

		# If max pad size is set and training the model set the test_pad_size
		# to max sequence length
		if max_length == -1:
			max_length = max_sequence
		return (max_length,
				preprocessing.sequence.pad_sequences(all_sequence_data,
													 maxlen=max_length,
													 dtype='int32', padding=padding,
													 truncating=truncate))
	@staticmethod
	def validation_split(train_data, train_y, validation_size=0.2,seed=1234):
		'''
		:param train_data: Training features. Specifically a list of dict like \
		structures that contain `text` key.
		:param train_y: Target values
		:validation_size: The fraction of the training data to be set aside \
		for validation data
		:type train_data: list
		:type train_y: list
		:type validation_size: float Default 0.2
		:returns: A tuple of length 4 which contains: 1. Training features, \
		2. Training Target Values, 3. Validation features, 4. Validation Targets
		:rtype: tuple
		'''

		splitter = ShuffleSplit(n_splits=1, test_size=validation_size,random_state=seed)
		train_data = np.asarray(train_data)
		train_y = np.asarray(train_y)
		for train_indexs, validation_index in splitter.split(train_data, train_y):
			return (train_data[train_indexs], train_y[train_indexs],
					train_data[validation_index], train_y[validation_index])

	def create_training_data(self, train_data, train_y, validation_size=0.2,seed=1234):
		'''
		:param train_data: Training features. Specifically a list of dict like \
		structures that contain `text` key.
		:param train_y: Target values
		:validation_size: The fraction of the training data to be set aside \
		for validation data
		:type train_data: list
		:type train_y: list
		:type validation_size: float Default 0.2
		:returns: A tuple of length 2 where the first value is a list of \
		Integers that reprsent the words in the text features where each Integer \
		corresponds to a Word Vector in the embedding vector. Second value are \
		the target values. Both lists in the tuples contain training data in the \
		first part of the list and the second part of the list based on the \
		validation split contains the validation data.
		:rtype: tuple
		'''

		# Create train and validation splits
		data = self.validation_split(train_data, train_y,
									 validation_size=validation_size,seed=seed)
		temp_train, temp_train_y, validation_data, validation_y = data
		# Convert from a sequence of dictionaries into texts and then integers
		# that represent the tokens in the text within the embedding space.
		all_train_text = [data['text'] for data in temp_train]
		self.test_pad_size, sequence_train_data = self.process_text(all_train_text,
																	self.pad_size)
		all_val_text = [data['text'] for data in validation_data]
		_, sequence_val_data = self.process_text(all_val_text, self.test_pad_size)
		# Stack the validation data with the training data to complie with Keras.
		all_data = np.vstack((sequence_train_data, sequence_val_data))
		all_y = np.hstack((temp_train_y, validation_y))
		# Convert the training true values into categorial data format
		num_classes = np.unique(all_y).shape[0]
		all_y = keras.utils.to_categorical(all_y, num_classes=num_classes)\
				.astype(np.float32)
		return all_data, all_y

	@staticmethod
	def cross_val(train_data, train_y, lstm_model,
				  validation_size=0.2, cv=5, scorer=None,
				  reproducible=True,seed=1234,bias_reg=0,input_reg=0):
		'''
		:param train_data: Training features. Specifically a list of dict like \
		structures that contain `text` key.
		:param train_y: Target values of the training data
		:param lstm_model: Model that has a fit and predict function
		:param valida
		:param seed: random seed for partitioning
		'''
		splitter = KFold(n_splits=cv,shuffle=True,random_state=seed)
		train_data = np.asarray(train_data)
		train_y = np.asarray(train_y)
		all_predictions = []
		scores = []
		for train_index, test_index in splitter.split(train_data, train_y):
			temp_train_data = train_data[train_index]
			temp_train_y = train_y[train_index]
			temp_test_data = train_data[test_index]
			temp_test_y = train_y[test_index]
			lstm_model.fit(temp_train_data, temp_train_y,
						   validation_size=validation_size,
						   reproducible=reproducible,seed=seed,bias_reg=regularizers.l2(bias_reg),input_reg=regularizers.l2(input_reg))
			predictions = lstm_model.predict(temp_test_data)
			predictions = np.argmax(predictions, axis=1)
			if scorer is not None:
				num_classes = np.unique(predictions).shape[0]
				temp_test_y = keras.utils.to_categorical(temp_test_y,
														 num_classes=num_classes)
				temp_test_y = np.argmax(temp_test_y, axis=1)
				scores.append(scorer(temp_test_y, predictions))
			all_predictions.append(predictions)
		return (all_predictions, scores)




	def fit(self, train_data, train_y, validation_size=0.2, verbose=1,
			reproducible=True, embedding_layer_trainable=False,seed=1234,bias_reg=0,input_reg=0):
		'''
		:param train_data: Training features. Specifically a list of dict like \
		structures that contain `text` key.
		:param train_y: Target values of the training data
		:param validation_size: The fraction of the training data to be set aside \
		for validation data
		:param verbose: Verbosity of the traning the model. 0=silent, \
		1=progress bar, and 2=one line per epoch
		:param reproducible: Wether or not to make the model to be reproducible. \
		This will slow done the training.
		:type train_data: list
		:type train_y: list
		:type validation_size: float. Default 0.2
		:type verbose: int. Default 1
		:type reproducible: bool. Default True.
		:returns: Nothing. The self.model will be fitted.
		:rtype: None
		'''
	
		self.model = None
		np.random.seed(seed)
		rn.seed(seed)
		tf.set_random_seed(seed)
		if reproducible:
			os.environ['PYTHONHASHSEED'] = '0'
			np.random.seed(seed)
			rn.seed(seed)
			# Forces tensorflow to use only one thread
			session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
										  inter_op_parallelism_threads=1)
			tf.set_random_seed(seed)

			sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
			K.set_session(sess)

		# Data pre-processing
		all_data, all_y = self.create_training_data(train_data, train_y,
													validation_size)
		num_classes = all_y.shape[1]

		# LSTM model
		embedding_matrix = self.embeddings.embedding_matrix
		vocab_size, vector_size = embedding_matrix.shape
		input_layer = Input(shape=(self.test_pad_size,))
		embedding_layer = Embedding(input_dim=vocab_size, output_dim=vector_size,
									input_length=self.test_pad_size,
									trainable=embedding_layer_trainable,
									weights=[embedding_matrix])(input_layer)
		lstm_layer = LSTM(self.lstm_dimension,bias_regularizer=bias_reg,kernel_regularizer=input_reg)(embedding_layer)
		predictions = Dense(num_classes, activation='softmax',)(lstm_layer)
		model = models.Model(inputs=input_layer, outputs=predictions)

		model.compile(optimizer=self.optimiser, metrics=['accuracy'],
					  loss='categorical_crossentropy')
		with tempfile.NamedTemporaryFile() as weight_file:
			# Set up the callbacks
			callbacks = None
			if self.patience is not None:
				model_checkpoint = ModelCheckpoint(weight_file.name,
												   monitor='val_loss',
												   save_best_only=True,
												   save_weights_only=True,
												   mode='min')
				early_stopping = EarlyStopping(monitor='val_loss', mode='min',
											   patience=self.patience)
				callbacks = [early_stopping, model_checkpoint]
			model.fit(all_data, all_y, validation_split=validation_size,
					  epochs=self.epochs, callbacks=callbacks,
					  verbose=verbose, batch_size=self.batch_size)
			# Load the best model from the saved weight file
			if self.patience is not None:
				model.load_weights(weight_file.name)
		self.model = model

	def predict(self, test_data):
		'''
		:param test_y: Test features. Specifically a list of dict like \
		structures that contain `text` key.
		:type test_y: list
		:returns: A list of predicted samples for the test data.
		:rtype: numpy.ndarray
		'''

		if self.model is None:
			raise ValueError('The model has not been fitted please run the '\
							 '`fit` method.')
		# Convert from a sequence of dictionaries into texts and then integers
		# that represent the tokens in the text within the embedding space.
		all_test_text = [data['text'] for data in test_data]
		_, sequence_test_data = self.process_text(all_test_text,
												  self.test_pad_size)
		return self.model.predict(sequence_test_data)
