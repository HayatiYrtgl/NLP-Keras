import numpy as np
import json
from keras.models import Sequential
from keras.layers import *
from keras.regularizers import L1L2
from keras.optimizer_v2.adam import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import tokenizer_from_json
from keras.metrics import Recall, Precision, AUC
from pandas import DataFrame

# preprocesing class
class PreProcessing:
	
	# const
	def __init__(self, path=None):
		
		# constructor variables
		self.tokenizer = Tokenizer()
		self.file_path = path
		self.corpus = None
		self.total_words = 100
		self.max_sequence_len = 1000
		self.input_sequence = []
		self.predictor = None
		self.label = None
		
		# run functions
		# corpus
		self.corpus_processing()
		
		# tokenization
		self.tokenization()
		
		# save
		self.tokenizer_saver()
	
	# corpus processing
	def corpus_processing(self):
		
		# data read and split
		data = open(self.file_path, "r", encoding="utf-8").read()
		
		self.corpus = data.split("\n")
		
	# tokenization
	def tokenization(self):
		
		# fit on text
		self.tokenizer.fit_on_texts(self.corpus)
		
		# total words
		self.total_words = len(self.tokenizer.word_index) + 1
		
		# for loops
		for line in self.corpus:
			
			token_list = self.tokenizer.texts_to_sequences([line])[0]
			
			# n_gram sequence
			for i in range(1, len(token_list)):
				
				n_gram_sequence = token_list[:i+1]
				
				# add the input sequence
				self.input_sequence.append(n_gram_sequence)
				
		# max sequence length
		self.max_sequence_len = max([len(i) for i in self.input_sequence])
		
		# input sequence to np array
		self.input_sequence = np.array(pad_sequences(self.input_sequence, maxlen=self.max_sequence_len))
		
		# predictor and labels
		self.predictor, self.label = self.input_sequence[:, :-1], self.input_sequence[:, -1]
		
		# label to categorical
		self.label = to_categorical(self.label, num_classes=self.total_words)
		
	# save to tokenizer
	def tokenizer_saver(self):
		
		tokenizer_to_json = self.tokenizer.to_json()
		
		with open("data/tokenizer.json", "w", encoding="utf-8") as f:
			
			# connected list
			c_list = [tokenizer_to_json, self.max_sequence_len-1]
			
			f.write(json.dumps(c_list, ensure_ascii=False))
			
		print("Saved")
		

# get tokenizer from json
class GetTokenizer:
	
	# const
	def __init__(self, path=None):
		
		self.path = path
		self.tokenizer = None
		
		with open(self.path) as file:
			
			data = json.load(file)
			
			self.tokenizer = tokenizer_from_json(data)

#  model creation
class CreateModel:
	
	def __init__(self, max_len, total_words, epoch, predictors, label, model_save_path, level=None):
		
		#  find out the level of model
		self.level = level.lower()
		self.max_len = max_len-1
		self.total_words = total_words
		self.epoch = epoch
		self.optimizer = Adam(learning_rate=0.001)
		self.preictors = predictors
		self.label = label
		self.save_path = model_save_path
		
		# model
		self.model = None
		
		if self.level == "m":
			
				# run the func
				
				self.model = self.m_model()
				
				# compile and fit
				self.compiler()
		
		elif self.level == "l":
			
			# run the func
			
			self.model = self.l_model()
	
			self.compiler()
		
		else:
			
			self.model = self.s_model()
			
			self.compiler()
			
	# small  model
	def s_model(self):
		
		# sequential
		model = Sequential()
		
		# layers
		model.add(Embedding(self.total_words, 50, input_length=self.max_len-1))
		
		model.add(Bidirectional(LSTM(75)))
		
		model.add(Dense(self.total_words, activation="softmax"))
		
		return model
	
	# medium model		
	def m_model(self):
		
		# sequential
		model = Sequential()
		
		# layers
		model.add(Embedding(self.total_words, 100, input_length=self.max_len-1))
		
		model.add(Bidirectional(LSTM(150)))
		
		model.add(Dense(self.total_words, activation="softmax"))
		
		return model
		
	# larrge model
	def l_model(self):
		
			model = Sequential()
			
			model.add(Embedding(self.total_words, 200, input_length=self.max_len-1))
				
			model.add(Bidirectional(LSTM(300, return_sequences=True, recurrent_regularizer=L1L2(l1=0.01, l2=0.01))))
			
			model.add(LSTM(150, recurrent_regularizer=L1L2(l1=0.01, l2=0.01)))
			
			model.add(Dense(self.total_words, activation="softmax"))
			
			return model
			
	
	# compiler
	def compiler(self):
		
		self.model.summary()
		
		self.model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=["accuracy", AUC(), Precision(), Recall()])
		
		history = self.model.fit(self.preictors, self.label, epochs=self.epoch, verbose=1)
		
		# history save
		self.model.save(f"{self.save_path}.h5")
		
		# save dataframe
		df = DataFrame(history.history)
		
		df.to_excel(f"{self.save_path}.xlsx")
		
# preprocess calss
pre_process_class = PreProcessing(path="data/kadin_tirad_veri_seti.txt")

create_model_class = CreateModel(max_len=pre_process_class.max_sequence_len, total_words=pre_process_class.total_words, epoch=125, predictors=pre_process_class.predictor, label=pre_process_class.label, model_save_path="data/new_model", level="s")
		
		
		
		
				
						
								
										
												
																