import os; os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

import math
import string
import numpy as np
import pickle


class Model:
	def __init__(self,Training_File):
		self.Model=None
		self.X_TRAIN=None
		self.Y_TRAIN=None

		Training_File=open(Training_File,'r',encoding="utf8")
		self.Training_Text=Training_File.read()
		Training_File.close()
		

		self.BOOL_TRAINED=False

		self.encoder_tokenizer=Tokenizer()
		self.LENGTH_VOCABULARY=0
		self.LENGTH_OF_SEQUENCE=0
		self.Predicted_Sentences=list()
		self.Tokens=list()
		self.Loaded_Tokenizer=None
		self.Loaded_Model=None

	def clean_text(self,document):
		#replaces clean_doc()
		document=document.replace("--"," ")
		Tokens=document.split()
		Tokens_=str.maketrans(" "," ",string.punctuation)
		
		T=list()
		for Token in Tokens:
			T.append(Token.translate(Tokens_))
		Tokens=T 
		Tokens=[x for x in Tokens if x.isalpha()]
		Tokens=[x.lower() for x in Tokens]
		return Tokens

	#replaces	def save_doc(lines, filename):
	def save_modified_file(self,Lines,file):
		Lines= '\n'.join(Lines)
		File=open(file,'w')
		File.write(Lines)
		File.close()

	def PREPROCESSING_(self):
		File=self.Training_Text
		self.Tokens=self.clean_text(File)
		Tokens=self.Tokens
		Length_Of_A_Sentence=51
		List_Of_Sequences=list()
		i=Length_Of_A_Sentence
		while(i<len(Tokens)):
			ini=i-(Length_Of_A_Sentence)
			till=i
			A_SEQUENCE=Tokens[ini:till]
			A_LINE=" ".join(A_SEQUENCE)
			List_Of_Sequences.append(A_LINE)
			i=i+1
		print(str("Total Sequences")+str(len(List_Of_Sequences)))
		self.save_modified_file(List_Of_Sequences,"Training_Sequences.txt")

	def MAKE_X_Y_FOR_TRAIN(self):
		Training_File_Final=open("Training_Sequences.txt","r",encoding = "ISO-8859-1")
		Training_Data_Final=Training_File_Final.read()
		Training_File_Final.close()

		ALL_LINES=Training_Data_Final.split("\n")
		self.encoder_tokenizer.fit_on_texts(ALL_LINES)
		ALL_SEQUENCES=self.encoder_tokenizer.texts_to_sequences(ALL_LINES)
		self.LENGTH_VOCABULARY=len(self.encoder_tokenizer.word_index)+1
		ALL_SEQUENCES=np.array(ALL_SEQUENCES)
		#self.LENGTH_OF_SEQUENCE=ALL_SEQUENCES[:,:-1].shape[1]
		print(str(self.LENGTH_OF_SEQUENCE)+str("Data Processing Completed"))

		self.X_TRAIN=ALL_SEQUENCES[:,:-1]
		self.Y_TRAIN=ALL_SEQUENCES[:,-1]
		self.Y_TRAIN=to_categorical(self.Y_TRAIN,self.LENGTH_VOCABULARY)
		self.LENGTH_OF_SEQUENCE=self.X_TRAIN.shape[1]
	def DEFINE_MODEL(self):
		
		self.Model=Sequential()
		self.Model.add(Embedding(self.LENGTH_VOCABULARY,50,input_length=self.LENGTH_OF_SEQUENCE))
		self.Model.add(LSTM(100,return_sequences=True))
		self.Model.add(LSTM(100))
		self.Model.add(Dense(100,activation='relu'))
		self.Model.add(Dense(self.LENGTH_VOCABULARY,activation='softmax'))
		self.Model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
		
	def MODEL_TRAIN(self):
		########## EPOCHS HERE!! FIT HERE!!
		print(self.X_TRAIN)
		print(self.Y_TRAIN)
		self.X_TRAIN=np.array(self.X_TRAIN)
		self.Y_TRAIN=np.array(self.Y_TRAIN)
		self.Model.fit(self.X_TRAIN,self.Y_TRAIN,batch_size=128,epochs=20)
		 ############################### epochs
		self.Model.save("Model_Weights_1.h5")
		pickle.dump(self.encoder_tokenizer,open('Tokenizer_1.pkl','wb'))
	
	def LOAD_WEIGHTS(self):
		
		self.Loaded_Model=load_model('Model_Weights_1.h5')
		self.Loaded_Tokenizer=pickle.load(open('tokenizer_1.pkl','rb'))
		
		print("Weights have been Loaded")

	def GET_NEXT_WORD(self,Input_Text,GET_TOP_N):
		local_model=self.Loaded_Model
		local_tokenizer=self.Loaded_Tokenizer

		encoded=self.encoder_tokenizer.texts_to_sequences([Input_Text])[0]
		encoded=pad_sequences([encoded],maxlen=50,truncating='pre')
		predictions=np.argsort(-np.array(local_model.predict_proba(encoded)))


		WORDS=list()
		Probabilites_OF_WORDS=list()

		if(GET_TOP_N>predictions.shape[1]):
			GET_TOP_N=predictions.shape[1]-1

		x=0
		while(x<GET_TOP_N):
			y=0
			for word,index in local_tokenizer.word_index.items():
				if(index==predictions[0][x]):
					WORDS.append(word)
					break
			x=x+1

		return WORDS

	def GENERATE_SENTENCES(self,Sentence,WordsLeft):
		if(len(WordsLeft)==0):
			self.Predicted_Sentences.append(Sentence)
		else:
			SEED=Sentence+""
			NEXT_SEED=self.GET_NEXT_WORD(SEED,1000)
			for word in WordsLeft:
				if(word in NEXT_SEED):
					local_words_left=WordsLeft.copy()
					local_words_left.remove(word)
					local_next_sentence=Sentence+" "+word
					self.GENERATE_SENTENCES(local_next_sentence,local_words_left)


	def UNJUMBLE(self,FILE_STR,INPUT_NUM):
		self.ALL_WORDS=set(self.Tokens)
		Testing_File=FILE_STR
		TOTAL_COUNT=0
		CORRECT_COUNT=0
		INCORRECT_COUNT=0
		WORD_ABSENT_COUNT=0
		RESULTS=[0]*(math.factorial(INPUT_NUM)+1)

		for LINE in open(Testing_File,encoding="utf-8"):
			print('-'*100)
			print("The Original Sentence: ",LINE)
			TOTAL_COUNT=TOTAL_COUNT+1
			Input_Line=LINE
			CORRECT_SENTENCE=LINE
			CORRECT_SENTENCE=CORRECT_SENTENCE.strip().lower()

			Input_Line=Input_Line.strip().lower()
			Input_Words=Input_Line.split()

			VALID=True


			for a_word in Input_Words:
				if(not(a_word in self.ALL_WORDS)):
					print(str("THIS WORD")+str(a_word))
					WORD_ABSENT_COUNT=WORD_ABSENT_COUNT+1
					VALID=False
					break
			print()
			if(VALID):
				#Output_Sentences=list()
				for x in range(len(Input_Words)):
				
					local_input_words=Input_Words.copy()
					start=local_input_words[x]
					local_input_words.remove(start)
					self.GENERATE_SENTENCES(start,local_input_words)

				if(len(self.Predicted_Sentences)!=0):
					print("The Predictions are: ")
					print(self.Predicted_Sentences)
				else:
					print()
				RESULTS[len(self.Predicted_Sentences)]+=1

				if(CORRECT_SENTENCE in self.Predicted_Sentences):
					CORRECT_COUNT=CORRECT_COUNT+1
					print()
					print("WAS FOUND!")
					print()
				else:
					INCORRECT_COUNT=INCORRECT_COUNT+1
				self.Predicted_Sentences=list()
			print()

		print("TOTAL SENTENCES COUNT:  ",TOTAL_COUNT)
		print("CORRECT PREDICTION COUNT:  ",CORRECT_COUNT)
		print("INCORRECT PREDICTION COUNT:  ",INCORRECT_COUNT)
		print("WORD NOT PRESENT COUNT:  ",WORD_ABSENT_COUNT)
		print("ACCURACY IS: ")
		print(round((CORRECT_COUNT/TOTAL_COUNT)*100,2))




if(__name__=="__main__"):
	MODEL=Model("Training_File.txt")
	MODEL.PREPROCESSING_()
	MODEL.MAKE_X_Y_FOR_TRAIN()
	MODEL.DEFINE_MODEL()
	#MODEL.MODEL_TRAIN()
	##SAVE MODEL
	MODEL.LOAD_WEIGHTS()
	MODEL.UNJUMBLE("Test2.txt",1)

