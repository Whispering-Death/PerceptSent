import sys
import string
import pickle
import math
wordprobs= {}

mapper= {'True':1, 'Fake':-1, 'Pos': 1, 'Neg':-1}



stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


identifier_list=[]
predicted_labels_list=[]
review_dict = dict()

vanilla_weights_f1 = dict()
vanilla_weights_f2 = dict()
vanilla_bias_f1 = 0
vanilla_bias_f2 = 0
words_set= set()


def writefile():
	outputfile = open('percepoutput.txt','w')

	for index in range(len(identifier_list)):
		outputfile.write(identifier_list[index]+' '+predicted_labels_list[index][0]+' '+predicted_labels_list[index][1]+'\n')



def calculateAccuracy(file):
	global predicted_labels_list,identifier_list
	f= open(file,'r')
	count = 0
	print(predicted_labels_list)

	index=0
	for line in f:
		tokens = line.split()
		#print(tokens)
		if tokens[1]==predicted_labels_list[index][0] and tokens[2]==predicted_labels_list[index][1]:
			count+=1
		index+=1
	print(count)

def isstopword(word):
	global stopwords
	if word in stopwords:
		return False
	else:
		return True

def readfile(modelname, filename):

	global vanilla_bias_f2, vanilla_bias_f1, vanilla_weights_f1, vanilla_weights_f2
	f= open(filename,'r')

	f1 = open(modelname,'rb')
	

	vanilla_weights_f1 = pickle.load(f1)
	vanilla_bias_f1 = pickle.load(f1)
	vanilla_weights_f2 = pickle.load(f1)
	vanilla_bias_f2 = pickle.load(f1)
	#print(wordprobs)
	for review in f:
		review = review.strip()
		temp_list= review.split(" ")
		
		review = review.translate(str.maketrans('','',string.punctuation)).lower()
		tokList = review.split()
		identifier  = temp_list[0]
		

		identifier_list.append(identifier)
		tokList = list(filter(isstopword, tokList))
		
		tokList=tokList[1:]


		temp_dict = dict()
		for word in tokList:
			if word not in temp_dict:
				temp_dict[word]=1
			else:
				temp_dict[word]+=1

		activation_val=0

		for word in temp_dict:
			if word in vanilla_weights_f1:
				activation_val= activation_val+ vanilla_weights_f1[word]*temp_dict[word]
		activation_val+=vanilla_bias_f1

		

		predicted_label1= 'True'

		predicted_label2 = 'Pos'

		if activation_val<=0:
			predicted_label1='False'



		activation_val=0
		activation_val=0

		for word in temp_dict:
			if word in vanilla_weights_f2:
				activation_val= activation_val+ vanilla_weights_f2[word]*temp_dict[word]
		activation_val+=vanilla_bias_f2


		if activation_val<=0:
			predicted_label2='Neg'


		predicted_labels_list.append((predicted_label1, predicted_label2))
	#print(predicted_labels_list)
	f.close()
	f1.close()



			

if __name__=='__main__':
	readfile(sys.argv[1], sys.argv[2])
	#calculateAccuracy('dev-key.txt')
	writefile()
