import sys
import string
import pickle
wordprobs= {}

mapper= {'True':1, 'Fake':-1, 'Pos': 1, 'Neg':-1}


stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]



review_dict = dict()

vanilla_weights_f1 = dict()
vanilla_weights_f2 = dict()
vanilla_bias_f1 = 0
vanilla_bias_f2 = 0
words_set= set()

def isstopword(word):
	global stopwords
	if word in stopwords:
		return False
	else:
		return True



def preprocess(filename):
	f= open(filename,"r")
	
	for review in f:
		review = review.strip()
		#print(review)

		# Eliminate punctuation

		temp_list= review.split(" ")

		review = review.translate(str.maketrans('','',string.punctuation)).lower()
		tokList = review.split()

		# Remove stopwords
		tokList = list(filter(isstopword, tokList))

		#l1 = list(filter(lambda x: stopwords, review))
		
		identifier  = temp_list[0]
		truth = temp_list[1]
		posneg = temp_list[2]

		tokList = tokList[3:]
		review_dict[identifier]= {}

		for word in tokList:
			words_set.add(word)
			if word not in review_dict[identifier]:
				review_dict[identifier][word]=1
			else:
				review_dict[identifier][word]+=1
			
	
	for word in words_set:
		vanilla_weights_f2[word]=0
		vanilla_weights_f1[word]=0
		
	f.close()


def storeModel():
	global vanilla_weights_f1, vanilla_weights_f2, vanilla_bias_f1, vanilla_bias_f2

	f1 = open('vanillamodel.txt', 'w')
	model = open("vanillamodel.pickle",'wb')
	f1.write('Weights for True/Fake feature:\n')
	f1.write(str(vanilla_weights_f1))

	f1.write('\n*********************************************************************\n')
	f1.write('Bias for True/Fake feature: '+str(vanilla_bias_f1))

	f1.write('\nWeights for Pos/Neg feature:\n')
	f1.write(str(vanilla_weights_f2))

	f1.write('\n*********************************************************************\n')
	f1.write('Bias for Pos/Neg feature: '+str(vanilla_bias_f2))



	

	f1.close()
	pickle.dump(vanilla_weights_f1, model)
	pickle.dump(vanilla_bias_f1, model)
	pickle.dump(vanilla_weights_f2, model)
	pickle.dump(vanilla_bias_f2, model)




def vanilla(filename):

	global vanilla_weights_f1, vanilla_weights_f2, vanilla_bias_f1, vanilla_bias_f2
	f= open(filename,"r")

	#print("initial weights: ")
	#print(weights_dict)

	for review in f:
		review = review.strip()
		temp_list= review.split(" ")
		review = review.translate(str.maketrans('','',string.punctuation)).lower()
		tokList = review.split()
		# Remove stopwords
		tokList = list(filter(isstopword, tokList))
		identifier  = temp_list[0]
		truth = temp_list[1]
		posneg = temp_list[2]
		y1= mapper[truth]
		y2= mapper[posneg]
		#print(str(y1)+" , "+str(y2))

		tokList=tokList[3:]


		temp_dict = dict()
		for word in tokList:
			if word not in temp_dict:
				temp_dict[word]=1
			else:
				temp_dict[word]+=1

		activation_val = 0
		#print(temp_dict)

		for key_word in temp_dict:
			if key_word not in vanilla_weights_f1:
				print("yes")
			activation_val+= temp_dict[key_word]*vanilla_weights_f1[key_word]
		activation_val+= vanilla_bias_f1

		if activation_val*y1 <=0:
			for key_word in temp_dict:
				vanilla_weights_f1[key_word]= vanilla_weights_f1[key_word]+y1*temp_dict[key_word]
			vanilla_bias_f1+= y1


		activation_val=0
		for key_word in temp_dict:
			if key_word not in vanilla_weights_f2:
				print("yes")
			activation_val+= temp_dict[key_word]*vanilla_weights_f2[key_word]
		activation_val+= vanilla_bias_f2

		if activation_val*y2 <=0:
			for key_word in temp_dict:
				vanilla_weights_f2[key_word]= vanilla_weights_f2[key_word]+y2*temp_dict[key_word]
			vanilla_bias_f2+= y2
		#print(weights_dict['way'])

	f.close()
	#print(weights_dict['way'])
	#print(vanilla_weights_f1['way'])
	#print(vanilla_weights_f2['way'])
	#print(vanilla_bias_f1)
	#print(vanilla_bias_f2)







		



		
if __name__=='__main__':
	preprocess(sys.argv[1])

	vanilla(sys.argv[1])

	storeModel()
	