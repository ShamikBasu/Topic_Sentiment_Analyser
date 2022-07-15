import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
from flask_cors import CORS

def cleanPunctuations(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepOnlyAlphabets(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def predClass(predNum):
    my_topic_dict = {
        0 : 'CONCLUSION',
        1 : 'VISUAL MEDIA',
        2 : 'ENDEAVOR',
        3 : 'ENTERTAINMENT',
        4 : 'SOLICITOUSNESS',
        5 : 'BLITHE',
        6 : 'CAROUSING',
        7 : 'CONCERNED',
        8 : 'MUSIC',
        9 : 'INDISPOSED',
        10 : 'GRATIFIED',
        11 : 'NOSTALGIC',
        12 : 'REVERT',
        13 : 'CELEBRATION',
        14 : 'AFFLICTED',
        15 : 'DESIRE',
        16 : 'RESENT',
        17 : 'SLEEP',
        18 : 'SOCIAL MEDIA',
        19 : 'ACCOMPLISH'
    }
    return my_topic_dict[predNum]

def sentimentAnalyzer(inputText): 
    sent = sid.polarity_scores(inputText)
    if(sent['compound']<0):
       sentType = 'Negative'
    elif(sent['compound']>0):
       sentType = 'Positive'
    else:
       sentType = 'Neutral'	
	    	   
    sentiment ={
	'sentimentType': sentType,
        'score': str(sent['compound'])
	}	
    return sentiment

import json
from flask import Flask, jsonify, request

app = Flask(__name__)
CORS(app) #any url can access this
#app.config['CORS_HEADERS'] = 'Content-Type'
@app.route('/parentClassifier/', methods=['POST'])
def getParentCategory():
    record = json.loads(request.data)
    #print(record)
    input_post = record['input_post']
    #print("INPUT POST::::",input_post)
    test = input_post
    test = cleanPunctuations(test)
    #print(test)
    test = keepOnlyAlphabets(test)
    #print(test)
    test= removeStopWords(test)
    #print(test)
    #test= stemming(test)
    #print(test)
    sgd = joblib.load('ParentSGD.pkl')
    prediction = sgd.predict([test])
    topic_type = predClass(prediction[0])
    sent = sentimentAnalyzer(test)
    res = {
    'input text' : input_post, 
    'type': topic_type,
    'type number' : str(prediction[0]),
    'sentiment' : sent
	}
    #print(prediction[0])
    return res



app.run(debug=True)