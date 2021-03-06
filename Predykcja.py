﻿﻿#Kategoryzacja utworów muzycznych na podstawie tekstu za pomocą modelu k-średnich

#Załadowanie pakietów

import pickle

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, render_template, jsonify

app = Flask(__name__, template_folder="/app/templates")


#Załadowanie modelu
loaded_model = pickle.load(open("model.sav", 'rb'))
#Załadowanie słownika
loaded_vec = pickle.load(open("vectorizer.pk", 'rb'))


#Przetwarzanie wstępne
stopWords = set(stopwords.words('english'))
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')


def preprocessing(txt):
    wordsFiltered = []
    words = tokenizer.tokenize(txt)
    for w in words:
        if w.lower() not in stopWords and not w.isdigit():
            wordsFiltered.append(ps.stem(w.lower()))
    return corpa


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api', method=['POST'])
def service():
    #Pobranie danych
    data = request.json['claim']
    
    #Przetwarzanie wstępne
    stopWords = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    data_prepro = preprocessing(data)
    data_str = [" ".join(words) for words in data_prepro]


    #Wyznaczenie kategorii
    dic = {0: 'hiphop', 1: 'pop', 2: 'metal', 3: 'rock'}
    vec_words = loaded_vec.transform(data_str)
    predicted_cat = loaded_model.predict(vec_words)
    song = dic[predicted_cat]
    
return jsonify({'success': True, 'result':{'category' : song}})

if __name__== '__main__':
    app.run(host='127.0.0.1', port = 8080, debug = True)



