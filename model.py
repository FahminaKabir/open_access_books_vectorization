# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:37:08 2020

@author: Fahmina Kabir, Israt Tasnim
"""

#%%
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
import os
import json
import string

#%%
def createContentlist():
    
    fileList = []
    contentList = []

    # List all files in a directory using os.listdir
    pathname = 'E:\\4.1\\thesisGroup9\\booksForDataset\\text\\'
    for files in os.listdir(pathname):
        if os.path.isfile(os.path.join(pathname, files)):
            fileList.append(files)
    
    for files in fileList:
        
        x =  os.path.join(pathname, files)
              
        f=open(x , "r", encoding="utf8")
        if f.mode == 'r':
            contents = f.read()
            contentList.append(contents)
    return contentList
    print(contentList)
    
#%%
conlist = createContentlist()
def getSentences(contlist):
    listlistOfWords = []
    listOfSentence = []
    
    for content in contlist:
        lower = (content.lower())
        l = sent_tokenize(lower)
        listOfSentence = listOfSentence + l
    #print(listOfSentence)

    for sentence in listOfSentence:
        tokenizer = RegexpTokenizer(r'\w+') #To separate a sentence into words without puctuation, we use RegexpTokenizer(r'\w+') as our tokenizer. Tokenization without punctuation is useful for text analysis.
        wordlist = tokenizer.tokenize(sentence)
        listlistOfWords.append(wordlist)
        
    return listlistOfWords
    #print(listlistOfWords)
    
sentences = getSentences(conlist)
print('corpus created')

#%%

nltk.download('stopwords')
from nltk.corpus import stopwords

# All english stopwords list
english_stops = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

for i in range(len(sentences)):
  x = [word for word in sentences[i] if word not in english_stops]
  
  y = [w for w in x if w.isalpha()]
  
  for j in range(len(y)):
    y[j] = lemmatizer.lemmatize(y[j]) 
  sentences[i] = y

#%%
  
corpus_json = {'corpus' : sentences ,
               'info' : { 'language' : 'english',
                         'source' : 'openlibra'
                   }
               }

with open('E:\\4.1\\thesisGroup9\\GIT\\ThesisGroup9\\word2vec\\corpus.json', 'w', encoding='utf-8') as file:
    json.dump(corpus_json , file, ensure_ascii=False) 


#%%
    	
from gensim.models import Word2Vec
# define training data

with open('E:\\4.1\\thesisGroup9\\GIT\\ThesisGroup9\\word2vec\\corpus.json', "r", encoding='utf-8') as file:
    data = json.load(file)

sentences = data['corpus']

# train model
model = Word2Vec(sentences, min_count=1)

#print(model)
# summarize vocabulary
words = list(model.wv.vocab)
#print(words)

# save model
model.save('E:\\4.1\\thesisGroup9\\GIT\\ThesisGroup9\\word2vec\\my_word2vec_model2.bin')
# load model
new_model = Word2Vec.load('E:\\4.1\\thesisGroup9\\GIT\\ThesisGroup9\\word2vec\\my_word2vec_model2.bin')

vocabu = len(model.wv.vocab)
print('Vocabulary :' + str(vocabu))