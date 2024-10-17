# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:37:08 2020

@author: Fahmina Kabir, Israt Tasnim
"""

#%%

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import pandas as pd
from gensim.models import KeyedVectors

#%%

filename = 'E:\\4.1\\thesisGroup9\\GIT\\ThesisGroup9\\word2vec\\my_word2vec_model2.bin'  # trained word2vec,books vec theke create kora ase, she toiri korse gensim diye, tai amra gensim use korchi. model ta load korchi.
model = Word2Vec.load(filename)

vocab = list(model.wv.vocab)
X = model[vocab[100:130]]

S = vocab[100:130]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

df = pd.DataFrame(X_tsne, index= S, columns=['x', 'y'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'])
ax.set_ylabel('Word Compressed Distance in Y Direction', color='blue')
ax.set_xlabel('Word Compressed Distance in X Direction', color='blue')


for word, pos in df.iterrows():
    ax.annotate(word, pos)
    
plt.show()

#%%
import gensim
import numpy as np

filename = 'E:\\4.1\\thesisGroup9\\GIT\\ThesisGroup9\\word2vec\\my_word2vec_model2.bin'  # trained word2vec,books vec theke create kora ase, she toiri korse gensim diye, tai amra gensim use korchi. model ta load korchi.

model = Word2Vec.load(filename)
#%%

def display_closestwords(model, word):
    
    arr = np.empty((0,100), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0] 
    y_coords = Y[:, 1]
    # display scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.set_title('', color='blue')
    ax.set_ylabel('Word Compressed Distance in Y Direction', color='blue')
    ax.set_xlabel('Word Compressed Distance in X Direction', color='blue')
    plt.scatter(x_coords, y_coords)
    

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    
    plt.show()
 
display_closestwords(model, 'computer')
display_closestwords(model, 'window')
display_closestwords(model, 'month')

#%%
#similarity = model.similarity('computer', 'virtual')
#similarity = model.similarity('window', 'frame')
similarity = model.similarity('month', 'week')
print(similarity)

#cosine distance = 1 - cosine similarity
distance = 1 - similarity
print(distance)
