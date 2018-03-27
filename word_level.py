import numpy as np
import time
import math
from pickle import dump
from nltk.corpus import gutenberg
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

corpus = list(gutenberg.sents(gutenberg.fileids()[0:3]))
for i in range(len(corpus)):
    corpus[i] = list(map(lambda x: x.lower(), corpus[i]))


def remove_punctuation(corpus):
    cleaned_corpus = []

    punctuations = ['!', '(', ')', '-', '[', ']', '{', ';', ':', "'",
                    '\\', '<', '>', '.', '/', '?', '~', '&', "''", ',', '--', '``', '"']
    for sent in corpus:
        sent1 = []
        for word in sent:
            if word not in punctuations:
                sent1.append(word)
        cleaned_corpus.append(sent1)
    return cleaned_corpus


corpus = remove_punctuation(corpus)
c = list()
for sentence in corpus:
    for words in sentence:
        c.append(words)
corpus = c
length = 8 + 1
sequence = list()
for i in range(length, len(corpus)):
    seq = corpus[i - length:i]
    sequence.append(seq)
print("Total Sequences", len(sequence))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequence)
sequences = tokenizer.texts_to_sequences(sequence)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size :', vocab_size)
sequences = np.array(sequences)
x, y = sequences[:, :-1], sequences[:, -1]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.1)
seq_length = xtrain.shape[1]
model = Sequential()
model.add(Embedding(vocab_size, 20, input_length=seq_length))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
start_time = time.time()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(xtrain, ytrain, epochs=100, verbose=1)
print("--- %s seconds ---" % (time.time() - start_time))

model_json = model.to_json()
with open("model_final.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_final.h5")
print("Saved model to disk")
dump(tokenizer, open('tokenizer_final.pkl', 'wb'))
print("Tokenizer dumped")

loss,_=model.evaluate(xtest,ytest,verbose=1)
print("perperxility is:",math.exp(loss))
