import numpy as np
import math
from sklearn.model_selection import train_test_split
from keras.callbacks import LambdaCallback
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop


def on_epoch_end(epoch, logs):
    model_json = model.to_json()
    with open("model{}.json".format(epoch), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("assignment2_weights_{}.h5".format(epoch))


if __name__ == '__main__':
    with open("text.txt", encoding='utf-8') as f:
        text = f.read().lower()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    maxlen = 8
    step = 2
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    print('Calculating X and Y')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.1)

    print('Training model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    model.fit(xtrain, ytrain,batch_size=128,epochs=100,callbacks=[print_callback])

    loss=model.evaluate(xtest,ytest,batch_size=128, verbose=1)
    print("perperxility is:",math.exp(loss))