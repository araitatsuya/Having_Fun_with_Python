#########################
#
# fast.ai Part 1. Lesson ~6
# 
#   Using a double-stack LSTM model trained with Nietzsche text corpus
#   to predict a next charactor for a given sequence of charactors (i.e. a part of sentence). 
#
#   http://wiki.fast.ai/index.php/Lesson_6_Timeline
#   https://www.youtube.com/watch?v=ll9y1U0SoVY&t=76m50s
#   
#   Q:"In the double LSTM model, what is the input to the second LSTM model
#   in addition to the out put of the first LSTM?
#
#   A:"Previous output of its own hidden state.
#

### All Modules
from __future__ import division, print_function

import numpy as np
from numpy.random import choice
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, TimeDistributed, Activation
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam

## You might needed these to save weights
import h5py 
import cython

########################
#
# Preprocessing
#
########################

### Get Nietzsche Text File. 
path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

#chars : sorted list of charactors used in the text. 
chars = sorted(list(set(text)))
vocab_size = len(chars)+1
print('total chars:', vocab_size)
chars.insert(0, "\0")

#string of charactors: The last 6 charactors are gibberish
''.join(chars[1:-6])

#charactor -> id, id -> charactor in dict
#char_indices = {c:i for i,c in enumerate(chars)} #I would write this. 
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Assigning a numerical id on each charactor in the text.
# idx is a list. 
idx = [char_indices[c] for c in text]

print('corpus length:', len(idx))
# First 10 charactor-indices
idx[:10]

# First 70 charactor-IDs -> ID_Charactor_dict -> First 70 charactors. 
''.join(indices_char[i] for i in idx[:70])


maxlen = 40
sentences = []
next_chars = []
for i in range(0, len(idx) - maxlen+1):
    sentences.append(idx[i: i + maxlen])
    next_chars.append(idx[i+1: i+maxlen+1])

# sentences: 600862 x 40 list
# next_chars: 600862 x 40 list
# sentences[0][:10]
    
print('nb sequences:', len(sentences))

# list -> np.array
sentences = np.concatenate([[np.array(o)] for o in sentences[:-2]])
next_chars = np.concatenate([[np.array(o)] for o in next_chars[:-2]])

# shape: 600860 x 40
sentences.shape, next_chars.shape

########################
#
# Keras Model / Training
#
########################

n_fac = 24
model=Sequential([
        # Embedding(60, 24, input_length=40),
        Embedding(vocab_size, n_fac, input_length=maxlen),
        # dropout_W: Fraction of the input units to drop for input gates.
        # dropout_U: Fraction of the input units to drop for recurrent connections.
        LSTM(512, input_dim=n_fac, return_sequences=True, dropout_U=0.2, dropout_W=0.2,
             consume_less='gpu'),
        Dropout(0.2),
        LSTM(512, return_sequences=True, dropout_U=0.2, dropout_W=0.2,
             consume_less='gpu'),
        Dropout(0.2),
        # https://keras.io/layers/core/
        # Dense(x)
        # Just your regular densely-connected NN layer.
        # e.g. model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        #
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        # e.g. model.add(Dense(32))
        #
        # https://keras.io/layers/wrappers/
        # TimeDistributed(Dense(x))
        # This wrapper applies a layer to every temporal slice of an input.
        # It takes a batch of x damples.
        #
        # Consider a batch of 32 samples, where each sample is a sequence of 10 vectors of 16 dimensions.
        # The batch input shape of the layer is then (32, 10, 16), and the input_shape,
        # not including the samples dimension, is (10, 16).
        # You can then use TimeDistributed to apply a Dense layer to each of the 10 timesteps, independently:
        # e.g. TimeDistributed(Dense(8), input_shape=(10, 16))
        # -> model.output_shape == (None, 10, 8)
        # The output will then have shape (32, 10, 8).
        # 
        # In subsequent layers, there is no need for the input_shape:
        # e.g. model.add(TimeDistributed(Dense(32)))
        # -> model.output_shape == (None, 10, 32)
        # The output will then have shape (32, 10, 32).
        #
        # vocab_size = 60
        TimeDistributed(Dense(vocab_size)),
        Activation('softmax')
    ])

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())
model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 40, 24)            1440
_________________________________________________________________
lstm_1 (LSTM)                (None, 40, 512)           1099776
_________________________________________________________________
dropout_1 (Dropout)          (None, 40, 512)           0
_________________________________________________________________
lstm_2 (LSTM)                (None, 40, 512)           2099200
_________________________________________________________________
dropout_2 (Dropout)          (None, 40, 512)           0
_________________________________________________________________
time_distributed_1 (TimeDist (None, 40, 60)            30780
_________________________________________________________________
activation_1 (Activation)    (None, 40, 60)            0
=================================================================
Total params: 3,231,196
Trainable params: 3,231,196
Non-trainable params: 0
_________________________________________________________________
'''

#####################
#
# Model Fit
#
#####################

# Input: sentences.shape (600860, 40)
# Output: next_chars.shape (600860, 40)
# Output: np.expand_dims(next_chars,-1).shape (600860, 40, 1)
print('*------1-------*')
model.fit(sentences, np.expand_dims(next_chars,-1), batch_size=64, nb_epoch=1)
# Save weights and load weights
model.save_weights('char_rnn.h5') 

#####################
#
# Results
#
#####################

def print_example():
    # seed_string="ethics is a basic foundation of all that" # This is from the fast.ai class.
    seed_string = "i am always wondering why this kind of t" # Make sure There are 40 charactors. 
    for i in range(320):
        x=np.array([char_indices[c] for c in seed_string[-40:]])[np.newaxis,:]
        preds = model.predict(x, verbose=0)[0][-1]
        preds = preds/np.sum(preds)
        next_char = choice(chars, p=preds)
        seed_string = seed_string + next_char
    print(seed_string)

model.load_weights('char_rnn.h5')
print_example()
'''
... After many epochs

i am always wondering why this kind of testing with
his maching taste. finally we then inverts the savage of human soul has required
 too judgments the
charm of an ordinary literature and elversable power, her divine way had been be
stowed through them it against gifts him
to stir acts, and in the scientific demonstration of an
act as invanity as the pour fear
'''

### What is going on.
seed_string = "i am always wondering why this kind of t"
x=np.array([char_indices[c] for c in seed_string[-40:]])
x.shape # (40,)
x=np.array([char_indices[c] for c in seed_string[-40:]])[np.newaxis,:]
x.shape # (1,40)
preds = model.predict(x, verbose=0)
preds.shape # (1,40,60)
preds[0][-1].shape #(60,) What you want is the last charactor of preds. 
preds = preds[0][-1]
preds = preds/np.sum(preds)
max(preds) # 0.29986468 -> This corresponds to char_id of 35 -> "h"
next_char = choice(chars, p=preds) # "h"
seed_string = seed_string + next_char
# seed_string = "i am always wondering why this kind of t" + "h"
# seed_string[-40:] = " am always wondering why this kind of th"
# Repeat !
