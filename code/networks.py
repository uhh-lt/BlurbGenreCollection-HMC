import gensim
from keras.constraints import max_norm
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.optimizers import Adam, RMSprop, Adagrad
from keras.models import Model
import keras.losses
from keras.layers import Embedding
from loader import read_all_genres
from keras import layers, models
import numpy as np
import os
import math
from data_helpers import ml
from capsulelayers import CapsuleLayer, PrimaryCap, Length
from keras import backend as K
import sys
import pickle


def create_model_capsule(preload, embedding_dim, sequence_length, num_filters,
 language, num_classes, use_static, init_layer, vocabulary, learning_rate,  dense_capsule_dim, n_channels, routings = 3):
     """
     Implementation of capsule network
     """
    over_time_conv = 100#50
    #capsule_dim = 16
    inputs = Input(shape=(sequence_length,), dtype='int32')

    embedding = pre_embedding(embedding_dim = embedding_dim, seq_length = sequence_length,
        input = inputs, use_static = use_static, voc = vocabulary, lang = language)

    if language == 'DE':
        primarycaps = PrimaryCap(embedding, dim_capsule=8, n_channels= 45, kernel_size=over_time_conv,
        strides=1, padding='valid', name = 'primarycaps')
    elif language == 'EN':
        primarycaps = PrimaryCap(embedding, dim_capsule=8, n_channels= 55, kernel_size=over_time_conv,
        strides=1, padding='valid', name = 'primarycaps')
    else:
        primarycaps = PrimaryCap(embedding, dim_capsule=8, n_channels= n_channels, kernel_size=over_time_conv,
        strides=1, padding='valid', name = 'primarycaps')

    dense = CapsuleLayer(num_capsule=num_classes, dim_capsule=dense_capsule_dim, routings=routings,
                             name='digitcaps')(primarycaps)

    out_caps = Length(name='capsnet')(dense)
    model = Model(inputs=inputs, outputs=out_caps)

    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=[margin_loss],
                  metrics=['categorical_accuracy'])

    #initilizes the weight of transformation matrix W
    if init_layer:
        weights = model.layers[-2].get_weights()[0]
        co_occurences = co_occurence_weights(weights[0].shape[1], num_classes, language)
        print(len(co_occurences), len(co_occurences[0]))
        for i, co_occurence in enumerate(co_occurences):
            if i >= weights.shape[1]:
                break
            for j, weight in enumerate(co_occurence):
                #initilzes the  weights between dim of primary and one complete  dense capsule
                weights[j][i][0] = weights[j][i][0] if weight != 0 else 0
                #weights[j][i][0][0] = weight if weight != 0 else weights[j][i][0][0]
        model.layers[-2].set_weights([weights])
    print(model.summary())
    return model


def margin_loss(y_true, y_pred):
    """
    Margin loss as described in Sabour et al. (2017)
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def co_occurence_weights(num_units, num_classes, language):
    """
     loads the co-occurence matrix with respective weights
    """
    parent_child = []
    w = math.sqrt(6) / math.sqrt(num_units + num_classes)
    _, occurences = read_all_genres(language)

    for occurence in occurences:
        if occurence[0].issubset(set(ml.classes_)):
            frequency = occurence[1]
            w_f = w # * math.sqrt(frequency)
            binary_rel = ml.transform([occurence[0]])
            parent_child.append([w_f if x==1 else 0 for x in binary_rel[0]])
    print(len(occurences))
    return parent_child



def create_model_cnn(preload, embedding_dim, sequence_length, num_filters,
 language, num_classes, use_static, init_layer, vocabulary, learning_rate):
     """
     Implementation of Kims et al. CNN,
     """
    filter_sizes = [3,4,5]
    drop = 0.5
    embedding = None
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding_1d = pre_embedding(embedding_dim = embedding_dim, seq_length = sequence_length,
        input = inputs, use_static = use_static, voc = vocabulary, lang = language)
    embedding = Reshape((sequence_length, embedding_dim, 1))(embedding_1d)
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim),
     padding='valid', kernel_initializer='normal', activation='relu',
      kernel_constraint=max_norm(3))(embedding)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim),
     padding='valid', kernel_initializer='normal', activation='relu',
      kernel_constraint=max_norm(3))(embedding)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim),
     padding='valid', kernel_initializer='normal', activation='relu',
      kernel_constraint=max_norm(3))(embedding)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1,1),
     strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1),
     strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1),
     strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)

    output2 = Dense(units = num_classes, activation = 'sigmoid')(dropout)

    model = Model(inputs=inputs, outputs=output2)
    adam = Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['categorical_accuracy'])
    print(model.summary())
    #Initlizes weight of dense layer
    if(init_layer):
        weights, bias = model.layers[-1].get_weights()
        parent_child = co_occurence_weights(weights.shape[0], num_classes, language)
        for i, entry in enumerate(parent_child):
            if i >= weights.shape[0]:
                break
            weights[i] = [0 if entry[j] == 0 else entry[j] for j in range(len(weights[i]))]
        model.layers[-1].set_weights([weights, bias])
    return model




def create_model_lstm(preload, embedding_dim, sequence_length, num_units,
 language, num_classes, use_static, init_layer, vocabulary, learning_rate):
    """
    Implementation of simple LSTM with recurrent dropout
    """

    model = Sequential()
    pre_embedding(model = model, embedding_dim = embedding_dim,
     seq_length = sequence_length, use_static = use_static,
      voc = vocabulary, lang = language)
    lstm_out = num_units
    model.add(LSTM(lstm_out, recurrent_dropout = 0.5))
    model.add(Dense(units=num_classes, activation='sigmoid'))
    if language == 'EN':
        optimizer = Adam(lr = learning_rate)
    else:
        optimizer = RMSprop(lr = learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    print(model.summary())
    #initilzes weight of dense layer
    if(init_layer):
        weights, bias = model.layers[-1].get_weights()
        parent_child = co_occurence_weights(weights.shape[0], num_classes, language)
        print(len(parent_child))
        for i, entry in enumerate(parent_child):
            if i >= weights.shape[0]:
                break
            weights[i] = [0 if entry[j] == 0 else entry[j] for j in range(len(weights[i]))]
        model.layers[-1].set_weights([weights, bias])
    return model



def pre_embedding(embedding_dim, seq_length, use_static, voc, lang, input = None, model = None):
    """
    Loads mebedding for model
    """
    embed_saved_path =  os.path.join(os.path.dirname(__file__), '../resources',
    'embed_' + str(lang) + '_' + str(seq_length) + '_' + "dev")
    if os.path.exists(embed_saved_path):
        print("Loading Embedding Matrix...")
        embed_saved_file = open(embed_saved_path, 'rb')
        embedding_matrix = pickle.load(embed_saved_file)
    else:
        if lang == 'DE':
            w2v_german_dir = os.path.join(os.path.dirname(__file__), '../resources', 'wiki.de.vec')
            w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_german_dir,
              binary=False, unicode_errors='ignore')
        else:
            w2v = {}
            w2v_english_dir = os.path.join(os.path.dirname(__file__), '../resources',
             'wiki.en.vec')
            w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_english_dir, binary=False)

        print("Embedding Voc Size", len(w2v.wv.vocab))
        print("The unseen string -EMPTY- is not in the embedding: ", "-EMPTY-" not in w2v.wv.vocab)
        count = 0
        embedding_matrix = np.random.uniform(-0.25, 0.25, (len(voc) + 1, embedding_dim))

        for word, i in voc.items():

            if word not in  w2v.wv.vocab:
                continue
            embedding_vector = w2v.wv.get_vector(word)

            if embedding_vector is not None:
                count+=1
                embedding_matrix[i] = embedding_vector

        print("Found: ", count, " words")
        print("Out of", len(voc.items()), "Words in dataset")
        embed_saved_file = open(embed_saved_path, 'wb')
        pickle.dump(embedding_matrix, embed_saved_file)

    trainable = not use_static
    if input != None:
        embedding =  Embedding(len(voc) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=seq_length,
                                trainable= trainable)(input)
        return embedding
    elif model != None:
        model.add(Embedding(len(voc) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=seq_length,
                                trainable= trainable))
