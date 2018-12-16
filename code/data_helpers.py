"""
Author: Rami Aly, E-mail: `rami.aly@outlook.com`
"""
import numpy as np
np.set_printoptions(threshold=np.nan)
import re
import itertools
from collections import Counter
import io
from loader import Blurb_Loader #load_data_multiLabel, read_relations, load_outlier
from comp_questions_loader import CompQ_Loader
from sklearn.preprocessing import MultiLabelBinarizer
from predictors import clean_text, spacy_tokenizer,spacy_init, clean_str, spacy_tokenizer_basic
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import pickle
import os
ml = MultiLabelBinarizer()
data_loader = None

UNSEEN_STRING = "-EMPTY-"


def get_level_genre(relations, genre):
    """
    return hierarchy level of genre
    """
    height = 0
    curr_genre = genre
    last_genre = None
    while curr_genre != last_genre:
        last_genre = curr_genre
        for relation in relations:
            if relation[1] == curr_genre:
                height+=1
                curr_genre = relation[0]
                break
    return height, curr_genre


def extract_hierarchies(language):
    """
    Returns dictionary with level and respective genres
    """
    hierarchies_inv = {}
    relations, singletons = data_loader.read_relations()
    genres = set([relation[0] for relation in relations] +
    [relation[1] for relation in relations]) | singletons
    #genres, _= read_all_genres(language, max_h)
    for genre in genres:
        if not genre in hierarchies_inv:
            hierarchies_inv[genre] = 0
    for genre in genres:
        # genre_t = ml.transform([[genre]])
        # print(genre_t)
        hierarchies_inv[genre], _ = get_level_genre(relations, genre)
    hierarchies = {}
    for key,value in hierarchies_inv.items():
        if not value in hierarchies:
            hierarchies[value] = [key]
        else:
            hierarchies[value].append(key)
    return [hierarchies,hierarchies_inv]


def remove_genres_not_level(language, labels, outputs, level, exact_level):
    """
    Removes genres from output that are not on the respective level of the hierarchy
    """
    hierarchies, _ = extract_hierarchies(language)
    all_genres = set([])
    if exact_level:
        all_genres = all_genres | set(hierarchies[level])
        print("Labels on current hierarchy", all_genres)
    else:
        for i in range(level + 1):
            all_genres = all_genres | set(hierarchies[i])
    labels_string = ml.inverse_transform(labels)
    outputs_string = ml.inverse_transform(outputs)
    labels_pruned = []
    outputs_pruned = []
    for label in labels_string:
        label_c = list(label).copy()
        for genre in label:
            if genre not in all_genres:
                label_c.remove(genre)
        labels_pruned.append(label_c)
    for label in outputs_string:
        label_c = list(label).copy()
        for genre in label:
            if genre not in all_genres:
                #print(genre)
                label_c.remove(genre)
        outputs_pruned.append(label_c)
    outputs_pruned = ml.transform(outputs_pruned)
    labels_pruned = ml.transform(labels_pruned)
    return [labels_pruned, outputs_pruned]



def load_data_and_labels(spacy, lowfreq, dataset, level, dev = False):
    """
    preprocessing of blurbs and labels of dataset
    """
    global ml
    print(dataset)
    path = filename = os.path.join("..","resources")
    filename = os.path.join(path, dataset + "_" + "spacy_pruned")
    print(filename)

    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(filename):
        if spacy:
            spacy_init(language = 'EN')

        fp = open(filename, 'wb')
        data = {}
        train, dev, test = data_loader.load_data_multiLabel()

        X_train, y_train = ([x[0] for x in train], [x[1] for x in train])
        X_test, y_test = ([x[0] for x in test], [x[1] for x in test])
        X_dev, y_dev = ([x[0] for x in dev], [x[1] for x in dev])

        X_train = atomic_load_data(spacy, lowfreq, X_train)
        X_test = atomic_load_data(spacy, lowfreq, X_test)
        X_dev = atomic_load_data(spacy, lowfreq, X_dev)
        # y_train = [set(np.asarray(label).ravel()) for label in y_train]
        # y_dev = [set(np.asarray(label).ravel()) for label in y_dev]
        # y_test = [set(np.asarray(label).ravel()) for label in y_test]

        data['X_train'] = X_train
        data['y_train'] = y_train
        data['X_test'] = X_test
        data['y_test'] = y_test
        data['X_dev'] = X_dev
        data['y_dev'] = y_dev
        pickle.dump(data, fp)

    else:
        print("Loading preprocessed input...")
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
            if dev:
                X_dev = data['X_dev']
                y_dev = data['y_dev']
            else:
                X_train = data['X_train'] + data['X_dev']
                y_train = data['y_train'] + data['y_dev']

            print("Finished loading input.")
    print("Lenght Train data", len(X_train))
    print("Length Test data", len(X_test))
    print("Example entry: ", X_train[0])
    y_train = ml.fit_transform(y_train)
    y_test = [[x for x in sample if x in ml.classes_] for sample in y_test]
    y_test = ml.transform(y_test)
    if dev:
        y_dev = [[x for x in sample if x in ml.classes_] for sample in y_dev]
        y_dev = ml.transform(y_dev)
        values = [X_train, X_dev, X_test, y_train, y_dev, y_test]
        return values
    else:
        return [X_train, X_test, y_train, y_test]


def atomic_load_data(spacy, lowfreq, x_text):
    """
    cleans blurb, applies spacy and removes low frequency words
    """
    print("Applying Preprocessing....")
    if spacy:
        x_text = [clean_str(x) for x in x_text]
        x_text = [str(x) for x in x_text]
        x_text = [spacy_tokenizer_basic(x) for x in x_text]

    else:
        #no string cleaning for russian
        #x_text = [clean_str(sent) for sent in x_text]
        x_text = [s.split(" ") for s in x_text]
    if lowfreq:
        MIN_FRE = 2
        freq = {}
        for blurb in x_text:
            for word in blurb:
                if word not in freq:
                    freq[word] = 1
                else:
                    freq[word]+=1

        x_text_n = []
        for blurb in x_text:
            sentence = blurb.copy()
            for i,word in enumerate(blurb):
                if freq[word] < MIN_FRE:
                    sentence[i] = UNSEEN_STRING
            x_text_n.append(sentence)
        x_text = x_text_n
    return x_text


def get_parents(child, relations):
    """
    Get the parent of a genre
    """
    parents = []
    current_parent = child
    last_parent = None
    while current_parent != last_parent:
        last_parent = current_parent
        for relation in relations:
            if relation[1] == current_parent:
                current_parent = relation[0]
                parents.append(current_parent)

    return parents

def get_genre_level(genre, hierarchy):
    for key, value in hierarchy.items():
        if genre in value:
            return key


def adjust_hierarchy_threshold(output, output_b, language, max_h = 1, threshold = 0.4):
    """
    Adjusts output of network by applying correction method by threshold
    """
    print("Adjusting Hierarchy")
    print(len(output[0]))
    relations,_ = data_loader.read_relations()
    hierarchy, _ = extract_hierarchies(language)
    new_output = []
    outputs = ml.inverse_transform(output_b)
    for i,output_el in enumerate(outputs):
        labels = set(list(output_el))
        if len(labels) > 1:
            labels_cp = labels.copy()
            labels_hierarchy = {}
            for level in hierarchy:
                for label in labels:
                    if label in hierarchy[level]:
                        if level in labels_hierarchy:
                            labels_hierarchy[level].add(label)
                        else:
                            labels_hierarchy[level] = set([label])
            for level in labels_hierarchy:
                if level > 0:
                    for label in labels_hierarchy[level]:
                        all_parents = get_parents(label, relations)
                        average_confidence = 0.
                        for element in all_parents:
                            element_index = list(ml.classes_).index(element)
                            confidence = output[i][element_index]
                            average_confidence += confidence
                        average_confidence = average_confidence / len(all_parents)
                        if average_confidence > threshold:
                            labels = labels | set(all_parents)
                        else:
                            labels.remove(label)
        new_output.append(tuple(list(labels)))
        #print("Corrected", labels)
    return ml.transform(new_output)



def adjust_hierarchy(output_b, language, max_h = 1, mode = 'semi_transitive'):
    """
    Correction of nn predictions by either restrictive or transitive method
    """
    global ml
    print("Adjusting Hierarchy")
    relations,_ = data_loader.read_relations()
    hierarchy, _ = extract_hierarchies(language)
    new_output = []
    outputs = ml.inverse_transform(output_b)
    for output in outputs:
        labels = set(list(output))
        if len(labels) > 1:
            labels_cp = labels.copy()
            labels_hierarchy = {}
            for level in hierarchy:
                for label in labels:
                    if label in hierarchy[level]:
                        if level in labels_hierarchy:
                            labels_hierarchy[level].add(label)
                        else:
                            labels_hierarchy[level] = set([label])
            for level in labels_hierarchy:
                if level > 0:
                    for label in labels_hierarchy[level]:
                        all_parents = get_parents(label, relations)
                        missing = [parent for parent in all_parents if not parent in labels]
                        no_root = True
                        for element in missing:
                            if element in labels and get_genre_level(element, hierarchy) == 0:
                                labels = labels | all_parents
                                no_root = False

                        if len(missing) >= 1:
                            if mode == "restrictive":
                                    labels.remove(label)
                            elif mode == "semi_transitive":
                                if no_root:
                                    labels.remove(label)
                                else:
                                    labels = labels | set(all_parents)
                            elif mode == "transitive":
                                labels = labels | set(all_parents)
        new_output.append(tuple(list(labels)))
    return ml.transform(new_output)

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index    print ml.classes_
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    #print vocabulary_inv
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]



def load_data(spacy=False, lowfreq= True, max_sequence_length = 200, type = 'EN', level = 1, dev = False, outlier = False):
    """
    Pipeline for loading the feature sets for all data splits
    Also applies padding and replacement of unknown words
    """
    global data_loader
    if type == 'EN':
        data_loader = Blurb_Loader()
    elif type =='COMPQ':
        data_loader = CompQ_Loader()


    if dev:
        data  = load_data_and_labels(spacy, lowfreq, type, level, dev)
        X_train, X_dev, X_test, y_train, y_dev, y_test = data
    else:
        X_train, X_test, y_train, y_test = load_data_and_labels(spacy, lowfreq, type, level, dev)


    sentences_padded_train = pad_sequences(X_train, maxlen=max_sequence_length, dtype='str', padding = 'post', truncating ='post')
    vocabulary_train, vocabulary_inv_train = build_vocab(sentences_padded_train)
    x, y = build_input_data(sentences_padded_train, y_train, vocabulary_train)

    if dev:
        X_dev = [[a if a in vocabulary_train else UNSEEN_STRING for a in sentence] for sentence in X_dev]
        sentences_padded_dev = pad_sequences(X_dev, maxlen=max_sequence_length, dtype='str', padding = 'post', truncating ='post')
        x_dev, y_dev = build_input_data(sentences_padded_dev, y_dev, vocabulary_train)

    if outlier:
        X_outlier, y_outlier = load_outlier_and_labels(type)
        X_outlier = [[a if a in vocabulary_train else UNSEEN_STRING for a in sentence] for sentence in X_outlier]
        sentences_padded_outlier = pad_sequences(X_outlier, maxlen=max_sequence_length, dtype='str', padding = 'post', truncating ='post')
        x_outlier, y_outlier = build_input_data(sentences_padded_outlier, y_outlier, vocabulary_train)


    X_test = [[a if a in vocabulary_train else UNSEEN_STRING for a in sentence] for sentence in X_test]
    sentences_padded_test = pad_sequences(X_test, maxlen=max_sequence_length, dtype='str', padding = 'post', truncating ='post')
    x_test, y_test = build_input_data(sentences_padded_test, y_test, vocabulary_train)

    print("Training samples:" + str(x.shape) + ", Test samples:" + str(x_test.shape) + " Number of Genres:" + str(len(y[0])))
    if dev:
        return [x, y, x_dev, y_dev, x_test, y_test, vocabulary_train, vocabulary_inv_train]
    if outlier:
        return [x, y, x_test, y_test, vocabulary_train, vocabulary_inv_train, x_outlier, y_outlier]
    else:
        return [x, y, x_test, y_test, vocabulary_train, vocabulary_inv_train]
