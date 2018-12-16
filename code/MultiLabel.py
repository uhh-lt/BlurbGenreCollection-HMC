#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Rami Aly, E-mail: `rami.aly@outlook.com`
"""

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
import string
punctuations = string.punctuation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomTreesEmbedding, BaggingClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.externals import joblib
import os
from predictors import predictors, vectorizerSpacy, spacy_init
from os.path import join
import json
import codecs
import argparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np
import io
import pickle
import string
punctuations = string.punctuation



METHODS_MULTI = ["LinearSVC"]
#Methods testedd on trial, LinearSVC was the best
#["LogisticRegressionL1", "LogisticRegressionL2", "LinearSVC", "Bagging","RandomForest", "MultinomialNB","AdaBoost"]

CV_NUM = 3



def get_root_classes(labels, language):
    """
    Returns most special classes of a blurb
    """
    all_relations = set([])
    relations_dict = {}
    REL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../crawler', language , 'hierarchy.txt')
    with io.open(REL_FILE, mode = 'r', encoding ='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            rel = line.split('\t')
            if len(rel) > 1:
                rel = (rel[0], rel[1][:-1])
            else:
                 rel = (rel[0][:-1], rel[0][:-1])
            all_relations.add(rel)
    root_labels = []
    for label in labels:
        root_label = []
        #print(label)
        for genre in label:
            if len(label) == 1:
                root_label = label
                break
            is_root = True
            children  = [child for (parent, child) in all_relations if parent == genre]
            for child in children:
                if child in label:
                    is_root = False
            if is_root:
                root_label.append(genre)
        #print(root_label)
        root_label = set(root_label)
        root_labels.append(root_label)
    return root_labels


def train_test_root(classifier, train, train_label, test, test_label, language):
    """
    Method for evaluation performance of SVM only on leaf classes
    train test pipeline
    """
    lb = MultiLabelBinarizer()
    vectorizer = vectorizerSpacy()
    clas = OneVsRestClassifier(classifier)
    pipe = Pipeline([
                     ('vectorizer', vectorizer),
                     ('classifier',clas)])

    train_label = get_root_classes(train_label, language)
    test_label = get_root_classes(test_label, language)

    y = lb.fit_transform(train_label)

    pipe.fit(train, y)

    pred_data = pipe.predict(test)

    test_labels = lb.transform(test_label)
    print(test_label[0:5])
    print(lb.inverse_transform(pred_data[0:5]))


    print("F1: " + str(f1_score(test_labels, pred_data, average='micro')))
    print("F1_macro: " + str(f1_score(test_labels, pred_data, average='macro')))
    print("Recall: " + str(recall_score(test_labels, pred_data, average='micro')))
    print("Recall_macro: " + str(recall_score(test_labels, pred_data, average='macro')))
    print("Precision: " + str(precision_score(test_labels, pred_data, average='micro')))
    print("Precision_macro: " + str(precision_score(test_labels, pred_data, average='macro')))
    print("Accuracy:" + str(accuracy_score(test_labels, pred_data)))


def train_test(classifier, train, train_label, test, test_label):
    """
    Method for evaluation performance of SVM
    train test pipeline
    """
    vectorizer = vectorizerSpacy()
    lb = MultiLabelBinarizer()
    clas = OneVsRestClassifier(classifier)
    pipe = Pipeline([
                     ('vectorizer', vectorizer),
                     ('classifier',clas)])
    y = lb.fit_transform(train_label)
    print(len(y))
    pipe.fit(train, y)
    pred_data = pipe.predict(test)

    print((str(precision_score(lb.transform(test_label), pred_data, average='micro')) + '\n' + str(recall_score(lb.transform(test_label), pred_data, average='micro')) + '\n' +
     str(f1_score(lb.transform(test_label), pred_data, average='macro')) + '\n' + str(f1_score(lb.transform(test_label), pred_data, average='micro')) +
      '\n'  +str(accuracy_score(lb.transform(test_label), pred_data))).replace(".", ","))

    print("F1: " + str(f1_score(lb.transform(test_label), pred_data, average='micro')))
    print("F1_macro: " + str(f1_score(lb.transform(test_label), pred_data, average='macro')))
    print("Recall: " + str(recall_score(lb.transform(test_label), pred_data, average='micro')))
    print("Precision: " + str(precision_score(lb.transform(test_label), pred_data, average='micro')))
    print("Accuracy:" + str(accuracy_score(lb.transform(test_label), pred_data)))



def crossval(classifier, data):
    X, y = ([x[0] for x in data],[x[1] for x in data])
    y = MultiLabelBinarizer().fit_transform(y)
    pipe = Pipeline([("cleaner", predictors()),
                     ('vectorizer', vectorizerSpacy()),
                     ('classifier', OneVsRestClassifier(classifier))])
    for scoring in ["precision_micro", "recall_micro", "f1_micro", "accuracy"]:
        s = cross_val_score(pipe, X, y, cv=CV_NUM, scoring=scoring, n_jobs=CV_NUM)
        print("%s: %.2f +- %.2f" % (scoring, s.mean(), s.std()))



def create_classifier(type):
    if type == "LogisticRegressionL2": clf = LogisticRegression(penalty='l2', tol=0.0001, C=1.0)  #, class_weight='auto')
    elif type == "LogisticRegressionL1": clf = LogisticRegression(penalty='l1', tol=0.0001, C=1.0)  #, class_weight='auto')
    elif type == "MultinomialNB": clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    elif type == 'LinearSVC': clf = LinearSVC(C=0.5)
    elif type == 'SVC':  clf = SVC(C=1, probability=True)
    elif type == 'RandomForest': clf = RandomForestClassifier(n_estimators = 100)
    elif type == 'AdaBoost': clf = AdaBoostClassifier()
    elif type == 'RandomTreesEmbedding': clf = RandomTreesEmbedding()
    elif type == 'Bagging': clf = BaggingClassifier()
    elif type == "SGD": clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter= 5, random_state=42)
    else: clf = LogisticRegression(penalty='l2', tol=0.0001, C=1.0)  # , class_weight='auto')
    return clf


def run(mode, lang, level, dev):
    filename = os.path.join("..","resources",lang + "_" + "spacy_pruned")
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
        if dev:
            X_train = data['X_train']
            X_dev =  data['X_dev']
            X_train = [el for el in X_train]
            y_train = data['y_train']
            y_dev = data['y_dev']
        else:
            X_train = data['X_train'] + data['X_dev']
            X_train = [el for el in X_train]
            y_train = data['y_train'] + data['y_dev']
        print(X_train[:2])
        X_test = data['X_test']
        X_test = [el for el in X_test]
        y_test = data['y_test']

        print("Finished loading input.")

    classifier_path = os.path.join(os.path.dirname(__file__), '..' + 'checkpoints', 'baseline')
    spacy_init(lang)
    global METHODS_MULTI
    for method in METHODS_MULTI:
        print(classifier_path)
        sl = create_classifier(method)
        if mode == 'train_test':
            if dev:
                train_test(sl,X_train, y_train, X_dev, y_dev)
            else:
                train_test(sl, X_train, y_train, X_test, y_test)
        elif mode == 'cv':
            print(sl)
            crossval(sl, train)
        if mode == 'root':
            train_test_root(sl, X_train,y_train, X_test, y_test, language = lang)

def main():
    global args
    parser = argparse.ArgumentParser(description="MultiLabel classification of blurbs")
    parser.add_argument('--mode', type=str, default='train', choices=['train_test','cv', 'gridsearch', 'root'], help="Mode of the system.")
    parser.add_argument('--lang', type=str, default='train', choices=['EN','DE', 'WOS', 'TRIAL', 'COMPQ'], help=" Dataset to use")
    parser.add_argument('--level', type=int, choices=[0,1,2,3], help= "Level of hierarchy")
    parser.add_argument('--dev', type=bool, default= False, help=" Dataset use validation")



    args = parser.parse_args()
    print("Mode: ", args.mode)
    run(args.mode, args.lang, args.level, args.dev)



if __name__ == '__main__':
    main()
