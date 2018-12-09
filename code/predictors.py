"""
Python 3.x
Author: Rami Aly, E-mail: `rami.aly@outlook.com`
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stop_words import get_stop_words
import string
punctuations = string.punctuation
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os
import re

import string
punctuations = string.punctuation

import spacy
parser = None
stopwords = None


def spacy_init(language):
    """
    Initilizes spacy and stop-words for respective language
    """
    global parser, stopwords
    print("Initialize Spacy")
    if language == "EN":
        parser = spacy.load('en_core_web_sm')
        stopwords = get_stop_words('en')
    print("Intialization Spacy finished")



def clean_str(string):
    """
    Cleans a string from very weird punctions and symbols
    TODO: replace by space better?
    """
    string = re.sub(r"[^A-Za-z0-9()!?\'\`äöüß ]", "", string)
    return string.strip().lower()


class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}


def clean_text(text):
    """
    Basic utility function to clean the text
    """
    return text.strip().lower()

def identity_tokenizer(text):
    return text


def spacy_tokenizer_basic(sentence):
    """
    Very basic preprocessing(tokenizing, removal of stopwords, too many whitespaces)
    """
    #print(sentence)
    tokens = parser(sentence)
    #tokens = [tok.lower_ for tok in tokens]
    tokens = [tok.text for tok in tokens]
    #print(tokens)
    tokens = [tok for tok in tokens if tok not in stopwords and " " not in tok]
    # tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    #print(tokens)
    return tokens



def spacy_tokenizer(sentence):
    """
    Create spacy tokenizer that parses a sentence and generates tokens
    uses lemma of word if not pronoun
    """
    tokens = parser(sentence)
    tokens = [tok.lemma_.lower().strip() if tok.pos_ != "PRON" and tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    #print(tokens)
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    return tokens


def vectorizerSpacy():
    #cs = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,2))
    #cs = TfidfVectorizer(tokenizer = spacy_tokenizer_basic, ngram_range=(1,1))
    cs = TfidfVectorizer(tokenizer = identity_tokenizer, ngram_range=(1,2), lowercase = False)
    return cs
