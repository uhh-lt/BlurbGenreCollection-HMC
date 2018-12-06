#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import string
import os
import sys
import os.path
from os.path import join
from random import randint
from shutil import copyfile
import operator
from random import shuffle
import pickle
import re
#Maybe use yield, better style
#DIRECTORY = '/home/rami/Documents/Bachelorarbeit/resources'

DEST_DIRECTORY_EN = join(os.path.dirname(os.path.abspath(__file__)), '../resources', 'EN_pruned')

DEST_DIRECTORY_DE = join(os.path.dirname(os.path.abspath(__file__)), '../resources', 'DE_pruned')

WOS_DIRECTORY = join(os.path.dirname(os.path.abspath(__file__)), '../resources/WOS46985')



def load_data_multiLabel(language = 'EN', max_h = 1, dev = False):
    """
    Loads Multilabel data with max_h hierarchy level, so 1 includes all labels that have hat max height one in label tree
    @param dev: specifies whether the dev set should be used or not
    """
    if language == 'EN':
        dest_directory = DEST_DIRECTORY_EN
    elif language == 'DE':
        dest_directory = DEST_DIRECTORY_DE

    return (multi_label_atomic(join(dest_directory, 'train'), max_h),
     multi_label_atomic(join(dest_directory, 'dev'), max_h),
      multi_label_atomic(join(dest_directory, 'test'), max_h))



def load_rcv1(dev = False):
    from sklearn.datasets import fetch_rcv1
    rcv1_train = fetch_rcv1(subset='train', shuffle = True, random_state = 42)
    rcv1_train = list(zip(rcv1_train.data, [label for label in rcv1_train.target]))
    print(rcv1_train[0])
    sys.exit()


def text_cleaner(text):
    text = text.replace(".", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", "")
    text = text.replace("=", "")
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    return text.lower()

def load_WOS(dev = False):
    WOS_text = join(WOS_DIRECTORY, "X.txt")
    WOSL1= join(WOS_DIRECTORY, "YL1.txt")
    #Loads only the most specfic label tag
    WOSL2 = join(WOS_DIRECTORY, "Y.txt")
    with open(WOS_text) as f:
        content = f.readlines()
        content = [text_cleaner(x) for x in content]
    with open(WOSL1) as fk:
        contentk = fk.readlines()
    contentk = [x.strip() for x in contentk]
    with open(WOSL2) as fk:
        contentL2 = fk.readlines()
        contentL2 = [x.strip() for x in contentL2]
    #since WOSL1 does only label in respect to parents we have to relabel so that parent and child labels are unique
    parent_id_map = {"0": "134", "1":"135", "2":"136", "3": "137", "4":"138", "5":"139", "6":"140"}
    contentk = [parent_id_map[id] for id in contentk]
    return [content, contentk, contentL2]



def newsgroup():
    from sklearn.datasets import fetch_20newsgroups
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'), shuffle = True, random_state = 42)
    newsgroups_train = list(zip(newsgroups_train.data, [[label] for label in newsgroups_train.target]))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers','footers','quotes'), shuffle = True, random_state = 42)
    newsgroups_test = list(zip(newsgroups_test.data, [[label] for label in newsgroups_test.target]))
    return [newsgroups_train, newsgroups_test]


def read_relations(language = 'EN'):
    """
    Loads hierarchy file and returns set of relations
    """
    relations = set([])
    singeltons = set([])
    REL_FILE =  join(os.path.dirname(os.path.abspath(__file__)), '../crawler', language , 'hierarchy.txt')
    with open(REL_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:
            rel = line.split('\t')
            if len(rel) > 1:
                rel = (rel[0], rel[1][:-1])
            else:
                singeltons.add(rel[0][:-1])
                continue
            relations.add(rel)
    return [relations, singeltons]


def load_outlier(lang):
    """
    Loads low-frequency dataset
    """
    if lang == 'DE':
        outlier_directory = join(os.path.dirname(os.path.abspath(__file__)), '../resources', 'DE_outlier')
    elif lang == 'EN':
        outlier_directory = join(os.path.dirname(os.path.abspath(__file__)), '../resources', 'EN_outlier')
    return multi_label_atomic(outlier_directory)


def read_all_genres(language = 'EN'):
    """
    Loads list of label-cooccurences with frequency, sorted in descending order
    """
    occurences = []
    frequency = []
    hierarchy = set([])
    co_occurences_path =  os.path.join(os.path.dirname(__file__),
     '../resources', language + '_co_occurences')
    if os.path.exists(co_occurences_path):
        co_occurences_file = open(co_occurences_path, 'rb')
        occurences, frequency = pickle.load(co_occurences_file)
    else:
        if language == 'EN':
            dest_directory = DEST_DIRECTORY_EN
        elif language =='DE':
            dest_directory = DEST_DIRECTORY_DE
        for split in ['train', 'dev']:
            for filename in os.listdir(join(dest_directory, split)):
                soup = BeautifulSoup(open(join(dest_directory, split,  filename), 'rt').read(), "html.parser")
                for t in soup.findAll('topics'):
                    s1 = BeautifulSoup(str(t), "html.parser")
                    structure = ['d0', 'd1', 'd2', 'd3']
                    genres = set([])
                    for i in range(0, len(structure)):
                        for t1 in s1.findAll(structure[i]):
                            hierarchy.add(str(t1.string))
                            genres.add(str(t1.string))
                    if genres in occurences:
                        frequency[occurences.index(genres)] +=1
                    else:
                        occurences.append(genres)
                        frequency.append(1)
        co_occurence_file = open(co_occurences_path, 'wb')
        pickle.dump([occurences, frequency], co_occurence_file)

    occurences = zip(occurences, frequency)
    occurences = sorted(occurences, key=operator.itemgetter(1), reverse = True)

    return [hierarchy, occurences]



def multi_label_atomic(directory, max_h = 1):
    """
    Loads labels and blurbs of dataset
    """
    data = []
    for filename in os.listdir(directory):
        categories = set([])
        soup = BeautifulSoup(open(join(directory, filename), 'rt').read(), "html.parser")
        for t in soup.findAll('topics'):
            s1 = BeautifulSoup(str(t), "html.parser")
            structure = None
            if max_h == 1:
                structure = ['d0', 'd1']
            elif max_h == 2:
                structure = ['d0', 'd1', 'd2']
            elif max_h == 3:
                structure = ['d0', 'd1', 'd2', 'd3']
            for level in structure:
                for t1 in s1.findAll(level):
                    categories.add(str(t1.string))
        data.append((str(soup.find("body").string), categories))

    shuffle(data)
    return data
