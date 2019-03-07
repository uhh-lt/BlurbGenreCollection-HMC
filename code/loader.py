"""
Author: Rami Aly, E-mail: `rami.aly@outlook.com`
"""
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
from loader_abstract import Loader_Interface
#Maybe use yield, better style
#DIRECTORY = '/home/rami/Documents/Bachelorarbeit/resources'

DEST_DIRECTORY_EN = join(os.path.dirname(os.path.abspath(__file__)), '../', 'datasets')



class Blurb_Loader(Loader_Interface):


    def __init__(self):
        self._setName("EN")


    def load_data_multiLabel(self):
        """
        Loads Multilabel data with max_h hierarchy level, so 1 includes all labels that have hat max height one in label tree
        @param dev: specifies whether the dev set should be used or not
        """
        dest_directory = DEST_DIRECTORY_EN

        return (self.multi_label_atomic(join(dest_directory, 'BlurbGenreCollection_EN_train.txt')),
         self.multi_label_atomic(join(dest_directory, 'BlurbGenreCollection_EN_dev.txt')),
          self.multi_label_atomic(join(dest_directory, 'BlurbGenreCollection_EN_test.txt')))



    def read_relations(self):
        """
        Loads hierarchy file and returns set of relations
        """
        relations = set([])
        singeltons = set([])
        REL_FILE =  join(os.path.dirname(os.path.abspath(__file__)), '../datasets', 'hierarchy.txt')
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


    def load_outlier(self):
        """
        Loads low-frequency dataset
        """
        outlier_directory = join(os.path.dirname(os.path.abspath(__file__)), '../resources', 'EN_outlier')
        return multi_label_atomic(outlier_directory)



    def multi_label_atomic(self, directory):
        """
        Loads labels and blurbs of dataset
        """
        data = []
        soup = BeautifulSoup(open(join(directory), 'rt').read(), "html.parser")
        for book in soup.findAll('book'):
            categories = set([])
            book_soup = BeautifulSoup(str(book), "html.parser")
            for t in book_soup.findAll('topics'):
                s1 = BeautifulSoup(str(t), "html.parser")
                structure = ['d0', 'd1', 'd2', 'd3']
                for level in structure:
                    for t1 in s1.findAll(level):
                        categories.add(str(t1.string))
            data.append((str(book_soup.find("body").string), categories))
            #print(data[0])

        shuffle(data)
        return data
