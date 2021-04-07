# -*- coding: utf-8 -*-
from typing import List
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from PIL import ImageTk, Image
import csv
import nltk
from nltk.tokenize import RegexpTokenizer
import collections
from collections import Counter
import codecs
import sys
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from enum import Enum

from matplotlib.figure import Figure
from IPython import get_ipython
from IPython.display import display


class Analyzer:
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words("german")
        self.x_graph = range(1, 26)
        self.html_request = {}

    def bargraph(self, namen, daten, titel):  # fuer die schönen Graphen
        fig = plt.figure()
        plt.bar(self.x_graph, daten, align="center")
        plt.xticks(self.x_graph, namen, rotation=45, ha='right')
        plt.title(titel)
        plt.tight_layout()
        plt.show()

    async def add_html_url(self, url: str):
        if url not in self.html_request:
            self.html_request[url] = ""
        print(self.html_request)

    async def request_get_html(self, url: str):
        if url in self.html_request:
            self.html_request[url] = requests.get(url)
            print(self.html_request[url])
        else:
            print("url nicht gefunden")

    async def check_result_status(self, url):
        if url in self.html_request:
            result = self.html_request.get(url)
            print(result.status_code)

            if str(result.status_code) == "200":
                return True
            else:
                return False

    async def save_html(self, url):
        if url in self.html_request:
            soup = BeautifulSoup(self.html_request.get(url).content, "lxml")
            now = datetime.now()
            now = now.strftime("%Y_%m_%d_%H_%M_%S")
            with open("Soups/{}_output.html".format(now), "w", encoding='utf8') as file:
                file.write(str(soup))
        else:
            print("url nicht gefunden")

    def read_csv(self, input_path):
        with codecs.open(input_path, "r", "utf-8") as csv_input:
            esa_reader = csv.reader(csv_input, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
        return esa_reader

    def filter_words_keyword(self, data, keyword):
        result = []
        for column in data:
            for word in column[4].split():
                if keyword in word:
                    result.append(word)  # ranking aller begriffe mit keyword
        counted_results = Counter(result)
        counted_results_dict = dict(counted_results.most_common(25))
        results_keys = counted_results_dict.keys()
        results_values = counted_results_dict.values()
        return result, counted_results, results_keys, results_values

    def filter_words_stopwords(self, data):
        result = []
        for column in data:
            for word in column[4].split():
                if self.stopwords not in word:
                    result.append(word)  # ranking aller nicht stopworten
        counted_results = Counter(result)
        counted_results_dict = dict(counted_results.most_common(25))
        results_keys = counted_results_dict.keys()
        results_values = counted_results_dict.values()
        return result, counted_results, results_keys, results_values

    def filter_date(self, data):
        result = []
        for column in data:
            for datum in column[5].split():
                if Tokens.HYPHEN in datum:
                    result.append(datum)  # ranking aller begriffe mit keyword
        counted_results = Counter(result)
        counted_results_dict = dict(counted_results.most_common(25))
        results_keys = counted_results_dict.keys()
        results_values = counted_results_dict.values()
        return result, counted_results, results_keys, results_values

    def filter_tweets_date(self, data, search_date):
        result = []
        for column in data:
            for date in column[5]:
                if search_date in date:
                    result.append(date[4])  # ranking aller begriffe mit keyword
        return result

    def likelihood_keyword(self, data_one, data_two, keyword):
        result_words_one = []
        result_words_two = []
        result_same_words = []


        for column in data_one:
            for word in column[4].split():
                if keyword in word:
                    result_words_one.append(word)  # ranking aller nicht stopworten

        for column in data_two:
            for word in column[4].split():
                if keyword in word:
                    result_words_two.append(word)  # ranking aller nicht stopworten


        for word in result_words_one:
            if keyword in result_words_two:
                result_same_words.append(word)  # ranking aller nicht stopworten


        for word in result_words_two:
            if keyword in result_words_one:
                result_same_words.append(word)  # ranking aller nicht stopworten

        counted_results = Counter(result_same_words)
        counted_results_dict = dict(counted_results.most_common(25))
        results_keys = counted_results_dict.keys()
        results_values = counted_results_dict.values()

        return counted_results, results_keys, results_values


    def createplot(self, datax, datay, datadict, output_path):
        print("creating plot")
        fig, ax = plt.subplots(figsize=(5, 5))  # Create a figure and an axes.
        ax.plot(datax, datay, label='linear')  # Plot some data on the axes.
        # ax.plot(x, x ** 2, label='quadratic')  # Plot more data on the axes...
        # ax.plot(x, x ** 3, label='cubic')  # ... and some more.
        ax.set_xlabel('x label')  # Add an x-label to the axes.
        ax.set_ylabel('y label')  # Add a y-label to the axes.
        ax.set_title("DATA")  # Add a title to the axes.
        ax.legend()  # Add a legend.
        plt.grid(axis='both', color='0.95')
        fig.savefig(output_path + 'dfdataPLOT.png')

        #plt.show()
        dfdata = pd.DataFrame.from_dict(datadict)
        dfdata.to_csv(output_path + 'sorted_data.csv', header=True, quotechar=' ', index=True, sep=';', mode='a', encoding='utf8')
        return fig, ax
        #self.view.plt.plot([1, 2, 3, 4])
        #self.view.plt.ylabel('some numbers')
        #self.view.plt.show()

    def create_img(self, Image):
        return
        # display(Image.fromarray(image_np))
        # plt.figure(figsize=IMAGE_SIZE)
        # plt.imshow(image_np)
        # # plt.show(output_dict)
        # # plt.show()

###############################################################################
#                                                                             #
#  List of some                                        #
#                                                                             #
###############################################################################
class Tokens(Enum):
    HASHTAG = "#"
    AT = '@'
    HYPHEN = '-'
