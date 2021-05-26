import pandas as pd
import os
import re

import nltk
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')

import pickle
import os
import re
import time
import numpy as np
import pandas as pd
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt
%matplotlib inline

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.stem import WordNetLemmatizer
import spacy
import rouge
from summa import summarizer

from nltk.corpus import stopwords
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

WIKIHOW_DATASET_LINK = "https://query.data.world/s/pzgavxogamylsoe73yr4levyklwusi"

# read data from the CSV file (from the location it is stored)
Data = pd.read_csv(WIKIHOW_DATASET_LINK)
Data = Data.astype(str)
rows, columns = Data.shape

title_file = open('Title List.txt', 'wb')

# The path where the articles are to be saved
articles_path = "Articles"
summaries_path = "True Summaries"

if not os.path.exists(articles_path): os.makedirs(articles_path)
if not os.path.exists(summaries_path): os.makedirs(summaries_path)

							
# go over the all the articles in the data file
for row in range(rows):
    if Data.iloc[row, 0] == "\n":
        continue
    abstract = Data.iloc[row, 0]      # headline is the column representing the summary sentences
    article = Data.iloc[row, 2]       # text is the column representing the article

    #  a threshold is used to remove short articles with long summaries as well as articles with no summary
    if len(abstract) < (0.75*len(article)):
        # remove extra commas in abstracts
        abstract = abstract.strip()
        abstract = re.sub("\.,*\s+,*", ". ", abstract)
        abstract = re.sub("\n+", "\n", abstract)
        abstract = re.sub("[ ]+", " ", abstract)
        abstract = re.sub("\n[,]\s+", "\n", abstract)
        abstract = abstract.encode('utf-8')
        
        # remove extra commas in articles
        article = article.strip()
        article = re.sub("\.,*\s+,*", ". ", article)
        article = re.sub("\n+", "\n", article)
        article = re.sub("[ ]+", " ", article)
        article = re.sub("\n[,]\s+", "\n", article)
        article = article.encode('utf-8')
        
        # file names are created using the alphanumeric characters from the article titles.
        # they are stored in a separate text file.
        filename = Data.iloc[row, 1]
        filename = "".join(x for x in filename if x.isalnum())
        filename1 = filename + '.txt'
        filename = filename.encode('utf-8')
        title_file.write(filename+b'\n')

        with open(f"{summaries_path}/{filename1}",'wb') as t:
            t.write(abstract)

							
       with open(f"{articles_path}/{filename1}",'wb') as f:
            f.write(article)

title_file.close()


def get_WikiHow_articles(toy=False):
    ARTICLES_PATH = "Articles" 
    WIKIHOW_ARTICLES = []
    article_names = sorted(os.listdir(ARTICLES_PATH))
    if toy:
        article_names = article_names[:50]
    for i, article_name in enumerate(article_names, start=1):
        if not article_name.endswith(".txt"):
            continue
        if (i % 1000 == 0):
            print(f"{i} articles processed")
        with open(f'{ARTICLES_PATH}/{article_name}', 'rb') as f:
            content = f.read().decode('utf-8')
            WIKIHOW_ARTICLES.append(content)
    return WIKIHOW_ARTICLES


WIKIHOW_ARTICLES = get_WikiHow_articles()

import en_core_web_sm

class DataCleaning():
    def __init__(self, documents):
        print("\n\nCommencing data cleaning...\n\n")
        
        self.data = documents
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(['say', 'also', 'may', 'use', 'make', 'want', 'get', 'time', 'try', 'thing', 'good','even', 'could', 'keep', 'take', 'go', 'way', 'example', 'include', 'add', 'must', 'sure', 'provide', 'require', 'consider', 'leave', 'would', 'different', 'first', 'set', 'usually', 'feel', 'help', 'find', 'give', 'look'])
        
        #self.nlp = spacy.load('en', disable=['parser', 'ner'])
        self.nlp = en_core_web_sm.load()
				
        cleaned_data = self.remove_extra_spaces(self.data)
        cleaned_data = self.tokenization_and_stopword_removal(cleaned_data)
        cleaned_data =  self.bigram_generation(cleaned_data)
        cleaned_data = self.trigram_generation(cleaned_data)
        self.cleaned_data = self.lemmatization(cleaned_data)
    
    def remove_extra_spaces(self, documents):
        print("Removing extra spaces...")
        data = [re.sub(r"\d+", " ", doc) for doc in documents]
        data = [re.sub(r"\'", "", doc) for doc in data]
        return data
    
    def tokenization_and_stopword_removal(self, documents):
        print("Tokenizing and removing stopwords...")
        return [[word for word in simple_preprocess(str(doc)) if word not in self.stop_words] for doc in documents]
    
    def bigram_generation(self, documents_list):
        print("Generating valid bigrams...")
        self.bigram = gensim.models.Phrases(documents_list, min_count=5, threshold=100)
        self.bigram_mod = gensim.models.phrases.Phraser(self.bigram)
        return [self.bigram_mod[doc] for doc in documents_list]
    
    def trigram_generation(self, documents_list):
        print("Generating valid trigrams...")
        self.trigram = gensim.models.Phrases(self.bigram[documents_list], threshold=100) 
        self.trigram_mod = gensim.models.phrases.Phraser(self.trigram)
        return [self.trigram_mod[self.bigram_mod[doc]] for doc in documents_list]
    
    def lemmatization(self, documents, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        print("Lemmatizing documents...")
        docs_out = []
        for document in documents:
            doc = self.nlp(" ".join(document)) 
            docs_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags and token.lemma_ not in self.stop_words])
        return docs_out
		
dc = DataCleaning(WIKIHOW_ARTICLES)

with open("cleaned_data_wikihow.pkl", "wb") as outfile:
    pickle.dump(dc.cleaned_data, outfile)

print("Cleaned Data Stored")