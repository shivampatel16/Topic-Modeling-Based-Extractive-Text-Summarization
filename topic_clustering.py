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

class TopicClustering():
    
    lemmatizer = WordNetLemmatizer().lemmatize
    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    MIN_SENTENCE_LENGTH = 0
    MAX_SENTENCE_LENGTH = 1000 
    
    def __init__(self, documents, topic_model):
        print("\n\nCommencing topic clustering...\n\n")

        self.documents = documents
        self.lda_model = topic_model.lda_model
        self.corpus = topic_model.corpus
        self.dictionary = topic_model.dictionary
        self.dataframe = pd.DataFrame(columns=["Document No.", "Sentence No.", "Topic ID", "Topic Ratio", "Sentence"])

        self.sentence_groups = self.get_sentence_groups()
        self.distributions = self.get_sentence_distributions()
        self.topic_clusters = self.get_topicwise_clusters_per_document()
        self.documentwise_topic_ids = self.get_documentwise_topic_ids()
        
    
    def tokenizer(self, document): # returns a list of tokens
        text = re.sub('[^a-zA-Z]', ' ', document)
        tokens = text.lower().split()
        tokens = [TopicClustering.lemmatizer(tkn) for tkn in tokens]
        return tokens
    def get_sentence_groups(self): 
        print("\nGenerating sentence groups...")
        sentence_groups = []

							
        for i, document in enumerate(self.documents, start=1):
            if (i%1000 == 0):
                print(f"{i} documents processed")
            sentences = TopicClustering.sentence_detector.tokenize(document)
            sentence_group = []
            for (k, sentence) in enumerate(sentences):
                words = sentence.split()
                if (len(words) > TopicClustering.MIN_SENTENCE_LENGTH and len(words) < TopicClustering.MAX_SENTENCE_LENGTH):
                    sentence_group.append((k, sentence))
            sentence_groups.append(sentence_group)
        return sentence_groups
    
    def get_sentence_distributions(self):
        print("\nGenerating sentence distributions...")
        distributions = []
        dataframe_list = []
        for i, sentences in enumerate(self.sentence_groups, start=1):
            if (i%1000 == 0):
                print(f"{i} documents processed")
            sentence_distributions = []
            for k, sentence in sentences:
                tokens = self.tokenizer(sentence)
                if not tokens:
                    continue
                bow = self.dictionary.doc2bow(tokens)
                dist = self.lda_model.get_document_topics(bow)
                try:
                    dist = max(dist, key=lambda x: x[1]) # get dominant topic with percentage
                except ValueError as e:
                    logging.error(e)
                    continue
	 sentence_distributions.append((k, dist))
                dataframe_list.append({"Document No.": i-1, "Sentence No.": k, "Topic ID": dist[0], "Topic Ratio": dist[1], "Sentence": sentence})
            distributions.append(sentence_distributions)
        self.dataframe = pd.DataFrame(dataframe_list)

							
        self.dataframe.set_index(["Document No.", "Sentence No."], inplace=True)
        return distributions
    
    
    def get_topicwise_clusters_per_document(self):
        print("\nGenerating documentwise topic clusters...")
        topicwise_documents_list = []
        topic_ids, _ = zip(*self.lda_model.print_topics())
        
        for doc_no in range(len(self.sentence_groups)):
            if ((doc_no+1)%1000 == 0):
                print(f"{doc_no+1} documents processed")
            topicwise_document = {}
            grouped_doc_df = self.dataframe.loc[doc_no].groupby("Topic ID", sort=False) 
            for topic_id in topic_ids:
                topicwise_document[topic_id] = grouped_doc_df.get_group(topic_id)["Sentence"].tolist() if topic_id in grouped_doc_df.groups else []
            topicwise_documents_list.append(topicwise_document)
        return topicwise_documents_list
  
    
    def get_documentwise_topic_ids(self):
        print("\nGenerating documentwise dominant topic ids...")
        topic_id_list = []
        for i, document in enumerate(self.documents, start=1):
            if (i%1000 == 0):
                print(f"{i} documents processed") 
            tokens = self.tokenizer(document)
            if not tokens:
                continue
            bow = self.dictionary.doc2bow(tokens)
            topics_list = self.lda_model.get_document_topics(bow)
            topics_list = sorted(topics_list, key=lambda x: x[1], reverse=True) # get sorted list of dominant topics with percentage
            topics_list = [tl[0] for tl in topics_list]
            topic_id_list.append(topics_list)
        return topic_id_list


tc = TopicClustering(WIKIHOW_ARTICLES, tm)

with open("topic_clusters_wikihow.pkl", "wb") as outfile:
    pickle.dump(tc, outfile)

print("Topic Clusters formed")