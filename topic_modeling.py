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

class TopicModeling():
    
    MALLET_PATH = "mallet-2.0.8/bin/mallet"

    def __init__(self, cleaned_data, num_topics=5, verbose=False, lda_model=None, is_mallet=False):
        print("\n\nCommencing topic modeling...\n\n")
        self.cleaned_data = cleaned_data
        self.num_topics = num_topics
        self.is_mallet = is_mallet
        self.dictionary = corpora.Dictionary(cleaned_data)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in cleaned_data]

        if lda_model:
            self.lda_model = lda_model
        else:
            self.lda_model = self.generate_lda_model()

        if not self.is_mallet:
            self.perplexity = self.compute_perplexity()
        
        self.coherence_score_cv, self.coherence_score_umass = self.compute_coherence() 
        if verbose:
							

            self.describe()
    def generate_lda_model(self):
        if not self.is_mallet:
            print("Generating LDA model...")
            lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=self.num_topics, random_state=100, update_every=1, chunksize=1000, passes=10, alpha='auto', per_word_topics=True)
        else:
            print("Generating LDA MALLET model...")
            lda_mallet = gensim.models.wrappers.LdaMallet(TopicModeling.MALLET_PATH, corpus=self.corpus, num_topics=self.num_topics, id2word=self.dictionary)
            lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_mallet)
        return lda_model
    
    def compute_perplexity(self):
        print("Computing perplexity...")
        perplexity = self.lda_model.log_perplexity(self.corpus)
        
        return perplexity
    
    def compute_coherence(self):
        print("Computing coherence scores...")
        coherence_model_lda_cv = CoherenceModel(model=self.lda_model, texts=self.cleaned_data, dictionary=self.dictionary, coherence='c_v')
        coherence_score_cv = coherence_model_lda_cv.get_coherence()
        
        coherence_model_lda_umass = CoherenceModel(model=self.lda_model, texts=self.cleaned_data, dictionary=self.dictionary, coherence='u_mass')
        coherence_score_umass = coherence_model_lda_umass.get_coherence()
        
        return coherence_score_cv, coherence_score_umass

							
    def visualize_topics(self, filename="lda_vis.html"):
        print("\nVisualizing LDA distribution of topics...")
        pyLDAvis.enable_notebook()

        vis = pyLDAvis.gensim.prepare(self.lda_model, self.corpus, self.dictionary)        
        pyLDAvis.save_html(vis, filename)
        return vis
    
    def describe(self):
        topic_ids, _ = zip(*self.lda_model.print_topics())
        
        print(f"\n\nNumber of topics : {len(topic_ids)}")
        
        
        for tid in topic_ids:
            topic_terms = self.lda_model.show_topic(tid)
            terms, weights = zip(*topic_terms)
            print(f"Topic {tid}: ")
            print("\tTerms:")
            print("\t\t", end="")
            for t in terms:
                print(t, end=", ")
            print("\n\tWeights:")
            print("\t\t", end="")
            for w in weights:
                print(w, end=", ")
            print("\n")       
        if not self.is_mallet:
            print(f"Perplexity: {self.perplexity}")
        print(f"Coherence Score (c_v): {self.coherence_score_cv}")
        print(f"Coherence Score (u_mass): {self.coherence_score_umass}")


tm = TopicModeling(dc.cleaned_data,num_topics=14, verbose=True)

with open("topic_model_wikihow.pkl", "wb") as outfile:
    pickle.dump(tm, outfile)

print("Topic Model Stored")

# Generating Topic Modeling Plot
tm.vis = tm.visualize_topics()
tm.vis