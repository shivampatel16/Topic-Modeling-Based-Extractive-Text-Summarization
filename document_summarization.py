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

class DocSummarizing():

    def __init__(self, tc):
        self.topic_clusters = tc.topic_clusters
        self.topic_ids = tc.documentwise_topic_ids
        self.topic_ids_length = len(self.topic_ids)
        self.summaries = self.combine_summaries()
    
  
    def combine_summaries(self):
        complete_summaries = []
        for i, document in enumerate(self.topic_clusters):
            if (i%1000 == 0):
                print(f"{i} documents summarized")
            doc_summary = []
            for t_id in self.topic_ids[i]:
                if t_id in document.keys():
                    topic_content = ' '.join(document[t_id])
                    if (len(topic_content.split()) < 30):
                        summary = topic_content
                    else:
                        summary = self.textrank_summarize(topic_content)
                    doc_summary.append(summary)
                    if not summary:
                        print(f"Document ID {i}; Topic ID {t_id} - Empty")
            complete_summaries.append(' '.join(doc_summary))
        return complete_summaries
  
							
    def textrank_summarize(self, text):
        summary = summarizer.summarize(text, ratio=0.4)
        return summary


ds = DocSummarizing(tc)

with open("doc_summaries_tm_wikihow.pkl", "wb") as outfile:
    pickle.dump(ds.summaries, outfile)


print("Text Summaries Generated")