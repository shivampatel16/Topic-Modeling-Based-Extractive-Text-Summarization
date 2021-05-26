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

def get_WikiHow_true_summaries(toy=False):
    SUMMARIES_PATH = "True Summaries" 
    WIKIHOW_SUMMARIES = []
    article_names = sorted(os.listdir(SUMMARIES_PATH))
    if toy:
        article_names = article_names[:50]
    for i, article_name in enumerate(article_names, start=1):
        if not article_name.endswith(".txt"):
            continue
        if (i % 1000 == 0):
            print(f"{i} summaries processed")
        with open(f'{SUMMARIES_PATH}/{article_name}', 'rb') as f:
            content = f.read().decode('utf-8')
            WIKIHOW_SUMMARIES.append(content)
    return WIKIHOW_SUMMARIES

WIKIHOW_TRUE_SUMMARIES = get_WikiHow_true_summaries()

import rouge
def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

for aggregator in ['Avg', 'Best']:
    print('Evaluation with {}'.format(aggregator))
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=4,
                           limit_length=True,
                           length_limit=100,
                           length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
    all_hypothesis = ds.summaries
    all_references = WIKIHOW_TRUE_SUMMARIES

    scores = evaluator.get_scores(all_hypothesis, all_references)

    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
            for hypothesis_id, results_per_ref in enumerate(results):
                nb_references = len(results_per_ref['p'])
                for reference_id in range(nb_references):
                    print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                    print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
            print()
        else:
            print(prepare_results(results['p'], results['r'], results['f']))
    print()