# Chase Brown
# CSCI Deep Learning Program 4
# RNN

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import os

vocabulary_size = 8000
unknown_token = "UNKNOWN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# break up txt documents into sentences and concatenate them
sentences = []
for file in os.listdir('stories'):
    with open('stories/'+file, encoding='utf8') as f:
        lines = f.read()
        s = nltk.sent_tokenize(lines)
        for i in range(len(s)):
            s[i] = s[i].replace('\n', '').lower()
        for j in range(len(s)):
            s[j] = ['%s %s %s' % (sentence_start_token, s[j], sentence_end_token)]

        sentences = sentences + s


characters = list(map(chr, range(0, 256)))

# Tokenize the sentences into words
toke_sent = []
for i in range(len(sentences)):
    L = list(sentences[i][0])
    toke_sent = toke_sent + L

print('break')