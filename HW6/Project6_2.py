#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 11:40:16 2018

@author: tianxiang
"""
import pandas as pd
from nltk.tokenize import word_tokenize


def find_A(doc, word_list):
    return list(set(doc) & set(word_list))


def find_B(doc, word_list):
    retlist = []
    for w in doc:
        if w in word_list:
            retlist.append(w)
    return retlist


LM_Negative = pd.read_csv('LM_Negative.csv')
LM_Positive = pd.read_csv('LM_Positive.csv')
neg_words = []
pos_words = []


for item in LM_Negative.iloc[:, 0]:
    neg_words.append(item.replace('\xa0', '').lower())
for item in LM_Positive.iloc[:, 0]:
    pos_words.append(item.replace('\xa0', '').lower())

document = "My selection this month is a best-in-class retailer. " \
           "It has achieved stunning success where so many others in " \
           "its industry have staggered, lagged, and become downright outdated. " \
           "This is a retailer that values its employees because it " \
           "believes that happy employees make for happy customers and, presumably, " \
           "happy shareholders. This company keeps delivering amazing growth, quarter after " \
           "quarter, year after year." \
           "I'm speaking of Whole Foods Market. " \
           "Over the years, this company has grown in so many ways. At first, " \
           "it appeared to be little more than a haven for hippie types. It grew some more.  " \
           "Then some argued that organic food was just a fad. Whole Foods kept growing. Well," \
           " then the naysayers said it was just a haven for bohemian suburbanites and downtown hipsters with comfortable paychecks. " \
           "Oh, yeah? These days, I'm wondering if its next step is to take over the world as it shakes up traditional " \
           "grocers such as Safeway, Kroger, and even Wal-Mart."
           
words = word_tokenize(document)
print('\nPOSITIVE WORDS:\n')
pos_found_A = find_A(words, pos_words)
pos_found_B = find_B(words, pos_words)
print(pos_found_A)
print(pos_found_B)

print('\nNEGATIVE WORDS:\n')
neg_found_A = find_A(words, neg_words)
neg_found_B = find_B(words, neg_words)
print(neg_found_A)
print(neg_found_B)

print('\nBINARY APPROACH:\n')
print('Pos: %i' % len(pos_found_A))
print('Pos: %i' % len(neg_found_A))


print('\nTERM FREQUENCY APPROACH:\n')
print('Pos: %i' % len(pos_found_B))
print('Pos: %i' % len(neg_found_B))


print('\nPOLARITY:\n')
polarity = (len(pos_found_B) - len(neg_found_B)) / ((len(pos_found_B) + len(neg_found_B)))
print(polarity)
