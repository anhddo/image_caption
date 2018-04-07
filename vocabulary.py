import nltk
import pickle
import argparse
from collections import Counter
from operator import itemgetter
import itertools
import json
from utils import get_image_names


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def convert_sentence(self, ids):
        sentence = ''
        for i in ids:
            if self.idx2word[i] == '<end>':
                break
            sentence = sentence + ' '+ self.idx2word[i]

        return sentence

