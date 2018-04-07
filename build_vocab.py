import nltk
import pickle
import argparse
from collections import Counter
from operator import itemgetter
import itertools
import json
from vocabulary import Vocabulary
from utils import get_image_names

def build_vocab(caption, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    n = len(caption.keys())
    for i, key in enumerate(caption.keys()):
        for sentence in caption[key]:
            tokens = nltk.tokenize.word_tokenize(sentence)
            counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, n))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def get_image_caption(train_image_name_path, caption_path):
    image_names = get_image_names(train_image_name_path)
    assert len(image_names) > 0
    caption = {}
    with open(caption_path, 'r') as f:
        trim = lambda l: [l[0], l[1][2: -1]]
        lines = [trim(line.split("#")) for line in f.readlines()]
        lines = sorted(lines, key = itemgetter(0))

        for k, g in itertools.groupby(lines, key = itemgetter(0)):
            caption[k] = [e[1] for e in g]

    return image_names, caption

def main(args):
    _, caption = get_image_caption(args.train_image_name_path, args.caption_path)


    vocab = build_vocab(caption, threshold=args.threshold)
    with open(args.vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %args.vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_name_path', type=str,
            default='./data/Flickr8k_text/Flickr_8k.trainImages.txt', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--caption_path', type=str, 
                        default='data/Flickr8k_text/Flickr8k.token.txt')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
