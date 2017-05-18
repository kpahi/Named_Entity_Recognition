import pickle
from collections import Iterable

import nltk.chunk
from nltk import pos_tag, word_tokenize
from nltk.chunk import ChunkParserI
from nltk.tag import ClassifierBasedTagger

from ner_with_python import *


class NameEntity(ChunkParserI):

    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)
        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(train=train_sents, feature_detector=features, **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)

        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        return nltk.chunk.conlltags2tree(iob_triplets)

reader = read_gmb(corpus_root)
data = list(reader)
training_samples = data[:int(len(data) * 0.9)]
test_samples = data[int(len(data) * 0.9):]

print("#Training samples = %s" % len(training_samples))
print("#Test Samples = %s" % len(test_samples))


chunker = NameEntity(training_samples[:8000])

#print(chunker.parse(pos_tag(word_tokenize("I'm going to Germany this Monday."))))
