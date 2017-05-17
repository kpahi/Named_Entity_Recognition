import collections
import os
import os.path
import string
import sys

from nltk.stem.snowball import SnowballStemmer

ner_tags = collections.Counter()

basepath = os.path.dirname(__file__)
corpus_root = os.path.abspath(os.path.join(basepath, "gmb-2.2.0"))

# corpus_root = "gmb-2.2.0.zip"

for root, dirs, files in os.walk(corpus_root):
    for filename in files:
        if filename.endswith(".tags"):
            with open(os.path.join(root, filename), 'rb') as file_handle:
                #file_handle = zipfile.ZipFile('gmb-2.2.0.zip', 'r')
                file_content = file_handle.read().decode('utf-8').strip()
                annotated_sentences = file_content.split('\n\n')
                for annotated_sentence in annotated_sentences:
                    annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]
                    standard_form_tokens = []
                    for idx, annotated_token in enumerate(annotated_tokens):
                        annotations = annotated_token.split('\t')
                        word, tag, ner = annotations[0], annotations[1], annotations[3]

                        if ner != 0:
                            ner = ner.split('-')[0]

                        ner_tags[ner] += 1


# print(ner_tags)

# Training your own system
def features(tokens, index, history):
    """
    tokens = a POS-tagges sentences
    index = the index of the token we want to extract features for
    history = the previous predicated IOB tags

    """

    # init the SnowballStemmer
    stemmer = SnowballStemmer('english')

    # pad the sequences with the placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END2]', ('[END2]'))]
    history = ['[START2]', '[START1]'] + list(history)

    # shift the index with 2, to accommadate the padding
