import pickle
import sys
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import json
import gensim
import random
import numpy as np
sys.path.append("data_processing/codenn/src")
from data_processing.code_processing import *



iid_labeled = pickle.load(open('annotation_tool/crowd_sourcing/python_annotator/all_agreed_iid_to_label.pickle','rb'))

q_code_snippet = pickle.load(open('annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle', 'rb'))

qid_to_title = pickle.load(open('annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_qid_to_title.pickle','rb'))

qid_code_labeled = dict([(key, q_code_snippet[key]) for key in iid_labeled])

tokenized_code, bool_failed_var, bool_failed_token = tokenize_code_corpus(qid_code_labeled, "python")

code_samples = [' '.join(tokenized_code[key]) for key in tokenized_code]

question_samples = [qid_to_title[qid] for qid, code_idx in iid_labeled]

samples = code_samples + question_samples

samples_preprocessed = [gensim.utils.simple_preprocess(s, deacc=True) for s in samples]



# run model
size = 100
model = Word2Vec(samples_preprocessed, size=size, min_count=5, window=5, sg=1)
weights = model.syn0
d = dict([(k, v.index) for k, v in model.vocab.items()])
emb = np.zeros(shape=(len(vocab)+1, args.size), dtype='float32')

for i, w in vocab.items():
    if w not in d: continue
    emb[i, :] = weights[d[w], :]

np.save(open('word2vec_%d_dim.embeddings' % size, 'wb'), emb)

