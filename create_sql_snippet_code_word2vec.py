import matplotlib
import pickle
import sys
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import json
from gensim.models import Word2Vec
import random
import numpy as np
import os
from data_processing.code_processing_original import *
from keras.preprocessing.text import text_to_word_sequence
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from ast import literal_eval


q_code_snippet = pickle.load(open('annotation_tool/data/code_solution_labeled_data/source/sql_how_to_do_it_by_classifier_multiple_iid_to_code.pickle', 'rb'))

print(len(q_code_snippet))

tokenized_code, bool_failed_var, bool_failed_token = tokenize_code_corpus(q_code_snippet, "sql")

code_corpus = []
for key, value in tokenized_code.items():
  code_corpus.append(value)


# run model
# create word2vec
size = 100
model = Word2Vec(code_corpus, size=size, min_count=5, window=5, sg=1, iter=15)



iid_labeled = []
with open('final_collection/sql_multi_code_iids.txt','r') as f:
  lines = f.readlines()
  for line in lines:
    iid_labeled.append(literal_eval(line))

qid_to_title = pickle.load(open('annotation_tool/data/code_solution_labeled_data/source/sql_how_to_do_it_by_classifier_multiple_iid_to_code.pickle','rb'))

qid_code_labeled = dict([(key, q_code_snippet[key]) for key in iid_labeled])

tokenized_code, bool_failed_var, bool_failed_token = tokenize_code_corpus(qid_code_labeled, "sql")

code_samples = [' '.join(tokenized_code[key]) for key in tokenized_code]

question_samples = [qid_to_title[qid] for qid, code_idx in iid_labeled]

samples = code_samples + question_samples

print(len(samples))

tokenizer = Tokenizer()

tokenizer.fit_on_texts(samples)

word_index = tokenizer.word_index

print(len(word_index))

weights = model.wv.vectors
d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
emb = np.zeros(shape=(len(word_index)+1, size), dtype='float32')

for w, i in word_index.items():
    if w not in d: continue
    emb[i, :] = weights[d[w], :]

word_vectors = model.wv

np.save(open('word2vec_code_%d_dim.embeddings' % size, 'wb'), emb)

print(emb.shape)


sorted_by_word_count = sorted(tokenizer.word_counts.items(), key=lambda (word, count): count, reverse=True)
wanted_words = []
count = 0
for word, freq in sorted_by_word_count:
    if count<200:
        wanted_words.append(word)
        count += 1
    else:
        break

wanted_vocab = dict((k, word_vectors.vocab[k]) for k in wanted_words if k in word_vectors.vocab)


X = model[wanted_vocab] # X is an array of word vectors, each vector containing 150 tokens
tsne_model = TSNE(perplexity=40, n_components=2, init="pca", n_iter=5000, random_state=23)
Y = tsne_model.fit_transform(X)


fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(Y[:, 0], Y[:, 1])
words = list(wanted_vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(Y[i, 0], Y[i, 1]))


ax.set_yticklabels([]) #Hide ticks
ax.set_xticklabels([]) #Hide ticks

plt.show()
plt.savefig('code-tsne-output.png')
plt.clf()




