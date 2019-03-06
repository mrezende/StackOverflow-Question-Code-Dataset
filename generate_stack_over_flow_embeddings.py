import matplotlib
matplotlib.use('Agg')
import pickle
import sys
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import json
from gensim.models import Word2Vec
import random
import numpy as np
sys.path.append("data_processing/codenn/src")
from data_processing.code_processing import *
from keras.preprocessing.text import text_to_word_sequence
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



iid_labeled = pickle.load(open('annotation_tool/crowd_sourcing/python_annotator/all_agreed_iid_to_label.pickle','rb'))

q_code_snippet = pickle.load(open('annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle', 'rb'))

qid_to_title = pickle.load(open('annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_qid_to_title.pickle','rb'))

qid_code_labeled = dict([(key, q_code_snippet[key]) for key in iid_labeled])

tokenized_code, bool_failed_var, bool_failed_token = tokenize_code_corpus(qid_code_labeled, "python")

code_samples = [' '.join(tokenized_code[key]) for key in tokenized_code]

question_samples = [qid_to_title[qid] for qid, code_idx in iid_labeled]

samples = code_samples + question_samples

samples_preprocessed = [text_to_word_sequence(s) for s in samples]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(samples)

word_index = tokenizer.word_index

# run model
size = 100
model = Word2Vec(samples_preprocessed, size=size, min_count=1, window=5, sg=1, iter=15)
weights = model.wv.syn0
d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
emb = np.zeros(shape=(len(word_index)+1, size), dtype='float32')

for w, i in word_index.items():
    if w not in d: continue
    emb[i, :] = weights[d[w], :]

word_vectors = model.wv

wanted_words = []
count = 0
for word in word_vectors.vocab:
    if count<150:
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

plt.savefig('tsne-output.png')
plt.clf()

np.save(open('word2vec_%d_dim.embeddings' % size, 'wb'), emb)


# generate histograms for setence length and number of words

question_length = [len(qid_to_title[qid]) for qid, label in iid_labeled]

plt.hist(question_length, bins='auto')
plt.title("Question Length")
plt.savefig('question_length_hist.png')
plt.clf()

question_number_of_words = [len(text_to_word_sequence(qid_to_title[qid])) for qid, label in iid_labeled]

plt.hist(question_number_of_words, bins='auto')
plt.title("Number of Words of Question")
plt.savefig('question_number_of_words_hist.png')
plt.clf()

code_snippet_length = [len(q_code_snippet[key]) for key in iid_labeled]

plt.hist(code_snippet_length, bins='auto')
plt.title("Code Snippet Length")
plt.savefig('code_snippet_length_hist.png')
plt.clf()

code_snippet_number_of_words = [len(text_to_word_sequence(q_code_snippet[key])) for key in iid_labeled]

plt.hist(code_snippet_number_of_words, bins='auto')
plt.title("Number of Words of Code Snippet")
plt.savefig('code_snippet_number_of_words_hist.png')
plt.clf()

code_snippet_tokenized_length = [len(' '.join(tokenized_code[key])) for key in iid_labeled]

plt.hist(code_snippet_tokenized_length, bins='auto')
plt.title("Code Snippet Tokenized Length")
plt.savefig('code_snippet_tokenized_length_hist.png')
plt.clf()

code_snippet_tokenized_number_of_words = [len(text_to_word_sequence(' '.join(tokenized_code[key])))
                                          for key in iid_labeled]

plt.hist(code_snippet_tokenized_number_of_words, bins='auto')
plt.title("Number of Words of Code Snippet Tokenized")
plt.savefig('code_snippet_tokenized_number_of_words_hist.png')
plt.clf()


