import pickle
import sys
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import json
import random
sys.path.append("data_processing/codenn/src")
from data_processing.code_processing import *
import pandas as pd
from keras.preprocessing.text import text_to_word_sequence



iid_labeled = pickle.load(open('annotation_tool/crowd_sourcing/python_annotator/all_agreed_iid_to_label.pickle','rb'))

q_code_snippet = pickle.load(open('annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle', 'rb'))

qid_to_title = pickle.load(open('annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_qid_to_title.pickle','rb'))

qid_code_labeled = dict([(key, q_code_snippet[key]) for key in iid_labeled])

tokenized_code, bool_failed_var, bool_failed_token = tokenize_code_corpus(qid_code_labeled, "python")

all_tokenized_code, all_bool_failed_var, all_bool_failed_token = tokenize_code_corpus(q_code_snippet, "python")

code_samples = [' '.join(tokenized_code[key]) for key in tokenized_code]

question_samples = [qid_to_title[qid] for qid, code_idx in iid_labeled]

samples = code_samples + question_samples

with open('data/samples_for_tokenizer.json', 'w') as write_file:
    json.dump(samples, write_file)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(samples)

# training_set: [{'question': [96, 3968, 21507, 13287, 16531, 4502], 'answers': [15916]}]

sample = {}


qid_code_tokenized = {}

for key, label in iid_labeled.items():
    qid, code_idx = key
    if label == 1:
        if qid in qid_code_tokenized:
            qid_code_tokenized[qid].append(' '.join(tokenized_code[key]))
        else:
            qid_code_tokenized[qid] = [' '.join(tokenized_code[key])]

sof_data = []
for qid, answers in qid_code_tokenized.items():
    sample = {}
    sample['question'] = tokenizer.texts_to_sequences([qid_to_title[qid]])[0]
    sample['answers'] = tokenizer.texts_to_sequences(answers)
    sof_data.append(sample)


train, test = train_test_split(sof_data, test_size = 0.33, random_state=20)

with open('data/train.json', 'w') as write_file:
    json.dump(train, write_file)

answers = []
for key, label in iid_labeled.items():
    answers.append(' '.join(tokenized_code[key]))


sample_answers = tokenizer.texts_to_sequences(answers)

with open('data/answers.json', 'w') as write_file:
    json.dump(sample_answers, write_file)


test_data = []
for q in test:
    sample = {}
    sample['question'] = q['question']
    sample['good'] = q['answers']
    sample['bad'] = random.sample(sample_answers, 150)
    test_data.append(sample)


with open('data/test.json', 'w') as write_file:
    json.dump(test_data, write_file)


# export to csv

questions_with_correct_answer = [key for key, value in iid_labeled.items() if value == 1]

questions, question_length, question_number_of_words,\
code_snippets, code_snippet_length, code_snippet_number_of_words, labels,\
    at_least_one_correct_answer = \
                                   [qid_to_title[qid] for qid, label in iid_labeled],\
                                   [len(qid_to_title[qid]) for qid, label in iid_labeled],\
                                   [len(text_to_word_sequence(qid_to_title[qid])) for qid, label in iid_labeled],\
                                   [q_code_snippet[key] for key in iid_labeled], \
                                   [len(q_code_snippet[key]) for key in iid_labeled], \
                                   [len(text_to_word_sequence(q_code_snippet[key])) for key in iid_labeled], \
                                   [label for key, label in iid_labeled.items()], \
                                   [1 if key in questions_with_correct_answer else 0 for key in iid_labeled]


df1 = pd.DataFrame({"questions": questions, "question_length": question_length,
                   "question_number_of_words": question_number_of_words,
                   "code_snippets": code_snippets, "code_snippet_length": code_snippet_length,
                   "code_snippet_number_of_words": code_snippet_number_of_words,
                   "labels": labels,
                   "at_least_one_correct_answer": at_least_one_correct_answer})


code_snippets_tokenized, code_snippet_tokenized_length, code_snippet_tokenized_number_of_words = \
    [' '.join(tokenized_code[key]) for key in iid_labeled], \
    [len(' '.join(tokenized_code[key])) for key in iid_labeled], \
    [len(text_to_word_sequence(' '.join(tokenized_code[key]))) for key in iid_labeled]


df2 = pd.DataFrame({"questions": questions, "question_length": question_length,
                   "question_number_of_words": question_number_of_words,
                   "code_snippets_tokenized": code_snippets_tokenized,
                    "code_snippet_tokenized_length": code_snippet_tokenized_length,
                   "code_snippet_tokenized_number_of_words": code_snippet_tokenized_number_of_words,
                   "labels": labels,
                   "at_least_one_correct_answer": at_least_one_correct_answer})


df1.to_csv("python_annotated_dataset.csv", encoding='utf-8')
df2.to_csv("python_annotated_dataset_tokenized.csv", encoding='utf-8')

# export all multi-code answer posts

questions, question_length, question_number_of_words,\
code_snippets, code_snippet_length, code_snippet_number_of_words,\
    labels, at_least_one_correct_answer = \
                                   [qid_to_title[qid] for qid, code in q_code_snippet],\
                                   [len(qid_to_title[qid]) for qid, code in q_code_snippet],\
                                   [len(text_to_word_sequence(qid_to_title[qid])) for qid, code in q_code_snippet],\
                                   [q_code_snippet[key] for key in q_code_snippet], \
                                   [len(q_code_snippet[key]) for key in q_code_snippet], \
                                   [len(text_to_word_sequence(q_code_snippet[key])) for key in q_code_snippet], \
                                   [iid_labeled[key] if key in iid_labeled else "N/A" for key in q_code_snippet], \
                                   [1 if key in questions_with_correct_answer else 0 for key in q_code_snippet]


df3 = pd.DataFrame({"questions": questions, "question_length": question_length,
                   "question_number_of_words": question_number_of_words,
                   "code_snippets": code_snippets, "code_snippet_length": code_snippet_length,
                   "code_snippet_number_of_words": code_snippet_number_of_words,
                   "labels": labels,
                    "at_least_one_correct_answer": at_least_one_correct_answer})


code_snippets_tokenized, code_snippet_tokenized_length, code_snippet_tokenized_number_of_words = \
    [' '.join(all_tokenized_code[key]) for key in q_code_snippet], \
    [len(' '.join(all_tokenized_code[key])) for key in q_code_snippet], \
    [len(text_to_word_sequence(' '.join(all_tokenized_code[key]))) for key in q_code_snippet]


df4 = pd.DataFrame({"questions": questions, "question_length": question_length,
                   "question_number_of_words": question_number_of_words,
                   "code_snippets_tokenized": code_snippets_tokenized,
                    "code_snippet_tokenized_length": code_snippet_tokenized_length,
                   "code_snippet_tokenized_number_of_words": code_snippet_tokenized_number_of_words,
                   "labels": labels,
                    "at_least_one_correct_answer": at_least_one_correct_answer})


df3.to_csv('python_all_multi_question_code_pair.csv', encoding='utf-8')
df4.to_csv('python_all_multi_question_code_pair_tokenized.csv', encoding='utf-8')





