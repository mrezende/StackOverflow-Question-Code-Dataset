import pickle
import sys
sys.path.append("data_processing/codenn/src")
from data_processing.code_processing import *

iid_labeled = pickle.load(open('annotation_tool/crowd_sourcing/python_annotator/all_agreed_iid_to_label.pickle','rb'))

q_code_snippet = pickle.load(open('annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle', 'rb'))

qid_code_labeled = dict([(key, q_code_snippet[key]) for key in iid_labeled])

tokenized_code, bool_failed_var, bool_failed_token = tokenize_code_corpus(qid_code_labeled, "python")

code_samples = [' '.join(tokenized_code[key]) for key in tokenized_code]




