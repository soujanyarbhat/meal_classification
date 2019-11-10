# script that checks the acuuracy of test set(mimic-ed as unseen records)

import os
from os import listdir
from os.path import isfile, join

import pandas as pd

current_dir = os.path.dirname(__file__)
input_path = os.path.join(current_dir, '..', 'Output')
truth_path = os.path.join(current_dir, '..', 'Test', 'GroundTruth.csv')
# prediction files for all members
input_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
# ground truth DF
ground_truth_df = pd.read_csv(truth_path, header = None)
# print('GROUND TRUTH ------------------------------\n', ground_truth_df)
# Accuracy
for input_file in input_files:
    predictions_df = pd.read_csv(os.path.join(input_path, input_file), header = None)
    accuracy_df = ground_truth_df.where(ground_truth_df.values == predictions_df.values).notna()
    # print('PREDICTIONS ------------------------------\n', predictions_df)
    # print('ACCURACY ------------------------------\n', accuracy_df)
    hits = accuracy_df.sum()
    rows, cols = ground_truth_df.shape
    misses = rows - hits
    accuracy = (hits / rows) * 100
    print('{} -> {}'.format(input_file, accuracy))
