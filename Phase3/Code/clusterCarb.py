from sklearn.cluster import KMeans
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from pandas import read_csv
from pandas import concat
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics


def read_path():
    print("Reading input directory ... ")

    current_dir = os.path.dirname(__file__)
    folder_path = os.path.join(current_dir, '..', 'Input')
    pre_configured_path = os.path.abspath(folder_path)
    temp_data_set_folder = input('Please enter the data set directory path\n'
                                 'OR\nHit enter to use %s: ' % pre_configured_path)
    if temp_data_set_folder.strip() != '':
        folder_path = temp_data_set_folder.strip()

    print("Reading input directory ... DONE.")

    return folder_path


def read_data(input_path):
    print("Reading data ...")
    # Lists all files in input directory
    input_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    raw_carb_data = pd.DataFrame()
    for input_file in input_files:
        if "Amount" in input_file:
            input_df = pd.read_csv(os.path.join(input_path, input_file),header=None)
            input_df=input_df[:51]
            print(input_df)
            raw_carb_data = pd.concat([raw_carb_data, input_df])

    print("Carb data size - ", raw_carb_data.shape)

    # print("Meal data -\n", raw_meal_df.head())

    print("Reading data ... DONE.")

    return raw_carb_data

def kmeans_clustering(data):
    print("Extracting clusters ...")

    km = KMeans(n_clusters=10, random_state=42)
    km.fit(data)
    y_label = km.labels_
    print(y_label)
    sil_score = silhouette_score(data, y_label)
    print(sil_score)

    print("Extracting clusters ... DONE.")
    return data



if __name__ == '__main__':
    input_path = read_path()
    raw_carb_data = read_data(input_path)
    print(raw_carb_data)
    raw_carb_data = kmeans_clustering(raw_carb_data)
