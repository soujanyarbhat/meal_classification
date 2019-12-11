# import packages
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import sklearn
from pandas import read_csv
from pandas import concat
from pandas import DataFrame
from collections import defaultdict
from collections import Counter
from functools import partial
from scipy import stats
from scipy import fftpack
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import warnings

from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
from sklearn.cluster import DBSCAN

warnings.simplefilter(action = 'ignore', category = FutureWarning)
import pickle


class MealClustering:

    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'Output/')

    def __init__(self):
        print("Meal Clustering Model ...")


    def read_path(self):
        """
        Input path for data set- USER INPUT
        :return: input folder path
        """
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

    def read_data(self, input_path):
        """
        Reads meal data and no meal data separately
        :param input_path: path to input directory
        :return: raw meal data
        """
        print("Reading data ...")

        # Handles issues with ragged CSV file
        col_names = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10",
                     "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20",
                     "Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30", "Col31"]
        # Lists all files in input directory
        input_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
        raw_meal_df = pd.DataFrame()
        raw_carb_df = pd.DataFrame()
        for input_file in input_files:
            if "Amount" not in input_file:
                input_df = pd.read_csv(os.path.join(input_path, input_file), names = col_names)
                raw_meal_df = pd.concat([raw_meal_df, input_df])
            else:
                input_df = pd.read_csv(os.path.join(input_path, input_file), header=None)
                input_df = input_df[:51]
                raw_carb_df = pd.concat([raw_carb_df, input_df])

        print("Meal data size - ", raw_meal_df.shape)

        # print("Meal data -\n", raw_meal_df.head())

        print("Reading data ... DONE.")

        return raw_meal_df, raw_carb_df

    def preprocess_data(self, raw_meal_df, raw_carb_data):
        """
        Reverses columns
        Removes rows- all NaN or high NaN count
        Interpolates NaN values
        :param raw_meal_df:
        :param raw_carb_data:
        :return: processed meal data frame
        """
        print("Pre-processing ...")

        # TODO: Handle NaN - CURRENT: delete col 31. NOTE = no NaN in TEST DATA

        del raw_meal_df['Col31']
        raw_meal_df['carbs'] = raw_carb_data[0]
        processed_meal_df = raw_meal_df.iloc[:, ::-1].dropna(how = 'all', thresh=2)
        processed_carb_data = pd.DataFrame(processed_meal_df.carbs)
        processed_meal_df = processed_meal_df.drop('carbs', 1)
        processed_meal_df.interpolate(method='linear', inplace=True)
        print('Processed data size - ', processed_meal_df.shape)
        print("Pre-processing ... DONE.")
        return processed_meal_df, processed_carb_data

    def extract_velocity(self, data_df, new_features):
        """
        Windowed velocity(non-overlapping)- 30 mins intervals
        """
        #print("Extracting Velocity ...")

        rows, cols = data_df.shape
        window_size = 5
        for i in range(0, cols - window_size):
            new_features['Vel_' + str(i)] = (data_df.iloc[:, i + window_size] - data_df.iloc[:, i])

        #print("Extracting Velocity ... DONE.")

    def extract_mean(self, data_df, new_features):
        """
        # Windowed mean interval - 30 mins(non-overlapping)
        """
        #print("Extracting Mean ...")

        rows, cols = data_df.shape
        window_size = 5
        for i in range(0, cols, window_size):
            new_features['Mean_' + str(i)] = data_df.iloc[:, i:i + window_size].mean(axis = 1)

        #print("Extracting Mean ... DONE.")

    def extract_fft(self, data_df, new_features):
        """
        FFT- Finding top 8 values for each row
        """
        print("Extracting FFT ...")

        def get_fft(row):
            cgmFFTValues = abs(fftpack.fft(row))
            cgmFFTValues.sort()
            return np.flip(cgmFFTValues)[0:8]

        FFT = pd.DataFrame()
        FFT['FFT_Top2'] = data_df.apply(lambda row: get_fft(row), axis = 1)
        FFT_updated = pd.DataFrame(FFT.FFT_Top2.tolist(),
                                   columns = ['FFT_1', 'FFT_2', 'FFT_3', 'FFT_4', 'FFT_5', 'FFT_6', 'FFT_7', 'FFT_8'])

        print("Extracting FFT ... DONE.")
        return new_features.join(FFT_updated)

    # def extract_entropy(self, data_df, new_features):
    # """
    # Calculates entropy(from occurences of each value) of given series
    # """
    #     print("Extracting Entropy ...")
    #
    #     def get_entropy(series):
    #         series_counts = series.value_counts()
    #         entropy = scipy.stats.entropy(series_counts)
    #         return entropy
    #
    #     new_features['Entropy'] = data_df.apply(lambda row: get_entropy(row), axis = 1)
    #     print("Extracting Entropy ... DONE.")
    #     # Plotting
    #     # plt.plot(new_features['Entropy'], 'r-')
    #     # plt.ylabel('Entropy')
    #     # plt.xlabel('Days')
    #     # plt.savefig('Entropy.png')

    def extract_polyfit(self, data_df, new_features):
        """
        Calculates polynomial fit coeffs for the data points
        """
        print("Extracting polynomial fit coeffs ...")

        poly = pd.DataFrame()
        rows, cols = data_df.shape
        poly_degree = 5
        poly['PolyFit'] = data_df.apply(lambda row: np.polyfit(range(cols), row, poly_degree), axis = 1)
        poly_df_cols = []
        for i in range(poly_degree + 1):
            poly_df_cols.append('poly_fit' + str(i + 1))
        poly_updated = pd.DataFrame(poly.PolyFit.tolist(),
                                    columns = poly_df_cols)

        print("Extracting polynomial fit coeffs ... DONE.")
        if new_features.empty:
            return poly_updated
        return new_features.join(poly_updated)

    # def extract_clusters(self, new_features):
    #     """
    #     Forms 2 clusters in input data and adds it as feature
    #     """
    #     print("Extracting clusters ...")
    #
    #     km = TimeSeriesKMeans(n_clusters=2, random_state=42)
    #     km.fit(new_features)
    #     y_label = km.labels_
    #     new_features['km_clusters'] = y_label
    #
    #     print("Extracting clusters ... DONE.")

    # def extract_noise(self, data_df, new_features):
    # """
    # uses noise as feature(noise = 1, no noise = 0)
    # """
    #     dip_window = 6
    #     raise_window = 20
    #     max_cgm = 250
    #     min_cgm = 150
    #     new_features['noise'] = 1
    #     noises = data_df[data_df.iloc[:, dip_window] > max_cgm].index
    #     # Marks the noise as no-meal
    #     new_features.loc[noises, 'noise'] = 0
    #     noises = data_df[data_df.iloc[:, raise_window] < min_cgm].index
    #     new_features.loc[noises, 'noise'] = 0

    # def extract_max_min(self, data_df, new_featues):
    # """
    # Finds max - min for each time-series instance
    # """
    #     print("Extracting max - min ...")
    #
    #     new_featues['max_min'] = data_df.apply(lambda row: max(row) - row[0], axis = 1)
    #
    #     print("Extracting max - min ... DONE.")
    #     return new_featues

    def extract_features(self, data_df):
        """
        Extracts 4 features from time series data
        1. Maximum window velocity
        2. Windowed mean
        3. Polynomial fit
        4. K-means clustering
        :param data_df:
        :return:
        """
        print('Extracting Features ...')
        # Feature Matrix
        feature_df = pd.DataFrame()
        # FEATURE 1 -> Windowed velocity(non-overlapping)- 30 mins intervals
        self.extract_velocity(data_df, feature_df)
        # FEATURE 2 -> Windowed mean interval - 30 mins(non-overlapping)
        self.extract_mean(data_df, feature_df)
        # FEATURE 3 -> FFT- Finding top 8 values for each row
        #feature_df = self.extract_fft(data_df, feature_df)
        # FEATURE 4 -> Calculates entropy(from occurrences of each value) of given series
        # self.extract_entropy(data_df, feature_df)
        # FEATURE 5 -> Calculates polynomial fit coefficients of given series
        # feature_df = self.extract_polyfit(data_df, feature_df)
        # FEATURE 6 -> KMeans Clustering(n = 2)
        # self.extract_clusters(feature_df)
        # FEATURE 7 -> Calculate if noise present
        # self.extract_noise(data_df, feature_df)
        # FEATURE 8 -> Calculates max - first value
        # feature_df = self.extract_max_min(data_df, feature_df)

        #feature_df['std'] = data_df.std(axis=1)
        #feature_df['skew'] = data_df.skew(axis=1)
        #feature_df['Max'] = data_df.max(axis=1)
        #feature_df['Min'] = data_df.min(axis=1)
        #feature_df['Var'] = data_df.var(axis=1)
        # print("Feature size - ", feature_df.shape)
        #feature_df.to_csv("output.csv")
        print('Extracting Features ... DONE.')
        return feature_df

    def h_clustering(self, new_features):
        """
        Forms 10 clusters with input features to compare with carb clusters
        """
        print("Hierarchical Clustering ...")

        cluster = AgglomerativeClustering(n_clusters = 10, affinity='euclidean', linkage='ward')
        cluster.fit(new_features)
        y_label = cluster.labels_
        h_clusters_df = pd.DataFrame(y_label)
        new_features['h_clusters'] = y_label
        print("Hierarchical Clustering ... DONE.")
        return h_clusters_df

    def km_clustering(self, data, file_path):
        """
        Forms 10 clusters out of hierarchical clusters
        """
        print("\nK-Means Clustering ...")
        km = KMeans(n_clusters=10)
        km.fit(data)
        y_label = km.labels_
        print('SSE Score - ', km.inertia_)
        print('K-Means Cluster Labels -')
        print(*y_label, sep = ',')
        np.savetxt(file_path, y_label, delimiter = ",", fmt = '%1.0f')
        print('K-Means cluster labels saved in file - \n', file_path)
        print("K-Means Clustering ... DONE.")
        return y_label
    
    def dbscan_clustering(self, data, file_path):
        print("\nDBSCAN Clustering.....")
        data = StandardScaler().fit_transform(data)
        db = DBSCAN(eps = 0.3, min_samples = 2).fit(data)
        y_label = db.labels_
        #print(f"Silhouette score : {sil_score}")
        print('DBSCAN Cluster Labels -')
        print(*y_label, sep = ',')
        np.savetxt(file_path, y_label, delimiter = ",", fmt = '%1.0f')
        print('DBSCAN cluster labels saved in file - \n', file_path)
        print("DBSCAN Clustering ... DONE.")
        return y_label

    # PCA - NOT USED
    def reduce_dimensions(self, feature_df):
        print("Dimensionality reduction ...")

        # Standardizes feature matrix
        feature_df = StandardScaler().fit_transform(feature_df)
        pca_k = 9
        pca = PCA(n_components = pca_k)
        principal_components_trans = pca.fit_transform(feature_df)
        pca_df_cols = []
        for i in range(pca_k):
            pca_df_cols.append('principal components ' + str(i + 1))
        pca_df = pd.DataFrame(data = principal_components_trans, columns = pca_df_cols)

        # print(principal_components.components_)  # Principal Components vs Original Features
        print(pca.explained_variance_ratio_.cumsum())
        print("PCA dataframe - \n", pca_df.head())
        print("Dimensionality reduction ... DONE.")
        return pca_df

    def carbs_cluster(self, data):
        # print(data)

        max_val = data['carbs'].max()
        min_val = data['carbs'].min()
        range_val = (max_val - min_val) / 10
        carbs_labels = []
        for i in range(len(data)):
            if data.iloc[i]['carbs'] == max_val:
                carbs_labels.append(9)
            else:
                carbs_labels.append(int((data.iloc[i]['carbs'] - min_val) // range_val))

        return carbs_labels

    def map_feature_labels(self, feature_label, carbs_label):
        carb_to_feature = defaultdict(partial(np.array, 0))
        for i in range(0, len(carbs_label)):
            carb_to_feature[carbs_label[i]] = np.append(carb_to_feature[carbs_label[i]], feature_label[i])
        final_data = defaultdict()
        # for key in carb_to_feature:
        #     counts = Counter(carb_to_feature[key])
        #     final_data[key] = counts.most_common(1)[0][0]
        # print(final_data)
        # TODO: Problem here.. since some clusters are not even there in the final mapped data
        # for i in range(len(feature_label)):
        #     feature_label[i] = list(final_data.keys())[list(final_data.values()).index(feature_label[i])]

        return feature_label

    def cluster_validation(self, labels_pred, labels_true):
        print('Validating clusters ...')
        # cluster_score = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
        # cluster_score = stats.entropy(labels_pred, labels_true)
        cluster_score = sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
        print(f"ADJUSTED RAND CLUSTER SCORE:{cluster_score}")
        print('Validating clusters ... DONE.')

    def run_model(self):
        """
        controller function to run all tasks
        :return:
        """
        input_path = self.read_path()
        raw_meal_df, raw_carb_df = self.read_data(input_path)
        processed_df, processed_carb_df = self.preprocess_data(raw_meal_df, raw_carb_df)
        feature_df = self.extract_features(processed_df)
        #reduced_feature_df = self.reduce_dimensions(feature_df)

        # K-MEANS
        h_clusters_df = self.h_clustering(feature_df)
        feature_labels = self.km_clustering(h_clusters_df, self.OUTPUT_PATH + "kmeans_labels.csv")
        carb_labels = self.carbs_cluster(processed_carb_df)
        # feature_labels = self.map_feature_labels(feature_labels, carb_labels)
        # print(f"Feature Labels: {feature_labels}\n Carb Labels:{carb_labels}")
        #self.cluster_validation(feature_labels, carb_labels)

        # DBSCAN
        dbscan_labels = self.dbscan_clustering(h_clusters_df, self.OUTPUT_PATH + "dbscan_labels.csv")
        self.cluster_validation(dbscan_labels, carb_labels)


if __name__ == '__main__':
    meal_cluster = MealClustering()
    meal_cluster.run_model()
