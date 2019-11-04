# TASKS:
# 1. Read data from CSV
# 2. Data pre-processing
# 3. Feature extraction
# 4. Dimensionality reduction
# 5. Training(Classification)
# 6. Testing(Classification)
# 7. Result analysis

# import packages
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from pandas import read_csv
from pandas import concat
from pandas import DataFrame
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np

class MealClassifier:

    def __init__(self):
        print("Meal Classification Model ...")

    # Input path for data set- USER INPUT
    def read_path(self):
        print("Reading input directory ... ")

        current_dir = os.path.dirname(__file__)
        folder_path = os.path.join(current_dir, '..' + os.sep, 'Input')
        pre_configured_path = os.path.abspath(folder_path)
        temp_data_set_folder = input('Please enter the data set directory path\n'
                                     'OR\nHit enter to use %s: ' % pre_configured_path)
        if temp_data_set_folder.strip() != '':
            folder_path = temp_data_set_folder.strip()

        print("Reading input directory ... DONE.")

        return folder_path

    # Reads meal data and no meal data separately
    def read_data(self, input_path):
        print("Reading data ...")

        # Handles issues with ragged CSV file
        col_names = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10",
                     "Col11", "Col12", "Col13", "Col14", "Col15", "Col16","Col17", "Col18", "Col19", "Col20",
                     "Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30", "Col31"]
        # Lists all files in input directory
        input_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
        raw_meal_df = pd.DataFrame()
        raw_nomeal_df = pd.DataFrame()
        for input_file in input_files:
            input_df = pd.read_csv(os.path.join(input_path, input_file), names = col_names)
            if "Nomeal" in input_file:
                raw_nomeal_df = pd.concat([raw_nomeal_df, input_df])
            else:
                raw_meal_df = pd.concat([raw_meal_df, input_df])

        print("Meal data size -\n", raw_meal_df.shape)
        print("No Meal data size -\n", raw_nomeal_df.shape)
        print("Meal data -\n", raw_meal_df.head())
        print("No Meal data -\n", raw_nomeal_df.head())

        print("Reading data ... DONE.")

        return raw_meal_df, raw_nomeal_df

    # Reverses columns
    # Adds label to meal and no meal data
    # Interpolates NaN values
    # Shuffles rows
    # Removes rows- all NaN or high NaN count
    def process_data(self, raw_meal_df, raw_nomeal_df):
        print("Pre-processing ...")

        # TODO: Handle NaN - CURRENT: delete col 31

        del raw_meal_df['Col31']
        del raw_nomeal_df['Col31']

        processed_meal_df = raw_meal_df.iloc[:, ::-1].dropna(how = 'all')
        processed_nomeal_df = raw_nomeal_df.iloc[:, ::-1].dropna(how = 'all')
        processed_meal_df.loc[:, 'meal'] = 1
        processed_nomeal_df.loc[:, 'meal'] = 0
        concat_df = pd.concat([processed_meal_df, processed_nomeal_df])
        meal_df = concat_df.drop('meal', 1)
        meal_df.interpolate(method='linear', inplace=True)
        meal_df = pd.concat([meal_df, concat_df['meal']], axis = 1)
        meal_df = meal_df.sample(frac = 1)
        meal_df = meal_df.reset_index(drop=True)

        print("Processed data size- ", meal_df.shape)
        print("Processed data -\n", meal_df.head())
        print("Pre-processing ... DONE.")

        return meal_df

    # Windowed velocity(non-overlapping)- 30 mins intervals
    def extract_velocity(self, data_df, new_features):
        print("Extracting Velocity ...")

        rows, cols = data_df.shape
        window_size = 5
        velocity_df = pd.DataFrame()
        for i in range(0, cols - window_size):
            velocity_df['Vel_' + str(i)] = (data_df.iloc[:, i + window_size] - data_df.iloc[:, i])
        new_features['Window_Velocity_Max'] = velocity_df.max(axis = 1)

        print("Extracting Velocity ... DONE.")
        # Plotting
        # plt.plot(new_features['Window_Velocity_Max'], 'r-')
        # plt.ylabel('Window_Velocity_Max')
        # plt.xlabel('Days')
        # plt.savefig('Velocity.png')

    # Windowed mean interval - 30 mins(non-overlapping)
    def extract_mean(self, data_df, new_features):
        print("Extracting Mean ...")

        rows, cols = data_df.shape
        window_size = 5
        for i in range(0, cols, window_size):
            new_features['Mean_' + str(i)] = data_df.iloc[:, i:i + window_size].mean(axis = 1)

        print("Extracting Mean ... DONE.")
        # Plotting
        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(1, 1, 1)
        # ax.set_ylabel('Mean')
        # ax.set_xlabel('Days')
        # ax.set_title('Windowed Means')
        # ax.plot(new_features.iloc[:, 1:7], '-')
        # ax.legend(('Mean_0', 'Mean_6', 'Mean_12', 'Mean_18', 'Mean_24', 'Mean_30'), loc='upper right')
        # fig.savefig('WindowedMean.png')

    # FFT- Finding top 8 values for each row
    def extract_fft(self, data_df, new_features):
        print("Extracting FFT ...")

        def get_fft(row):
            cgmFFTValues = abs(scipy.fftpack.fft(row))
            cgmFFTValues.sort()
            return np.flip(cgmFFTValues)[0:8]

        FFT = pd.DataFrame()
        FFT['FFT_Top2'] = data_df.apply(lambda row: get_fft(row), axis = 1)
        FFT_updated = pd.DataFrame(FFT.FFT_Top2.tolist(),
                                   columns = ['FFT_1', 'FFT_2', 'FFT_3', 'FFT_4', 'FFT_5', 'FFT_6', 'FFT_7', 'FFT_8'])

        print("Extracting FFT ... DONE.")
        return new_features.join(FFT_updated)

    def extract_entropy(self, data_df, new_features):
        print("Extracting Entropy ...")

        # Calculates entropy(from occurences of each value) of given series
        def get_entropy(series):
            series_counts = series.value_counts()
            entropy = scipy.stats.entropy(series_counts)
            return entropy

        new_features['Entropy'] = data_df.apply(lambda row: get_entropy(row), axis = 1)
        print("Extracting Entropy ... DONE.")
        # Plotting
        # plt.plot(new_features['Entropy'], 'r-')
        # plt.ylabel('Entropy')
        # plt.xlabel('Days')
        # plt.savefig('Entropy.png')

    # Extracts 4 features from time series data
    # 1. Maximum window velocity
    # 2. Windowed mean
    # 3. Entropy
    # 4. FFT
    def extract_features(self, data_df):

        # Feature Matrix
        feature_df = pd.DataFrame()
        # FEATURE 1 -> Windowed velocity(non-overlapping)- 30 mins intervals
        self.extract_velocity(data_df, feature_df)
        # FEATURE 2 -> Windowed mean interval - 30 mins(non-overlapping)
        self.extract_mean(data_df, feature_df)
        # FEATURE 3 -> FFT- Finding top 8 values for each row
        feature_df = self.extract_fft(data_df, feature_df)
        # FEATURE 4 -> Calculates entropy(from occurrences of each value) of given series
        self.extract_entropy(data_df, feature_df)

        print("Feature size - ", feature_df.shape)
        print("Features - \n", feature_df.head())
        return feature_df

    # PCA
    def reduce_dimensions(self, feature_df):
        print("Dimensionality reduction ...")

        # Standardizes feature matrix
        feature_df = StandardScaler().fit_transform(feature_df)

        pca = PCA(n_components = 5)
        principal_components = pca.fit(feature_df)
        principal_components_trans = pca.fit_transform(feature_df)
        pc_time_series = pd.DataFrame(data = principal_components_trans,
                                      columns = ['principal component 1', 'principal component 2',
                                                 'principal component 3',
                                                 'principal component 4', 'principal component 5'])

        # print(principal_components.components_)  # Principal Components vs Original Features
        print(principal_components.explained_variance_ratio_.cumsum())
        print("Dimensionality reduction ... DONE.")
        return pc_time_series
        # plotting explained variance versus principle componenets
        # pcs = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
        # plt.bar(pcs, principal_components.explained_variance_ratio_ * 100)
        # plt.savefig('variance.png')

    def run_model(self):

        input_path = self.read_path()
        raw_meal_df, raw_nomeal_df = self.read_data(input_path)
        processed_df = self.process_data(raw_meal_df, raw_nomeal_df)
        processed_unlabelled_df = processed_df.drop('meal', 1)
        feature_df = self.extract_features(processed_unlabelled_df)
        reduced_feature_df = self.reduce_dimensions(feature_df)

        # TODO
        reduced_feature_df = pd.concat([reduced_feature_df, processed_df['meal']], axis = 1)
        print(reduced_feature_df.head())
        X_train, X_test, y_train, y_test = train_test_split(reduced_feature_df, processed_df.meal)
        # classifiers
        classifiers = {
            "LogisticRegression": LogisticRegression(),
            "KNearest": KNeighborsClassifier(),
            "Support Vector Classifier": SVC(),
            "DecisionTreeClassifier": DecisionTreeClassifier()
        }
        for key, classifier in classifiers.items():
            classifier.fit(X_train, y_train)
            training_score = cross_val_score(classifier, X_train, y_train)
            print("Classifiers: ", classifier.__class__.__name__, "Has a training score of",
                  round(training_score.mean(), 2) * 100, "% accuracy score")




        print("Meal Classification Model ... DONE.")


if __name__ == '__main__':
    meal_classifier = MealClassifier()
    meal_classifier.run_model()

# ASSUMPTIONS-
# 1. Data in reverse order in each row wrt time
# 2. "Nomeal" in filename -> No meal data
# 3. NaN handling

# TASKS-
# a) Extract features from Meal and No Meal data
# b) Make sure that the features are discriminatory
# c) Each student trains a separate machine to recognize Meal or No Meal data
# d) Use k fold cross validation on the training data to evaluate your recognition system
# e) Each student write a function that takes one test sample as input
# and outputs 1 if it predicts the test sample as meal or 0 if it predicts test sample as No meal.
