import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from pandas import read_csv
from pandas import concat
from pandas import DataFrame
from scipy.interpolate import UnivariateSpline
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
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from mealclassifier import MealClassifier

class TestMealClassifier():

    OUTPUT_PATH_DIR = os.path.join(os.path.dirname(__file__), '..', 'Output')
    OUTPUT_FILE_NAMES = {
        'LogisticRegression_varun.sav': 'predictions_varun.csv',
        'GaussianNB_soujanya.sav': 'predictions_soujanya.csv',
        'SVC_aryan.sav': 'predictions_aryan.csv',
        'RandomForestClassifier_gourav.sav': 'predictions_gourav.csv'
    }

    def __init__(self):
        print("Testing Meal Classifier ...")

    # Converts test data to reduced feature matrix after pre-processing
    def process_data(self, file_path, meal_classifier):
        print('Processing test data ...')

        # Handles issues with ragged CSV file
        col_names = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10",
                     "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20",
                     "Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30", "Col31"]
        test_data_df = pd.read_csv(file_path, names = col_names)
        del test_data_df['Col31']
        test_data_df = test_data_df.iloc[:, ::-1].astype(int)
        feature_df = meal_classifier.extract_features(test_data_df)
        reduced_feature_df = meal_classifier.reduce_dimensions(feature_df)

        print('Processing test data ... DONE.')
        return reduced_feature_df

    # Tests all models and saves predictions in file
    def predict_save(self, reduced_feature_df, meal_classifier):
        print('Predicting test data ...')

        chosen_models = [f for f in listdir(meal_classifier.OUTPUT_PATH_MODEL)
                         if isfile(join(meal_classifier.OUTPUT_PATH_MODEL, f))]
        for chosen_model in chosen_models:
            model_path = os.path.join(meal_classifier.OUTPUT_PATH_MODEL, chosen_model)
            model = pickle.load(open(model_path, 'rb'))
            results = model.predict(reduced_feature_df)
            output_path = os.path.join(self.OUTPUT_PATH_DIR, self.OUTPUT_FILE_NAMES[chosen_model])
            with open(output_path, 'w') as fo:
                for result in results:
                    fo.write(str(result) + '\n')
            print('{} predictions stored in file - {}'.format(chosen_model, output_path))

        print('Predicting test data ... DONE.')

    # Imports saved model
    # Processes test data
    # For each model- Labels test data and saves to file
    def test_model(self):

        meal_classifier = MealClassifier()
        file_path = input('Please enter the data set directory path:\n')
        reduced_feature_df = self.process_data(file_path, meal_classifier)
        self.predict_save(reduced_feature_df, meal_classifier)

        print("Testing Meal Classifier ... DONE.")


if __name__ == '__main__':
    test_meal_classifier = TestMealClassifier()
    test_meal_classifier.test_model()