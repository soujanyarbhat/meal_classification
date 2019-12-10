# import packages
import os
from os import listdir
from os.path import isfile, join
import pandas as pd

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from mealClustering import MealClustering


class TestingMealClustering():

    OUTPUT_PATH_MODEL = os.path.join(os.path.dirname(__file__), '..', 'Model')

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
        for input_file in input_files:
            input_df = pd.read_csv(os.path.join(input_path, input_file), names=col_names)
            raw_meal_df = pd.concat([raw_meal_df, input_df])

        print("Meal data size - ", raw_meal_df.shape)

        print("Reading data ... DONE.")

        return raw_meal_df

    def preprocess_data(self, raw_meal_df):
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
        processed_meal_df = raw_meal_df.iloc[:, ::-1].dropna(how = 'all')
        processed_meal_df.interpolate(method='linear', inplace=True)
        print('Processed data size - ', processed_meal_df.shape)
        print("Pre-processing ... DONE.")
        return processed_meal_df

    def run_model(self):
        """
        controller function to run all tasks
        :return:
        """
        meal_obj = MealClustering()
        input_path = self.read_path()
        raw_meal_df = self.read_data(input_path)
        processed_df = self.preprocess_data(raw_meal_df)
        feature_df = meal_obj.extract_features(processed_df)
        h_clusters_df = meal_obj.h_clustering(feature_df)
        _ = meal_obj.km_clustering(h_clusters_df)


if __name__ == '__main__':
    meal_cluster = TestingMealClustering()
    meal_cluster.run_model()
