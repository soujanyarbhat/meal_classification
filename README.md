# meal_classification
Given sugar values over a 150 minutes time period, classify each instance as "meal" or "no meal".

TODO:
- Handle NaN values
- Change in features(if reqd.)
- Classifiers- check accuracy & remove warning messages

EXECUTION INSTRUCTIONS:

- python MealClassifier.py
- --enter-- for default input path

ABSTRACT:

The aim of the project is to study the glucose levels in diabetic patients during the course of a potential meal and design a model to detect whether a meal was actually taken in the recorded data. The data is collected using a CGM(continuous glucose monitoring) device recording sugar values. 
The time period defined by the problem statement was 30 minutes before the commencement of a meal to 120 minutes after, thereby a total of 150 minutes for each patient’s recorded meal data taken every 5 minutes. The glucose levels were expected to fluctuate depending upon the carbohydrates provided by the meal and the impact of insulin in the body.

FEATURE EXTRACTION DATA SET:
The data for each category is made up of 2 CSVs for each patient, wherein one file contains the recorded values while the other contains the timestamps of the corresponding glucose levels. Approximately, each patient has 30 recorded lunch meals distributed over time. The CGM glucose levels were integer values while the time series were real numbers. All the values in the CSV files was in inverse order of time for each row(e.g. The first recorded glucose value during a meal was present in the last index).

CLASSIFICATION DATA SET:
Data is available in terms of 2 CSVs per patient. One contains legitimiate meal data while the other contains data belonging to class label "no meal". This data was utilised for training the classification model based upon the features extracted previously.

FEATURES:
1. Maximum window velocity
2. Windowed mean
3. Fast fourier transform
4. Entropy

