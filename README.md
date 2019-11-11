# meal_classification
Given sugar values over a 150 minutes time period, classify each instance as "meal"(= 1) or "no meal"(= 0).

ABSTRACT:
The aim of the project is to design a model to detect whether a meal was actually taken in the recorded data or not.

FEATURES:
1. Maximum window velocity
2. Windowed mean
3. Fast fourier transform
4. Entropy
5. Polynomial fit
6. Clustering

PROJECT STRUCTURE:
1. Code - contains the classifier script and the testing script.
2. Input - contains the given data set for training model.
3. Model - cobntains the saved classifiers of each member.
4. Output - contains the predictions of chosen models.

EXECUTION INSTRUCTIONS:

1. Classifier script<br>
python mealclassifier.py<br>
Input- path to training data set directory(or ENTER for default "Input/" dir)<br>
Output- classifiers in "Model/" directory(as pickle files).
Filename format- "model_<member name\>.sav"

2. Test script<br>
python test_mealclassifier.py<br>
Input- path to test file<br>
Output- predictions in "Output/" directory(as CSV files).
Filename format- "predictions_<member name\>.sav"
