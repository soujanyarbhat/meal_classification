# meal clustering

GROUP (24) MEMBERS:
1. Varun Chaudhary
2. Soujanya R Bhat
3. Aryan Gupta
4. Gourav Agarwal

Given sugar values over a 150 minutes time period.
Carbohydrate file containing the carb levels in each meal taken.

ABSTRACT:
The aim of the project is to design a model to cluster meal data using K-Means and DBSCAN methods.

IMPLEMENTATION:
Hierarchical clustering was used to handle the initial centroid issue with K-Means clustering.

PROJECT STRUCTURE:
1. Code - contains the clustering script and the testing script.
2. Input - contains the given data set for clustering.
3. Output - contains K-Means and DBSCAN cluster labels for files in input folder.
4. Output_Test - contains K-Means and DBSCAN cluster labels for test script file input.

EXECUTION INSTRUCTIONS:

Using the terminal, navigate inside Code/ folder.

1. Clustering script<br>
python mealClustering.py<br>
Input- path to training data set directory(or ENTER for default "Input/" dir)<br>
Output(terminal)- 
a. Clustering labels for K-Means and DBSCAN. 
b. Location of files saved with cluster labels.
c. SSE Score(K-Means) and Adjusted Rand Score(DBSCAN)
Output (file)-
Cluster Labels corresponding to each row 
a. Output/kmeans_labels.csv
b. Output/dbscan_labels.csv

2. Test script<br>
python testing_mealClustering.py<br>
Input- path to test file<br>
Output(terminal)- 
a. Clustering labels for K-Means and DBSCAN. 
b. Location of files saved with cluster labels.
Output (file)-
Cluster Labels corresponding to each row 
a. Output_Test/kmeans_labels.csv
b. Output_Test/dbscan_labels.csv