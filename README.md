
**MEAL DETECTION PROJECT**
 
**Introduction**

The aim of the project is to study the glucose levels in diabetic patients during the course of a potential meal and design a model to detect whether a meal was actually taken in the recorded data. The data is collected using a CGM(continuous glucose monitoring) device along with records of bolus and basal values. For this phase/assignment, only the CGM recorded glucose levels were considered along with the time series.
The time period defined by the problem statement was 30 minutes before the commencement of a meal to 120 minutes after, thereby a total of 150 minutes for each patient’s recorded meal data taken every 5 minutes.

**Data Pre-processing**

Prior to feature extraction, the original data was processed to improve the results and understanding:
1. The column ordering was reversed for both CGM and time-series data.
2. The row ordering was reversed for both CGM and time-series data.
3. The timestamp values were converted to the format - “YYYY-mm-dd HH:MM:ss.SSSSSS”
4. The missing CGM values were linearly interpolated.
5. The missing time-series values were computed assuming each meal span was 2.5 hours with 5 minutes intervals.

!.[Sample Data](https://imgur.com/4rr8WcO)

!.[Sample Data](https://imgur.com/9QaA6BQ)
    
**Intuition**
	
The primary idea on choosing the features was to learn the common shape of given multiple time-series data. Considering a time series classification problem, the group deduced that it is imperative to pick features of the varying glucose levels that define the shape over time. Thereby, the template can be a decent metric to identify if the given test time-series is similar to the learnt model depending upon the feature values. For this purpose, different features were chosen that either smoothed the curve to provide a range of appropriate values or keep track of a high rise in glucose levels possibly due to food intake. Certain features were chosen to reduce noise impact or to define the data complexity.
	
Ideally, for the 2.5 hours meal window the glucose levels would take a normal value for the first 30 mins, then experience a sharp rise due to food intake, settle at the peak and fall due to the insulin keeping the glucose levels in check. It is this pattern that we wish our model to learn and use to classify the “meal” and “non-meal” class labels.




**Feature Extraction**
The following features were chosen to contain high discriminative features for the given time-series data:

Maximum Windowed Velocity

Windowed Averages

Fast Fourier Transformation

Entropy


Final Feature Matrix

Since patient 1 contained 33 days lunch data, our final feature matrix(as input to PCA) contains 33X16 where 16 is the number of features.



**Principal Component Analysis(PCA)**

 	PCA finds a new set of dimensions (or set of basic views) such that all the dimensions are orthogonal (and hence linearly independent) and ranked according to the variance of data along with them. It means more important principle axis comes first. 

This defines the goals of PCA - 
Find linearly independent dimensions (or basis of views) which can losslessly represent the data points.
Those newly found dimensions should allow us to predict/reconstruct the original dimensions. The reconstruction/projection error should be minimized.
PCA returns the following:
Components - a 16 x 16 matrix, representing the Eigenvectors and the columns are in decreasing order of their variance.
Explained Variance - The percentage of variance depicted by each Principal Component.
Selecting the first 5 Principal Components would represent 92.00% of the information with the first principal component contributing upto 52.45% of the information as shown in the Variance graph below. This depicts that 92% of the original data can be represented using only 5 features instead of 16, thereby PCA is optimally used for dimensionality reduction.


The higher value of the weight or component signifies that most of the data lies along that feature component of the eigenvector. 
Example - In principal component 1, the columns corresponding to FFT features have higher values which means that these features contribute maximum to the data along this principal component.

Since the selected features contribute to the different principal components in majority at different instances, the selection criteria can be termed successful.



The below plots show how each Principal component vector represents data when data is projected along these vectors individually.



**References**

[1] Ben D. Fulcher, (2012) “Feature-based time-series analysis” (pg. 10)

[2] Fiorini, Samuel & Martini, Chiara & Malpassi, Davide & Cordera, Renzo & Maggi, Davide & Verri, Alessandro & Barla, Annalisa. (2017). 
“Data-driven strategies for robust forecast of continuous glucose monitoring time-series.” Conference proceedings: ... Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE Engineering in Medicine and Biology Society. Conference. 2017. 1680-1683. 10.1109/EMBC.2017.8037164. 
[3] Chen, Xiao-yun et al. “Entropy-Based Symbolic Representation for Time Series Classification.” Fourth International Conference on Fuzzy Systems and Knowledge Discovery (FSKD 2007) 2 (2007): 754-760.
