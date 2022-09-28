# Codeup Individual Project

## US Accidents from 2016 to 2021

The purpose of this project is to analyze the selected dataset, answer questions regarding the data, and develop a machine learning model to predict the severity of an accident based on human and environmental circumstances. I obtained the dataset for this project from https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents?resource=download.

    - Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, 2019.

    - Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.

I am using this dataset for academic purposes only.

Initial Questions:

- What road conditions are most likely to result in an accident?
- What time of day are accidents most likely to occur? What time of year are accidents most likely to occur?
- Are there specific areas that are prone to crashes?

Questions regarding time:

- Have the number of accidents increased overall between 2016 and 2021?
- Has the severity of accidents changed between 2016 and 2021?

## Project Utility

Predicting accident severity based on environmental conditions and road features can be useful for first responders, drivers, and rideshare companies. Accurate predictions can help first responders gauge the amount of services and emergency aid needed based on the most commonly required responses for each level of severity. Drivers can get accurate updates on how long traffic will be delayed and if alternate routes are needed. Future utility includes providing warnings to drivers and first responders of potential accident locations and severity based on current environmental and road conditions.

## Data Dictionary

1. id: This is a unique identifier of the accident record. (string)
2. severity: Shows the severity of the accident, a number between 1 and 4, where 1 indicates the least impact on traffic (i.e., short delay as a result of the accident) and 4 indicates a significant impact on traffic (i.e., long delay). (int64)
3. start_time: Shows start time of the accident in local time zone. (datetime64)
4. end_time: Shows end time of the accident in local time zone. End time here refers to when the impact of accident on traffic flow was dismissed. (datetime64)
5. start_lat: Shows latitude in GPS coordinate of the start point. (float64)
6. start_lng: Shows longitude in GPS coordinate of the start point. (float64)
7. end_lat: Shows latitude in GPS coordinate of the end point. (float64)
8. end_lng: Shows longitude in GPS coordinate of the end point. (float64)
9. distance: The length of the road extent affected by the accident. (float64)
10. description: Shows natural language description of the accident. (string)
11. (dropped) number: Shows the street number in address field. (float64)
12. street: Shows the street name in address field. (string)
13. side: Shows the relative side of the street (Right/Left) in address field. (string)
14. city: Shows the city in address field. (string)
15. county: Shows the county in address field. (string)
16. state: Shows the state in address field. (string)
17. zipcode: Shows the zipcode in address field. (string)
18. (dropped) country: Shows the country in address field. (string)
19. timezone: Shows timezone based on the location of the accident (eastern, central, etc.). (string)
20. (dropped) airport_code: Denotes an airport-based weather station which is the closest one to location of the accident. (string)
21. weather_timestamp: Shows the time-stamp of weather observation record (in local time). (datetime)
22. temperature: Shows the temperature (in Fahrenheit). (float64)
23. wind_chill: Shows the wind chill (in Fahrenheit). (float64)
24. humidity: Shows the humidity (in percentage). (float64)
25. pressure: Shows the air pressure (in inches). (float64)
26. visibility: Shows visibility (in miles). (float64)
27. wind_direction: Shows wind direction. (string)
28. wind_speed: Shows wind speed (in miles per hour). (float64)
29. precipitation: Shows precipitation amount in inches, if there is any. (float64)
30. weather_condition: Shows the weather condition (rain, snow, thunderstorm, fog, etc.) (string)
31. amenity: A POI annotation which indicates presence of amenity in a nearby location. (int64)
32. bump: A POI annotation which indicates presence of speed bump or hump in a nearby location. (int64)
33. crossing: A POI annotation which indicates presence of crossing in a nearby location. (int64)
34. give_way: A POI annotation which indicates presence of give_way in a nearby location. (int64)
35. junction: A POI annotation which indicates presence of junction in a nearby location. (int64)
36. no_exit: A POI annotation which indicates presence of no_exit in a nearby location. (int64)
37. railway: A POI annotation which indicates presence of railway in a nearby location. (int64)
38. roundabout: A POI annotation which indicates presence of roundabout in a nearby location. (int64)
39. station: A POI annotation which indicates presence of station in a nearby location. (int64)
40. stop: A POI annotation which indicates presence of stop in a nearby location. (int64)
41. traffic_salming: A POI annotation which indicates presence of traffic_calming in a nearby location. (int64)
42. traffic_signal: A POI annotation which indicates presence of traffic_signal in a nearby loction. (int64)
43. (dropped) turning_loop: A POI annotation which indicates presence of turning_loop in a nearby location. (int64)
44. sunrise_sunset: Shows the period of day (i.e. day or night) based on sunrise/sunset. (int64)
45. civil_twilight: Shows the period of day (i.e. day or night) based on civil twilight. (int64)
46. nautical_twilight: Shows the period of day (i.e. day or night) based on nautical twilight. (int64)
47. astronomical_twilight: Shows the period of day (i.e. day or night) based on astronomical twilight. (int64)


## Project Planning

- Acquire the dataset from Kaggle and save to a local csv
- Prepare the data with the intent to discover the main predictors of crash severity; clean the data and encode categorical features if necessary; ensure that the data is tidy
- Split the data into train, validate, and test datasets using a 60/20/20 split and a random seed of 217
- Explore the data:
    - Univariate, bivariate, and multivariate analyses; statistical tests for significance, find the three primary features affecting crash severity; use distance, precipitation, and visibility for the first model
- Create graphical representations of the analyses
- Ask more questions about the data
- Document findings
- Train and test models:
    - Establish a baseline using the mode for severity
    - Select key features and train multiple classification models
    - Test the model on the validate set, adjust for overfitting if necessary
- Select the best model for the project goals:
    - Determine which model performs best on the validate set
- Test and evaluate the model:
    - Use the model on the test set and evaluate its performance (accuracy, precision, recall, f1, etc.)
- Visualize the model's performance on the test set
- Document key findings and takeaways, answer the questions
- Create a final report

## How to Reproduce this Project

- In order to reproduce this project, you will need access to the Kaggle datasets or the .csv of the dataset. Acquire the database from https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents, which I saved to a csv. The wrangle.py file has the necessary functions to acquire, prepare, and split the dataset.

- You will need to import the following python libraries into a python file or jupyter notebook:

    - import pandas as pd
    - import numpy as np
    - import acquire
    - import prepare
    - import matplotlib.pyplot as plt
    - import seaborn as sns
    - from scipy import stats
    - from sklearn.tree import DecisionTreeClassifier, plot_tree
    - from sklearn.metrics import classification_report
    - from sklearn.metrics import confusion_matrix
    - from sklearn.ensemble import RandomForestClassifier
    - from sklearn.neighbors import KNeighborsClassifier
    - from sklearn.linear_model import LogisticRegression

- Prepare and split the dataset. The code for these steps can be found in the wrangle.py file within this repository.

- Use pandas to explore the dataframe and scipy.stats to conduct statistical testing on the selected features.

- Use seaborn or matplotlib.pyplot to create graphs of your analyses.

- Conduct a univariate analysis on each feature using barplot for categorical variables and .hist for continuous variables.

- Conduct a bivariate analysis of each feature against churn and graph each finding.

- Conduct multivariate analyses of the most important features against severity and graph the results.

- Create models (decision tree, random forest, KNearest neighbors, and logistical regression) with the most important selected features using sklearn.

- Train each model and evaluate its accuracy on both the train and validate sets.

- Select the best performing model and use it on the test set.

- Graph the results of the test using probabilities.

- Document each step of the process and your findings.


## Key Findings and Takeaways

- The target variable caused the dataset to be unbalanced, as most accidents were classified as severity level 2. This resulted in a baseline accuracy using the mode to be 93 percent. In order to balance the dataset, I took a random sample of 65,000 level-2 severity accidents from the total dataset using random_seed=217 for reproducibility. I concatenated this sample with the total observations from the other severity classes into a new dataframe of 215,240 observations. This sampling did not take into account any features, so important data about key features of a crash may have been lost.

- The minimum viable product model is a decision tree classifier with a maximum depth of 4. I selected three features for the initial model: distance, precipitation, and visibility. I selected these features based on visualizations and statistical tests. I used a random_seed of 217 for reproducibility. The baseline prediction for the training set was .302. The model performed above baseline accuracy at .42 on train and .41 on validate, indicating that the decision tree was not overfit. 