import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

def split_data(df, column):
    '''This function takes in two arguments, a dataframe and a string. The string argument is the name of the
        column that will be used to stratify the train_test_split. The function returns three dataframes, a 
        training dataframe with 60 percent of the data, a validate dataframe with 20 percent of the data and test
        dataframe with 20 percent of the data.'''
    # split data into train and test with a test size of 20 percent and random state of 217, stratify on target
    train, test = train_test_split(df, test_size=.2, random_state=217, stratify=df[column])
    # split train again into train and validate with a validate size of 25 percent of train, stratify on target
    train, validate = train_test_split(train, test_size=.25, random_state=217, stratify=train[column])
    # return three dataframes, 60/20/20 split
    return train, validate, test

def plot_day_night(train):
    '''This function takes in a dataframe and returns a figure of four subplots. Each subplot shows the relationship
        between the severity of an accident and whether the accident occurred during the day or night. Each feature
        represents a different angle of the sun from the horizon.'''
    # set the figure size for the entire visualization
    plt.figure(figsize=[20,6])
    # subplot 1 of 4
    plt.subplot(1,4,1)
    # show the number of accidents for each level of severity for both day and night
    sns.countplot(x=train.severity, hue=train.sunrise_sunset)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Sunrise Sunset Angle')
    plt.legend(['Day','Night'])
    # subplot 2 of 4
    plt.subplot(1,4,2)
    # show the number of accidents for each level of severity for both day and night
    sns.countplot(x=train.severity, hue=train.civil_twilight)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Civil Twilight Angle')
    plt.legend(['Day','Night'])
    # subplot 3 of 4
    plt.subplot(1,4,3)
    # show the number of accidents for each level of severity for both day and night
    sns.countplot(x=train.severity, hue=train.nautical_twilight)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Nautical Twilight Angle')
    plt.legend(['Day','Night'])
    # subplot 4 of 4
    plt.subplot(1,4,4)
    # show the number of accidents for each level of severity for both day and night
    sns.countplot(x=train.severity, hue=train.astronomical_twilight)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Astronomical Twilight Angle')
    plt.legend(['Day','Night'])
    plt.suptitle('Do more accidents occur during the day or at night?')

def barplot_data(train):
    '''This function takes in a dataframe and returns a visualization of 8 subplots. Each subplot shows the relationship
        between the severity of an accident and a continuous feature recorded at the time of the accident.'''
    # set the figure size for the visualization
    plt.figure(figsize=[20,9])
    # subplot 1 of 8
    plt.subplot(2,4,1)
    # show the relationship between accident severity and the affected distance
    sns.barplot(x='severity', y='distance', data=train)
    plt.xlabel('Severity')
    plt.ylabel('Distance in Miles')
    plt.title('Length of road affected')
    # subplot 2 of 8
    plt.subplot(2,4,2)
    # show the relationship between accident severity and the recorded temperature in Fahrenheit
    sns.barplot(x='severity', y='temperature', data=train)
    plt.xlabel('Severity')
    plt.ylabel('Temperature in Fahrenheit')
    plt.title('Temp during the accident')
    # subplot 3 of 8
    plt.subplot(2,4,3)
    # show the relationship between accident severity and the recorded wind chill
    sns.barplot(x='severity', y='wind_chill', data=train)
    plt.xlabel('Severity')
    plt.ylabel('Wind Chill')
    plt.title('Wind chill during the accident')
    # subplot 4 of 8
    plt.subplot(2,4,4)
     # show the relationship between accident severity and the level of humidity
    sns.barplot(x='severity', y='humidity', data=train)
    plt.xlabel('Severity')
    plt.ylabel('Humidity (Percentage)')
    plt.title('Humidity level during accident')
    # subplot 5 of 8
    plt.subplot(2,4,5)
     # show the relationship between accident severity and the atmospheric pressure
    sns.barplot(x='severity', y='pressure', data=train)
    plt.xlabel('Severity')
    plt.ylabel('Barometric Pressure')
    plt.title('Barometric pressure during accident')
    # subplot 6 of 8
    plt.subplot(2,4,6)
     # show the relationship between accident severity and the visibility in miles
    sns.barplot(x='severity', y='visibility', data=train)
    plt.xlabel('Severity')
    plt.ylabel('Visibility (Miles)')
    plt.title('Visibility during accident')
    # subplot 7 of 8
    plt.subplot(2,4,7)
     # show the relationship between accident severity and the amount of precipitation
    sns.barplot(x='severity', y='precipitation', data=train)
    plt.xlabel('Severity')
    plt.ylabel('Preciipitation (inches)')
    plt.title('Precipitation during accident')
    # subplot 8 of 8
    plt.subplot(2,4,8)
     # show the relationship between accident severity and the recorded wind speed
    sns.barplot(x='severity', y='wind_speed', data=train)
    plt.xlabel('Severity')
    plt.ylabel('Wind Speed (mph)')
    plt.title('Wind speed during accident')
    plt.suptitle('How do environmental conditions affect severity?')
    plt.show()

def countplot_data(train):
    '''This function takes in a dataframe and returns 12 graphs showing the relationships between accident severity and
        road features. Each graph has a countplot of accidents and whether or not the specific feature was present.'''
    # set the size of the overall figure
    plt.figure(figsize=[20,16])
    # subplot 1 of 12
    plt.subplot(4,3,1)
    # plot if traffic calming was present at accident location
    sns.countplot(x=train.severity, hue=train.traffic_calming)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Traffic Calming')
    plt.legend(labels=['No','Yes'])
    # subplot 2 of 12
    plt.subplot(4,3,2)
    sns.countplot(x=train.severity, hue=train.station)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Station')
    plt.legend(labels=['No','Yes'])
    # subplot 3 of 12
    plt.subplot(4,3,3)
    sns.countplot(x=train.severity, hue=train.roundabout)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Roundabout')
    plt.legend(labels=['No','Yes'])
    # subplot 4 of 12
    plt.subplot(4,3,4)
    sns.countplot(x=train.severity, hue=train.railway)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Railway')
    plt.legend(labels=['No','Yes'])
    # subplot 5 of 12
    plt.subplot(4,3,5)
    sns.countplot(x=train.severity, hue=train.no_exit)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('No Exit')
    plt.legend(labels=['No','Yes'])
    # subplot 6 of 12
    plt.subplot(4,3,6)
    sns.countplot(x=train.severity, hue=train.junction)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Junction')
    plt.legend(labels=['No','Yes'])
    # subplot 7 of 12
    plt.subplot(4,3,7)
    sns.countplot(x=train.severity, hue=train.give_way)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Give Way')
    plt.legend(labels=['No','Yes'])
    # subplot 8 of 12
    plt.subplot(4,3,8)
    sns.countplot(x=train.severity, hue=train.crossing)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Crossing')
    plt.legend(labels=['No','Yes'])
    # subplot 9 of 12
    plt.subplot(4,3,9)
    sns.countplot(x=train.severity, hue=train.bump)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Bump')
    plt.legend(labels=['No','Yes'])
    # subplot 10 of 12
    plt.subplot(4,3,10)
    sns.countplot(x=train.severity, hue=train.amenity)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Amenity')
    plt.legend(labels=['No','Yes'])
    # subplot 11 of 12
    plt.subplot(4,3,11)
    sns.countplot(x=train.severity, hue=train.stop)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Stop')
    plt.legend(labels=['No','Yes'])
    # subplot 12 of 12
    plt.subplot(4,3,12)
    sns.countplot(x=train.severity, hue=train.traffic_signal)
    plt.xlabel('Severity')
    plt.ylabel('Accidents')
    plt.title('Traffic Signal')
    plt.legend(labels=['No','Yes'])
    plt.suptitle('Which road features affect accident severity?')

def plot_time_data(df):
    '''This function takes in a dataframe and returns four charts, each showing the number of accidents during a particular
        time period. Hours of the day from 0 to 23, months of the year, days of the month, and the number of accidents each
        year are visualized.'''
    # set start time as the datetime index and sort the index
    df_time = df.set_index('start_time').sort_index()
    # reconvert the new dataframe's index to datetime
    df_time.index = pd.to_datetime(df_time.index)
    # set the figure size
    plt.figure(figsize=[20,10])
    # subplot 1 of 4
    plt.subplot(2,2,1)
    # plot the number of accidents that occured during each hour of the day
    sns.countplot(x=df_time.index.hour)
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Accidents')
    plt.title('Accidents occur more frequently during the late afternoon')
    # subplot 2 of 4
    plt.subplot(2,2,2)
    # plot the number of accidents that occurred each day of the month
    sns.countplot(x=df_time.index.day)
    plt.xlabel('Day of the Month')
    plt.ylabel('Number of Accidents')
    plt.title('There is no relationship between accidents and day of the month')
    # subplot 3 of 4
    plt.subplot(2,2,3)
    # plot the number of accidents that occurred each month
    sns.countplot(x=df_time.index.month_name())
    plt.xticks(rotation=90)
    plt.xlabel('Month')
    plt.ylabel('Number of Accidents')
    plt.title('Accidents occur more frequently in April through June')
    # subplot 4 of 4
    plt.subplot(2,2,4)
    # plot the number of accidents that occurred each year
    sns.countplot(x=df_time.index.year)
    plt.xlabel('Year')
    plt.ylabel('Number of Accidents')
    plt.title('There is an upward trend in the number of recorded accidents each year')
    plt.suptitle('The Relationship between Vehicle Accidents and Different Measurements of Time')
    plt.show()

def stat_levene(x,y):
    '''This function takes in two arguments and conducts a Levene's test of equal variance. The alpha is set to .05.
        The function returns a statement of whether to reject or fail to reject the null hypothesis of equal variance.'''
    # set alpha
    alpha = .05
    # assign test values to variables
    stat, p = stats.levene(x,y)
    # check for significance according to p-value
    if p < alpha:
        print('we can reject the null hypothesis and posit that variance is inequal')
    else:
        print('we fail to reject the null hypothesis that variance is equal')

def stat_kruskal(x,y,z):
    '''This function takes in three arguments and conducts a Kruskal-Wallis analysis of variance test, with an alpha of
        .05. The function returns a statement of whether to reject or fail to reject the null hypothesis of no significant
        difference.'''
    # set alpha
    alpha = .05
    # assign test values to variables
    stat, p = stats.kruskal(x,y,z)
    # check for significance based on p-value
    if p < alpha:
        print('we can reject the null hypothesis that there is no mean difference.')
    else:
        print('we fail to reject the null hypothesis that there is no mean difference.')
    print(f'H: {stat}, p-value: {p}')

def stat_chi2(x,y):
    '''This function takes in two arguments and conducts a Chi2 test for independence at an alpha of .05. The function
        returns a crosstab of the expected values and the observed values, and a statement of whether to reject or fail
        to reject the null hypothesis of independence.'''
    # set alpha
    alpha = .05
    # assign test values to a crosstab
    observed = pd.crosstab(x, y)
    # conduct the chi2 test and save the four outputs to variables
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    # output values
    print('Observed')
    print(observed.values)
    print('\nExpected')
    print(expected.astype(int))
    print('\n----')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p}')
    # check for significance based on p-value
    if p < alpha:
        print('We reject the null hypothesis of independence.')
    else:
        print('We fail to reject the null hypothesis of independence.')

