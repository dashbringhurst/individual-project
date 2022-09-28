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
    '''This function'''
    plt.figure(figsize=[20,6])
    plt.subplot(1,4,1)
    sns.countplot(x=train.severity, hue=train.sunrise_sunset)
    plt.subplot(1,4,2)
    sns.countplot(x=train.severity, hue=train.civil_twilight)
    plt.subplot(1,4,3)
    sns.countplot(x=train.severity, hue=train.nautical_twilight)
    plt.subplot(1,4,4)
    sns.countplot(x=train.severity, hue=train.astronomical_twilight)

def barplot_data(train):
    plt.figure(figsize=[20,9])
    plt.subplot(2,4,1)
    sns.barplot(x='severity', y='distance', data=train)
    plt.subplot(2,4,2)
    sns.barplot(x='severity', y='temperature', data=train)
    plt.subplot(2,4,3)
    sns.barplot(x='severity', y='wind_chill', data=train)
    plt.subplot(2,4,4)
    sns.barplot(x='severity', y='humidity', data=train)
    plt.subplot(2,4,5)
    sns.barplot(x='severity', y='pressure', data=train)
    plt.subplot(2,4,6)
    sns.barplot(x='severity', y='visibility', data=train)
    plt.subplot(2,4,7)
    sns.barplot(x='severity', y='precipitation', data=train)
    plt.subplot(2,4,8)
    sns.barplot(x='severity', y='wind_speed', data=train)

def countplot_data(train):
    plt.figure(figsize=[20,16])
    plt.subplot(4,3,1)
    sns.countplot(x=train.severity, hue=train.traffic_calming)
    plt.subplot(4,3,2)
    sns.countplot(x=train.severity, hue=train.station)
    plt.subplot(4,3,3)
    sns.countplot(x=train.severity, hue=train.roundabout)
    plt.subplot(4,3,4)
    sns.countplot(x=train.severity, hue=train.railway)
    plt.subplot(4,3,5)
    sns.countplot(x=train.severity, hue=train.no_exit)
    plt.subplot(4,3,6)
    sns.countplot(x=train.severity, hue=train.junction)
    plt.subplot(4,3,7)
    sns.countplot(x=train.severity, hue=train.give_way)
    plt.subplot(4,3,8)
    sns.countplot(x=train.severity, hue=train.crossing)
    plt.subplot(4,3,9)
    sns.countplot(x=train.severity, hue=train.bump)
    plt.subplot(4,3,10)
    sns.countplot(x=train.severity, hue=train.amenity)
    plt.subplot(4,3,11)
    sns.countplot(x=train.severity, hue=train.stop)
    plt.subplot(4,3,12)
    sns.countplot(x=train.severity, hue=train.traffic_signal)

def plot_time_data(df):
    df_time = df.set_index('start_time').sort_index()
    df_time.index = pd.to_datetime(df_time.index)
    plt.figure(figsize=[20,10])
    plt.subplot(2,2,1)
    sns.countplot(x=df_time.index.hour)
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Accidents')
    plt.title('Accidents occur more frequently during the late afternoon')
    plt.subplot(2,2,2)
    sns.countplot(x=df_time.index.day)
    plt.xlabel('Day of the Month')
    plt.ylabel('Number of Accidents')
    plt.title('There is no relationship between accidents and day of the month')
    plt.subplot(2,2,3)
    sns.countplot(x=df_time.index.month_name())
    plt.xticks(rotation=90)
    plt.xlabel('Month')
    plt.ylabel('Number of Accidents')
    plt.title('Accidents occur more frequently in April through June')
    plt.subplot(2,2,4)
    sns.countplot(x=df_time.index.year)
    plt.xlabel('Year')
    plt.ylabel('Number of Accidents')
    plt.title('There is an upward trend in the number of recorded accidents each year')
    plt.suptitle('The Relationship between Vehicle Accidents and Different Measurements of Time')
    plt.show()

def stat_levene(x,y):
    alpha = .05
    stat, p = stats.levene(x,y)
    if p < alpha:
        print('we can reject the null hypothesis and posit that variance is inequal')
    else:
        print('we fail to reject the null hypothesis that variance is equal')

def stat_kruskal(x,y,z):
    alpha = .05
    stat, p = stats.kruskal(x,y,z)
    if p < alpha:
        print('we can reject the null hypothesis that there is no mean difference.')
    else:
        print('we fail to reject the null hypothesis that there is no mean difference.')
    print(f'H: {stat}, p-value: {p}')

def stat_chi2(x,y):
    alpha = .05
    observed = pd.crosstab(x, y)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    #output values
    print('Observed')
    print(observed.values)
    print('\nExpected')
    print(expected.astype(int))
    print('\n----')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p}')
    if p < alpha:
        print('We reject the null hypothesis of independence.')
    else:
        print('We fail to reject the null hypothesis of independence.')

