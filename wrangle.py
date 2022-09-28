import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def wrangle_data():
    '''This function reads the accidents.csv file from the current directory and cleans the dataset. Observations with
        null values are dropped, column names are adjusted to be python friendly, and boolean columns are mapped for 
        modeling. Unuseful columns are removed. The function returns a cleaned and prepared dataframe.'''
    # read the csv into pandas and save as a variable
    df = pd.read_csv('accidents.csv')
    # set all column names to lowercase letters
    df.columns = df.columns.str.lower()
    # drop columns that have no use (US is the only country, airport code is not needed, there are no observations with
    # a turning loop, and the building number is not required for predictions at this time)
    df = df.drop(columns=['number','country','airport_code','turning_loop'])
    # drop nulls
    df = df.dropna()
    # change weather_timestamp to datetime
    df.weather_timestamp = pd.to_datetime(df.weather_timestamp)
    # change start_time to datetime
    df.start_time = pd.to_datetime(df.start_time)
    # change end_time to datetime
    df.end_time = pd.to_datetime(df.end_time)
    # rename duplicate values to match
    df.wind_direction = df.wind_direction.str.replace('North','N').str.replace('West','W')\
    .str.replace('South','S').str.replace('East','E').str.replace('Variable', 'VAR')
    # map the column; day=0, night=1
    df.sunrise_sunset = df.sunrise_sunset.map({'Day': 0, 'Night': 1})
    # map the column; day=0, night=1
    df.civil_twilight = df.civil_twilight.map({'Day': 0, 'Night': 1})
    # map the column; day=0, night=1
    df.nautical_twilight = df.nautical_twilight.map({'Day': 0, 'Night': 1})
    # map the column; day=0, night=1
    df.astronomical_twilight = df.astronomical_twilight.map({'Day': 0, 'Night': 1})
    # map the column; true=1, false=0
    df.amenity = df.amenity.map({True: 1, False: 0})
    # map the column; true=1, false=0
    df.bump = df.bump.map({True: 1, False: 0})
    # map the column; true=1, false=0
    df.crossing = df.crossing.map({True: 1, False: 0})
    # map the column; true=1, false=0
    df.give_way = df.give_way.map({True: 1, False: 0})
    # map the column; true=1, false=0
    df.junction = df.junction.map({True: 1, False: 0})
    # map the column; true=1, false=0
    df.no_exit = df.no_exit.map({True: 1, False: 0})
    # map the column; true=1, false=0
    df.railway = df.railway.map({True: 1, False: 0})
    # map the column; true=1, false=0
    df.roundabout = df.roundabout.map({True: 1, False: 0})
    # map the column; true=1, false=0
    df.station = df.station.map({True: 1, False: 0})
    # map the column; true=1, false=0
    df.stop = df.stop.map({True: 1, False: 0})
    # map the column; true=1, false=0
    df.traffic_calming = df.traffic_calming.map({True: 1, False: 0})
    # map the column; true=1, false=0
    df.traffic_signal = df.traffic_signal.map({True: 1, False: 0})
    # rename columns to remove measurement types and parentheses
    df = df.rename(columns={'distance(mi)':'distance', 'temperature(f)':'temperature', 'wind_chill(f)':'wind_chill', 
                  'humidity(%)':'humidity', 'pressure(in)':'pressure', 'visibility(mi)':'visibility', 
                  'wind_speed(mph)':'wind_speed', 'precipitation(in)':'precipitation'})
    # remove outliers for wind_chill
    df = df[df.wind_chill < 120]
    # remove outliers for wind_speed
    df = df[df.wind_speed < 60]
    # return the dataframe
    return df

def downsample_data(df):
    '''This function takes in a dataframe and takes a sample of the over-represented severity level (2) that is similar
        in size to the other three levels. The three separate dataframes are concatenated and a downsampled dataframe
        is returned.'''
    # isolate the oversampled level in severity and take a random sample of 65000
    sev_two = df[df.severity==2].sample(65000, random_state=217)
    # take all samples of severity level 3 and save to a variable
    sev_three = df[df.severity==3]
    # take all samples of severity level 4 and save to a variable
    sev_four = df[df.severity==4]
    # concatenate the samples into a single dataframe
    df = pd.concat([sev_two, sev_three, sev_four])
    # return the downsampled dataframe
    return df

def time_index_data(df):
    '''This function takes in a dataframe and sets the start_time as the index. The index is sorted and a new column called
        total_time is created by subtracting the index datetime from the end_time. Observations with a total_time of more
        than one day are removed. The function returns the newly indexed dataframe for time series analysis.'''
    # set the start_time column as a datetime index and sort
    df = df.set_index(df.start_time).sort_index()
    # create a column called 'total_time' by subtracting end_time from the index datetime
    df['total_time'] = df.end_time - df.index
    # remove observations with a total time of greater than 1 day
    df = df[df.total_time <= '1 days']
    # return the datetime indexed dataframe
    return df
