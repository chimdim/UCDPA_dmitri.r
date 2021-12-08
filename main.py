# Data from https://coronavirus.jhu.edu/map.html  JHU CSSE
# Raw data for covid cases: https://github.com/CSSEGISandData/COVID-19
# Raw data for vaccine: https://github.com/govex/COVID-19/tree/master/data_tables/vaccine_data
# last update: 07.12.2021

# all data is stored in ./data folder


# import libraries
from pathlib import Path
from re import search
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Constants
data_folder = './data'


# Getter for data from files. Just a small demonstration of using functions.
def get_data(folder, name):
    for file_name in Path(folder).glob("*.csv"):
        if search(name, str(file_name)):
            df = pd.read_csv(file_name)
            return df


# creating dataframes using function
df_confirmed = get_data(data_folder, 'confirmed')
df_deaths = get_data(data_folder, 'deaths')
df_recovered = get_data(data_folder, 'recovered')
df_vaccine = get_data(data_folder, 'vaccine')

# checking the structure
print(df_confirmed.head())

# creating new dataframe for Germany
df_confirmed_de = df_confirmed[df_confirmed['Country'] == 'Germany']
df_confirmed_ru = df_confirmed[df_confirmed['Country'] == 'Russia']
df_confirmed_us = df_confirmed[df_confirmed['Country'] == 'US']



# transform the dataset and merge all together






# Sources

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.transpose.html



# df_confirmed = pd.read_csv('./data/time_series_covid19_confirmed_global.csv')
# df_deaths = pd.read_csv('./data/time_series_covid19_deaths_global.csv')
# df_recovered = pd.read_csv('./data/time_series_covid19_recovered_global.csv')
# df_vaccine = pd.read_csv('./data/time_series_covid19_vaccine_global.csv')
