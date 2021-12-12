# Data from https://coronavirus.jhu.edu/map.html  JHU CSSE
# Raw data for covid cases: https://github.com/CSSEGISandData/COVID-19
# Raw data for vaccine: https://github.com/govex/COVID-19/tree/master/data_tables/vaccine_data
# last update: 07.12.2021

# all data is stored in ./data folder


# import libraries
from pathlib import Path
from datetime import datetime
from re import search
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import requests
import re

# Constants
csv_data_folder = './data'
covid_api_url = "https://api.covid19api.com/summary"

# REST API based on John Hopkins University COVID-19 dataset.
response = requests.get(url=covid_api_url)

# check for successfully request (status code 200)
if response.status_code == 200:
    result = response.json()
else:
    print('connection error')

# print(result)

# get data for the countries into dataframe
df = pd.json_normalize(result['Countries'])

# print(df.info())

# List of columns to delete
del_columns = ['ID', 'CountryCode', 'Slug', 'NewRecovered', 'TotalRecovered']

df.drop(del_columns, axis=1, inplace=True)

# create an overview of covid19 cases worldwide
# df.loc['Total'] = df.sum(numeric_only=True)


# convert date to datetime Dtype and change the date format to YYYY-MM-DD using lambda function.
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))

# check whether the date of data is the same in all countries
date_is_same = df['Date'].eq(df['Date'].iloc[0]).all()

# if there are different days create a list with countries where the date != date in row 0.
if not date_is_same:
    date = df['Date'].iloc[0]
    country_list = []
    for index, val in df.iterrows():
         if (str(val['Date']) != str(date)):
             country_list.append(val['Country'])

    print(country_list)

else:
    print()




# Getter for data from files.
def get_data(name):
    for file_name in Path(csv_data_folder).glob("*.csv"):
        if search(name, str(file_name)):
            df = pd.read_csv(file_name)
            return df


# creating dataframes using function
df_confirmed = get_data('confirmed')
df_deaths = get_data('deaths')
df_recovered = get_data('recovered')
df_vaccine = get_data('vaccine')

# checking the structure
# print(df_confirmed.head())
# print(df_confirmed.head())

#


#
df_vaccine.rename(columns={'Country_Region': 'Country/Region', 'Province_State': 'Province/State'}, inplace=True)
df_confirmed.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)

# creating new dataframe for Germany
df_confirmed_de = df_confirmed[df_confirmed['Country/Region'] == 'Germany']
df_confirmed_ru = df_confirmed[df_confirmed['Country/Region'] == 'Russia']
df_confirmed_us = df_confirmed[df_confirmed['Country/Region'] == 'US']

# print('vaccine', df_vaccine.head())
# print('death', df_deaths.head())
# print('df_recovered', df_recovered.head())
# print('conf', df_confirmed.head())

# Todo: create a function for each dataset

# merge countries into one dataframe
merged_df = pd.concat([df_confirmed_de, df_confirmed_ru, df_confirmed_us])
# rework

test = merged_df.melt(id_vars=['Country'], var_name='Date', value_name='value')
test2 = test.pivot(index='Date', columns='Country', values="value")

test2.index = pd.to_datetime(test2.index)
test2.sort_index(inplace=True)

# Todo: End create function

# sns.set()
# plt.plot(test2)
# plt.title('Confirmed cases')
# plt.ylabel('Amount')
# plt.yticks()
# plt.show()
# sns.set_theme(style="whitegrid")
# sns.lineplot(data=test2, palette="tab10", linewidth=2.5)
# plt.show()

# transform the dataset and merge all together


# Sources

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.transpose.html


# df_confirmed = pd.read_csv('./data/time_series_covid19_confirmed_global.csv')
# df_deaths = pd.read_csv('./data/time_series_covid19_deaths_global.csv')
# df_recovered = pd.read_csv('./data/time_series_covid19_recovered_global.csv')
# df_vaccine = pd.read_csv('./data/time_series_covid19_vaccine_global.csv')
