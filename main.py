# Data from https://coronavirus.jhu.edu/map.html  - John Hopkins University (JHU)
# Raw data for covid cases: https://github.com/CSSEGISandData/COVID-19
# Raw data for vaccine: https://github.com/govex/COVID-19/tree/master/data_tables/vaccine_data
# last update: 07.12.2021

# Covid-19 API: "https://api.covid19api.com/summary" - based on JHU


# import libraries
from pathlib import Path
from re import search
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import requests


# Constants
csv_data_folder = './data'
covid_api_url = "https://api.covid19api.com/summary"
#
# # REST API based on John Hopkins University COVID-19 dataset.
# response = requests.get(url=covid_api_url)
#
# # check for successfully request (status code 200)
# if response.status_code == 200:
#     result = response.json()
# else:
#     print('connection error')
#
# # get data for the countries into dataframe
# df = pd.json_normalize(result['Countries'])
#
# # print(df.info())
#
# # List of columns to delete
# del_columns = ['ID', 'CountryCode', 'Slug', 'NewRecovered', 'TotalRecovered']
# df.drop(del_columns, axis=1, inplace=True)
#
# # convert date to datetime type and change the date format to YYYY-MM-DD using lambda function.
# df['Date'] = pd.to_datetime(df['Date'])
# df['Date'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
#
# # check whether the date of data is the same in all countries
# date_is_same = df['Date'].eq(df['Date'].iloc[0]).all()
#
# # if there are any date different - create a list with countries where the date != date in first row.
# if not date_is_same:
#     date = df['Date'].iloc[0]
#     country_list = []
#
#     for index, row in df.iterrows():
#          if (row['Date'] != date):
#              country_list.append(row['Country'])
#
#     if len(country_list) == 1:
#         formatted_country_string = country_list[0]
#     else:
#         formatted_country_string = ', '.join(map(str, country_list))
#         print("The data available for {country} is not updated and therefore exact information about the worldwide Covid-19 status is not possible.".format(country=formatted_country_string))
#
# else:
#     # create an overview of covid19 cases worldwide
#     total = df.sum(numeric_only=True)
#
#     print('###### Worldwide Covid-19 status update. Date: {date} ######'.format(date=pd.to_datetime(df['Date'].iloc[0]).strftime("%d %b %Y")))
#     print('Total cases: {:d}'.format(int(total['TotalConfirmed'])))
#     print('Total deaths: {:d}'.format(int(total['TotalDeaths'])))
#     print('New cases: {:d}'.format(int(total['NewConfirmed'])))
#     print('New deaths: {:d}'.format(int(total['NewDeaths'])))
#     print('###############################################')
#


# Getter for data from files.
def get_from_csv(name):
    for file_name in Path(csv_data_folder).glob("*.csv"):
        if search(name, str(file_name)):
            data = pd.read_csv(file_name)
            return data


# creating dataframes using function
df_confirmed = get_from_csv('confirmed')
df_deaths = get_from_csv('deaths')
df_recovered = get_from_csv('recovered')
df_vaccine = get_from_csv('vaccine')


# print(df_confirmed.head())
# print(df_deaths.head())
# print(df_recovered.head())
# print(df_vaccine.head())

df_vaccine.rename(columns={'Country_Region': 'Country/Region'}, inplace=True)

df_vaccine.drop(columns=['Province_State'],axis=1, inplace=True)
df_confirmed.drop(columns=['Province/State', 'Lat', 'Long'],axis=1, inplace=True)
df_deaths.drop(columns=['Province/State', 'Lat', 'Long'],axis=1, inplace=True)
df_recovered.drop(columns=['Province/State', 'Lat', 'Long'],axis=1, inplace=True)

# add "status" column to the dataframes
df_confirmed['Status'] = 'confirmed'
df_recovered['Status'] = 'recovered'
df_deaths['Status'] = 'death'


# list countries for analysis
countries = ['Germany', 'Russia', 'US']

df_confirmed = df_confirmed[df_confirmed['Country/Region'].isin(countries)]
df_recovered = df_recovered[df_recovered['Country/Region'].isin(countries)]
df_deaths = df_deaths[df_deaths['Country/Region'].isin(countries)]

print(df_deaths.head())


# print('vaccine', df_vaccine.head())
#print('death', df_deaths.head())
#print('df_recovered', df_recovered.head())
# print('conf', df_confirmed.head())

# Todo: create a function for each dataset

# merge countries into one dataframe
merged_df = pd.concat([df_confirmed, df_recovered, df_deaths])
# rework
#print(merged_df.head())


test = merged_df.melt(id_vars=['Country/Region', 'Status'], var_name='Date', value_name='value')
#test2 = test.pivot(index='Date', columns='Country/Region', values="value")

print(test.tail(20))

# test2.index = pd.to_datetime(test2.index)
# test2.sort_index(inplace=True)

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
