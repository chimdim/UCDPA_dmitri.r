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
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import requests


# Constants
csv_data_folder = './data'
covid_api_url = "https://api.covid19api.com/summary"
years = mdates.YearLocator()
months = mdates.MonthLocator()
yearsFmt = mdates.DateFormatter('%Y')


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
df_vaccine = get_from_csv('vaccine')


# print(df_confirmed.head())
# print(df_deaths.head())
# print(df_vaccine.head())

df_vaccine.rename(columns={'Country_Region': 'Country/Region'}, inplace=True)

df_vaccine.drop(columns=['Province_State'],axis=1, inplace=True)
df_confirmed.drop(columns=['Province/State', 'Lat', 'Long'],axis=1, inplace=True)
df_deaths.drop(columns=['Province/State', 'Lat', 'Long'],axis=1, inplace=True)


# add "status" column to the dataframes
#df_confirmed['Status'] = 'confirmed'
#df_deaths['Status'] = 'death'

def convert_data(country):
    confirmed = df_confirmed[df_confirmed['Country/Region'] == country]
    deaths = df_deaths[df_deaths['Country/Region'] == country]
    confirmed_transformed = confirmed.melt(id_vars=['Country/Region'], var_name='Date', value_name='Confirmed')
    deaths_transformed = deaths.melt(id_vars=['Country/Region'], var_name='Date', value_name='Deaths')
    merged_data = pd.merge(confirmed_transformed, deaths_transformed, how="inner")

    # format date to datetime object
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])

    # calculate daily data for confirmed and death cases
    merged_data['Daily Deaths'] = merged_data['Deaths'] - merged_data['Deaths'].shift()
    merged_data['Daily Confirmed'] = merged_data['Confirmed'] - merged_data['Confirmed'].shift()

    # fixing the value in the first row for calculated values and convert column to int32 type
    merged_data['Daily Deaths'] = merged_data['Daily Deaths'].fillna(0).astype('int32')
    merged_data['Daily Confirmed'] = merged_data['Daily Confirmed'].fillna(0).astype('int32')

    return merged_data


df_ru = convert_data('Russia')
df_de = convert_data('Germany')
df_us = convert_data('US')


# Todo: End create function

def draw_chart(data,country,month):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(data['Date'], data['Confirmed'],data['Date'],data['Deaths'])
    ax.set(xlabel="Date", ylabel="Confirmed cases", title="COVID-19 confirmed cases and deaths in "+country )
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x,p: format(int(x), ',')))
    # plt.title('Confirmed cases')
    # plt.ylabel('Amount')
    plt.yticks()
    plt.show()
    # sns.set_theme(style="whitegrid")
    # sns.lineplot(data=test2, palette="tab10", linewidth=2.5)
    # plt.show()

draw_chart(df_de,'Germany',6)



# Sources

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.transpose.html

