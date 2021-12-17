# Data from https://coronavirus.jhu.edu/map.html  - John Hopkins University (JHU)
# Raw data for covid cases: https://github.com/CSSEGISandData/COVID-19
# Raw data for vaccine: https://github.com/govex/COVID-19/tree/master/data_tables/vaccine_data
# last update: 07.12.2021

# Covid-19 API: "https://api.covid19api.com/summary" - based on JHU


# import libraries
from datetime import datetime
from pathlib import Path
from re import search
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dateutil.relativedelta import relativedelta
import numpy as np
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
df_vaccine = get_from_csv('vaccine')

# print(df_confirmed.head())
# print(df_deaths.head())
# print(df_vaccine.head())

df_vaccine.rename(columns={'Country_Region': 'Country/Region'}, inplace=True)

df_vaccine.drop(columns=['Province_State'], axis=1, inplace=True)
df_confirmed.drop(columns=['Province/State', 'Lat', 'Long'], axis=1, inplace=True)
df_deaths.drop(columns=['Province/State', 'Lat', 'Long'], axis=1, inplace=True)


# add "status" column to the dataframes
# df_confirmed['Status'] = 'confirmed'
# df_deaths['Status'] = 'death'

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

def draw_chart(data, country, **kwargs):
    daily_conf = data['Daily Confirmed']
    daily_deaths = data['Daily Deaths']

    # Todo: add months selector
    # if kwargs:
    #     months = int(kwargs.get('m', None))
    #     if months:
    #         date = data.iloc[-1]['Date']
    #         new_date = pd.Timestamp(datetime.date(date)-relativedelta(months=months))
    #         #print(date < new_date)
    #         daily_conf = data[data['Date'] < new_date]



    color_confirmed = 'blue'
    color_deaths = 'red'

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(8, 8))
    ax1.plot(data['Date'], data['Confirmed'], color=color_confirmed)

    ax2.bar(data['Date'], daily_conf, color=color_confirmed)

    ax1.set_title("COVID-19 confirmed cases in " + country)
    ax1.tick_params(axis='y', labelsize=10, labelcolor=color_confirmed)
    # ax1.set_ylabel('Confirmed cases', color=color_confirmed, fontsize=12)
    ax1.legend(['Confirmed cases'])
    ax1.grid(True, alpha=.4)

    ax2.grid(True, alpha=.4)
    # ax2.set_ylabel("Daily", color=color_confirmed, fontsize=12)
    ax2.legend(['Daily cases'])
    ax2.tick_params(axis='y', labelsize=10, labelcolor=color_confirmed)

    ax3.set_title("COVID-19 deaths in " + country)
    ax3.plot(data['Date'],data['Deaths'], color=color_deaths)
    # ax3.set_ylabel('Deaths total', color=color_deaths, fontsize=12)
    ax3.legend(['Deaths total'])
    ax3.tick_params(axis='y', labelsize=10, labelcolor=color_deaths)
    ax3.grid(True, alpha=.4)

    ax4.tick_params(axis='y', labelsize=10, labelcolor=color_deaths)
    ax4.bar(data['Date'], daily_deaths, color=color_deaths)
    #ax4.set_ylabel('Deaths daily', color=color_deaths, fontsize=12)
    ax4.legend(['Deaths daily'])
    ax4.grid(True, alpha=.4)

    # fix to avoid scientific notation in the yaxis
    all_axes = [ax1, ax2, ax3, ax4]
    for ax in all_axes:
        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    fig.tight_layout()
    plt.show()


# arguments: dataframe, Country, months as m= number as string
draw_chart(df_de, 'Germany', m='3')
draw_chart(df_ru, 'Russia')
draw_chart(df_us, 'USA')





# Sources

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.transpose.html
