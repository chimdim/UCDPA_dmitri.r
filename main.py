# Data from https://coronavirus.jhu.edu/map.html  - Johns Hopkins University (JHU)
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
COLOR_CONFIRMED = 'blue'
COLOR_DEATHS = 'red'
TERMS = {'confirmed_title': 'COVID-19 confirmed cases in ', 'deaths_title': 'COVID-19 deaths in '}


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

# print(df_confirmed.info())
# print(df_confirmed.describe())

df_vaccine.rename(columns={'Country_Region': 'Country/Region'}, inplace=True)

df_vaccine.drop(columns=['Province_State'], axis=1, inplace=True)
df_confirmed.drop(columns=['Province/State', 'Lat', 'Long'], axis=1, inplace=True)
df_deaths.drop(columns=['Province/State', 'Lat', 'Long'], axis=1, inplace=True)


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

# there is no "na" data in dataframe
#print(df_ru.notna().sum())
#print(df_de.notna().sum())
#print(df_us.notna().sum())


def draw_chart(data, country, **kwargs):
    daily_conf = data['Daily Confirmed']
    daily_deaths = data['Daily Deaths']

    # Todo: add months selector
    # if kwargs:
    #     months = int(kwargs.get('m', None))
    #     if months:
    #         date = data.iloc[-1]['Date']
    #         new_date = pd.Timestamp(datetime.date(date)-relativedelta(months=months))
    #         # print(date < new_date)
    #         daily_conf = data[data['Date'] < new_date]


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(8, 8))
    ax1.plot(data['Date'], data['Confirmed'], color=COLOR_CONFIRMED)

    ax2.bar(data['Date'], daily_conf, color=COLOR_CONFIRMED)

    ax1.set_title(TERMS['confirmed_title'] + country)
    ax1.tick_params(axis='y', labelsize=10, labelcolor=COLOR_CONFIRMED)
    # ax1.set_ylabel('Confirmed cases', color=COLOR_CONFIRMED, fontsize=12)
    ax1.legend(['Confirmed cases'])
    ax1.grid(True, alpha=.4)

    ax2.grid(True, alpha=.4)
    # ax2.set_ylabel("Daily", color=COLOR_CONFIRMED, fontsize=12)
    ax2.legend(['Daily cases'])
    ax2.tick_params(axis='y', labelsize=10, labelcolor=COLOR_CONFIRMED)

    ax3.set_title(TERMS['deaths_title'] + country)
    ax3.plot(data['Date'],data['Deaths'], color=COLOR_DEATHS)
    # ax3.set_ylabel('Deaths total', color=COLOR_DEATHS, fontsize=12)
    ax3.legend(['Deaths total'])
    ax3.tick_params(axis='y', labelsize=10, labelcolor=COLOR_DEATHS)
    ax3.grid(True, alpha=.4)

    ax4.tick_params(axis='y', labelsize=10, labelcolor=COLOR_DEATHS)
    ax4.bar(data['Date'], daily_deaths, color=COLOR_DEATHS)
    #ax4.set_ylabel('Deaths daily', color=COLOR_DEATHS, fontsize=12)
    ax4.legend(['Deaths daily'])
    ax4.grid(True, alpha=.4)

    # fix to avoid scientific notation in yaxis
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
