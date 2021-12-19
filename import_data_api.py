# Covid-19 API: "https://api.covid19api.com/summary" - based on JHU


# import libraries

import pandas as pd
import requests

# Constants
COVID_API_URL = "https://api.covid19api.com/summary"

# REST API based on John Hopkins University COVID-19 dataset.
response = requests.get(url=COVID_API_URL)

# check for successfully request (status code 200)
if response.status_code == 200:
    result = response.json()
else:
    print('connection error')

# normalize data for the countries into a flat table.
df = pd.json_normalize(result['Countries'])

# List of columns to drop
drop_columns = ['ID', 'CountryCode', 'Slug', 'NewRecovered', 'TotalRecovered']
df.drop(drop_columns, axis=1, inplace=True)

#print('count',df.count() )
#print('not null',df.notnull().sum())

# convert date to datetime type and change the date format to YYYY-MM-DD using lambda function.
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))

# check whether the date of data is the same in all countries
date_is_same = df['Date'].eq(df['Date'].iloc[0]).all()

# if there are any date different - create a list with countries where the date != date in first row.
if not date_is_same:
    date = df['Date'].iloc[0]
    country_list = []

    for index, row in df.iterrows():
         if (row['Date'] != date):
             country_list.append(row['Country'])

    if len(country_list) == 1:
        formatted_country_string = country_list[0]
    else:
        formatted_country_string = ', '.join(map(str, country_list))
        print("The data available for {country} is not updated and therefore exact information about the worldwide Covid-19 status is not possible.".format(country=formatted_country_string))

else:
    # create an overview of covid19 cases worldwide
    total = df.sum(numeric_only=True)

    print('###### Worldwide Covid-19 status update. Date: {date} ######'.format(date=pd.to_datetime(df['Date'].iloc[0]).strftime("%d %b %Y")))
    print('Total cases: {:d}'.format(int(total['TotalConfirmed'])))
    print('Total deaths: {:d}'.format(int(total['TotalDeaths'])))
    print('New cases: {:d}'.format(int(total['NewConfirmed'])))
    print('New deaths: {:d}'.format(int(total['NewDeaths'])))
    print('############################################################')
