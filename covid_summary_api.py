# Covid-19 REST API: "https://api.covid19api.com/summary" - based on Johns Hopkins University COVID-19 dataset


# import libraries
import pandas as pd
import requests
import re

# Constants
COVID_API_URL = "https://api.covid19api.com/summary"

response = requests.get(url=COVID_API_URL)

# Todo: change it to try catch block.
# check for successfully request (status code 200).
if response.status_code != 200:
    print('connection error')
else:
    result = response.json()


# create temp csv file for tests and error simulations
# df_temp = pd.json_normalize(result['Countries']).to_csv('test.csv')

# checking for data in response
if result['Countries'] is not None:

    # normalize data for the countries into a flat table.
    df = pd.json_normalize(result['Countries'])

    # List of columns to drop
    drop_columns = ['ID', 'CountryCode', 'Slug', 'NewRecovered', 'TotalRecovered']
    df.drop(drop_columns, axis=1, inplace=True)

    # print('count',df.count() )
    # print('not null',df.notnull().sum())

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
            if row['Date'] != date:
                country_list.append(row['Country'])

        if len(country_list) == 1:
            formatted_country_string = country_list[0]
        else:
            formatted_country_string = ', '.join(map(str, country_list))
            print(
                "The data available for {country} is not updated and therefore exact \ninformation about the worldwide Covid-19 status is not possible.".format(
                    country=formatted_country_string))

    else:
        # create an overview of covid19 cases worldwide
        total = df.sum(numeric_only=True)

        # convert API URL to "Data Source" string using Regex
        domain_name = re.search(r'\w*://(.*?)/\w*', COVID_API_URL).group(1)
        domain_name = re.sub('^api', 'www', domain_name)

        print('###### Worldwide Covid-19 status update. Date: {date} ######'.format(
            date=pd.to_datetime(df['Date'].iloc[0]).strftime("%d %b %Y")))
        print('Total cases: {:d}'.format(int(total['TotalConfirmed'])))
        print('Total deaths: {:d}'.format(int(total['TotalDeaths'])))
        print('New cases: {:d}'.format(int(total['NewConfirmed'])))
        print('New deaths: {:d}'.format(int(total['NewDeaths'])))
        print('#################################################################')
        print('Data source:' + ' ' + domain_name)

else:
    print("Please try again later")
