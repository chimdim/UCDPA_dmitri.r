import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

# dataset source: https://www.kaggle.com/mrdaniilak/russia-real-estate-20182021
# original dataset was removed from this project due large size

#test = pd.read_csv("data2/all_v2.csv")

#new_df = test[(test['geo_lat'] > 55.6) & (test['geo_lat'] < 55.8) & (test['geo_lon'] < 37.8) & (test['geo_lon'] > 37.4)]

#new_df['region'].unique()

#test[test['region'].isin([3,81])].to_csv('data2/moscow3.csv')

#test[test['region'] == 3].to_csv('data2/moscow.csv')



df_moscow = pd.read_csv('data2/moscow.csv')

# remove missing values
df_moscow.dropna(inplace=True)

#print(df_moscow.shape)
#print(df_moscow.info())
#print(df_moscow.describe())
#print(df_moscow.head(10))

#print(df_moscow.isna().sum())

# convert negative numbers in price column to positive
df_moscow['price'] = df_moscow['price'].abs()

# removing unrealistic prices for properties that may be due to monthly rent
df_moscow.drop(df_moscow[df_moscow['price'] < 1000000].index, inplace=True)

# removing too expensive properties > 1.8 million €
df_moscow.drop(df_moscow[df_moscow['price'] > 150000000].index, inplace=True)

# drop properties with less or equal rooms then 0
df_moscow.drop(df_moscow[df_moscow['rooms'] < 0].index, inplace=True)

# drop columns
df_moscow.drop(['time', 'Unnamed: 0', 'region'], axis=1, inplace=True)


#print(df_moscow.corr())
#print(df_moscow.info())



sns.heatmap(df_moscow.corr(), cmap="YlGnBu" )
plt.show()

# Charts
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8, 8))




