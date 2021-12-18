import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# removing too high prices
df_moscow.drop(df_moscow[df_moscow['price'] > 300000000].index, inplace=True)

# visualization of prices in the dataset
plt.figure(figsize=(10,6))
plt.hist(df_moscow['price'], bins=40)
plt.show()

# reduce max price to 65 million rubel which reflects most of the properties in dataset
df_moscow.drop(df_moscow[df_moscow['price'] > 65000000].index, inplace=True)

# display prices in millions
df_moscow['price'] = df_moscow['price']/1000000

# drop properties with less or equal rooms then 0
df_moscow.drop(df_moscow[df_moscow['rooms'] < 0].index, inplace=True)

# drop columns
drop_columns = ['time', 'Unnamed: 0', 'region']
df_moscow.drop(columns=drop_columns, inplace=True)

#print(df_moscow.corr())

fig,ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df_moscow.corr(), cmap="YlGnBu", linewidth=0.2, cbar_kws={"shrink": .6})
ax.xaxis.tick_top()
plt.xticks(rotation="30")
ax.set_title('Correlation matrix: Real estate in Moscow', fontsize=18, pad=20)
plt.show()


plt.figure(figsize=(10,6))
plt.hist(df_moscow['price'], bins=40)
plt.show()




