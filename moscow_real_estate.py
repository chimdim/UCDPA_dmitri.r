import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import linear_model


# --- original dataset was removed from this project due large size ---
# dataset source: https://www.kaggle.com/mrdaniilak/russia-real-estate-20182021

# test = pd.read_csv("data2/all_v2.csv")

# geo coordinates for Moscow
# new_df = test[(test['geo_lat'] > 55.6) & (test['geo_lat'] < 55.8) & (test['geo_lon'] < 37.8) & (test['geo_lon'] > 37.4)]

# new_df['region'].unique()
# test[test['region'].isin([3,81])].to_csv('data2/moscow3.csv')


# -- Export Moscow real estate dataset as CVS file for this project
# test[test['region'] == 3].to_csv('data2/moscow.csv')
df_moscow = pd.read_csv('data2/moscow.csv')

# remove missing values
df_moscow.dropna(inplace=True)

# print(df_moscow.shape)
# print(df_moscow.info())
# print(df_moscow.describe())
# print(df_moscow.head(10))

# print(df_moscow.isna().sum())

# convert negative numbers in price column to positive
df_moscow['price'] = df_moscow['price'].abs()

# removing unrealistic prices for properties that may be due to monthly rent
df_moscow.drop(df_moscow[df_moscow['price'] < 1000000].index, inplace=True)

# removing too high prices
df_moscow.drop(df_moscow[df_moscow['price'] > 300000000].index, inplace=True)

# visualization of prices in the dataset
plt.figure(figsize=(10, 6))
plt.hist(df_moscow['price'], bins=40)
# plt.show()

# reduce max price to 65 million rubles which reflects most of the properties in dataset
df_moscow.drop(df_moscow[df_moscow['price'] > 65000000].index, inplace=True)

# display prices in millions
df_moscow['price'] = df_moscow['price'] / 1000000

# drop properties with less or equal rooms then 0
df_moscow.drop(df_moscow[df_moscow['rooms'] < 1].index, inplace=True)

# drop columns
drop_columns = ['time', 'Unnamed: 0', 'region', 'date']
df_moscow.drop(columns=drop_columns, inplace=True)

# Charts
plt.figure(figsize=(10, 6))
sns.histplot(df_moscow['price'])
plt.xlabel('price (million rubles)')
plt.ylabel('real estate objects')
# plt.show()

# print(df_moscow.corr())

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df_moscow.corr(), cmap="YlGnBu", linewidth=0.2, cbar_kws={"shrink": .6})
ax.xaxis.tick_top()
plt.xticks(rotation="30")
ax.set_title('Correlation matrix: Real estate in Moscow', fontsize=18, pad=20)
# plt.show()

# print(df_moscow['price'].corr(df_moscow['area']))

x = df_moscow.drop('price', axis=1)
y = df_moscow['price']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=10)

# lin regression
lin_regression = linear_model.LinearRegression()
lin_regression.fit(X_train, y_train)

y_pred = lin_regression.predict(X_test)

print('training data:', lin_regression.score(X_train, y_train))
print('test data:', lin_regression.score(X_test, y_test))
print('intercept:', lin_regression.intercept_)
print('coef_:', lin_regression.coef_)

print(pd.DataFrame(data=lin_regression.coef_, index=X_train.columns, columns=['coef']))

# Ridge
ridge = linear_model.Ridge(alpha=0.9)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print('ridge', ridge.score(X_test, y_test))

# Lasso
plt.close()
lasso = linear_model.Lasso(alpha=0.9)
lasso.fit(X_train, y_train)
print('lasso', lasso.score(X_test, y_test))
# print( 'lasso coeff:', lasso.coef_)


# GridSearchCV
# params = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'C': [1, 5, 10], 'degree': [3, 8], 'coef0': [0.01, 10, 0.5],
#           'gamma': ('auto', 'scale')},
# svr = svm.SVR()
#
# search = GridSearchCV(estimator=svr, param_grid=params, cv=3, n_jobs=-1, verbose=2)
# search.fit(X_train, y_train)
# print(search.cv_results_)

classifiers = [
    linear_model.SGDRegressor(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    # linear_model.TheilSenRegressor(),
    linear_model.Lasso(),
    linear_model.Ridge(),
    linear_model.LinearRegression(),
    # svm.SVR()
]

for item in classifiers:
    print(item)
    clf = item
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test),'\n')