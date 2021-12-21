import pandas as pd
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor



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
# plt.figure(figsize=(10, 6))
# plt.hist(df_moscow['price'], bins=40)
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
df_moscow.reset_index(drop=True, inplace=True)

# Chart: Histogram
# fig, ax = plt.subplots(figsize=(8, 6))
# ax = sns.histplot(df_moscow['price'])
# plt.xlabel('Price (million rubles)')
# plt.ylabel('Real estate objects')
# plt.show()
# plt.close()
#
# # Chart: Heatmap
# fig, ax = plt.subplots(figsize=(8, 6))
# ax = sns.heatmap(df_moscow.corr(), cmap="YlGnBu", linewidth=0.2, cbar_kws={"shrink": .6})
# plt.xticks(rotation="-45")
# ax.set_title('Correlation matrix', fontsize=18, pad=20)
# fig.tight_layout()
# plt.show()

# Preparation for test and train data
x = df_moscow.drop('price', axis=1).values
y = df_moscow['price'].values.reshape(-1,1)



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=10)

# Print function
def print_results(model, name):
    print(name + ' cv_results: ', model.cv_results_)
    print(name + ' best_params: ', model.best_params_)
    print(name + ' best score: ', model.best_score_)
    print(name + '  best estimator: ', model.best_estimator_)


linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)
print(linear.score(X_test,y_test))

lasso = linear_model.Lasso()
lasso.fit(X_train, y_train)
print(lasso.score(X_test,y_test))

ridge = linear_model.Ridge()




# # HistGradientBoostingRegressor
# model = HistGradientBoostingRegressor(random_state=10)
# params = {
#     'max_iter': [10, 20, 50],
#     'max_depth': [3, 10]
# }
# hgb = GridSearchCV(estimator=model, param_grid=params, cv=5, verbose=2)
# hgb.fit(X_train, y_train)
# print_results(hgb, 'HistGradientBoostingRegressor')
#
#
# # XGBooster
# model2 = XGBRegressor()
# params2 = {
#     'max_depth': [3, 10],
#     'n_estimators': [50, 100],
#     'gamma': [0.05, 0.1]
# }
# xgb = GridSearchCV(estimator=model2, param_grid=params2, cv=5, verbose=2)
# xgb.fit(X_train, y_train)
# print_results(xgb, 'XGBRegressor')

# GradientBoostingRegressor
# model3 = GradientBoostingRegressor(random_state=10)
# params = {
#     'n_estimators': [50, 100],
#     'max_depth': [3, 10]
# }
# gb = GridSearchCV(estimator=model3, param_grid=params, cv=5, verbose=2)
# gb.fit(X_train, y_train)
# print_results(gb, 'GradientBoostingRegressor')


# Best estimator
# xg = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#                   colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
#                   gamma=0.05, gpu_id=-1, importance_type=None,
#                   interaction_constraints='', learning_rate=0.300000012,
#                   max_delta_step=0, max_depth=10, min_child_weight=1,
#                   monotone_constraints='()', n_estimators=100, n_jobs=8,
#                   num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
#                   reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
#                   validate_parameters=1, verbosity=None)
#
# xg.fit(X_train, y_train)
# print(xg.score(X_test, y_test))
# prediction = xg.predict(X_test)
# print(pd.DataFrame({'col1':prediction, 'col2':y_test},).reset_index(drop=True).head(20))
# pd.DataFrame({'col1':prediction, 'col2':y_test}).reset_index(drop=True).to_csv('test_prediction.csv')

