import pandas as pd
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
df = pd.read_csv('data2/moscow.csv')

# remove missing values
df.dropna(inplace=True)

# print(df.shape)
# print(df.info())
# print(df.describe())
# print(df.head(10))

# print(df.isna().sum())


# convert negative numbers in price column to positive numbers
df['price'] = df['price'].abs()

# removing unrealistic prices for properties that may be due to monthly rent or fake announcement
df.drop(df[df['price'] < 1000000].index, inplace=True)

# removing too high prices
df.drop(df[df['price'] > 300000000].index, inplace=True)

# reduce max price to 65 million rubles which reflects most of the properties in dataset
df.drop(df[df['price'] > 65000000].index, inplace=True)

# display prices in millions
df['price'] = df['price'] / 1000000

# drop properties with less or equal rooms then 0
df.drop(df[df['rooms'] < 1].index, inplace=True)

# drop properties where kitchen area > area
df.drop(df[df['kitchen_area'] > df['area']].index, inplace=True)

# drop properties where area < 30 
df.drop(df[df['area'] < 30].index, inplace=True)

# drop properties where area / rooms < 9 m2
df.drop(df[(df['area'] / df['rooms']) < 9].index, inplace=True)

# drop columns
drop_columns = ['time', 'Unnamed: 0', 'region', 'date']
df.drop(columns=drop_columns, inplace=True)



# Todo: create Dataframe for results. Results {'model': [Score, MAE, RMSE]}
model_name = np.empty(0, dtype="str")
model_mae = np.empty(0,dtype='float64')
model_rmse = np.empty([1,6], dtype='float64')
model_r2 = np.empty([1,6], dtype='float64')

model_name = np.append(model_name, 'sas')
model_name = np.append(model_name, 'sas2')
model_mae = np.append(model_mae, 4)
model_mae = np.append(model_mae, 2)
model_r2 = np.append(model_r2, 8)
model_r2 = np.append(model_r2, 3)
model_rmse = np.append(model_rmse, 1)
model_rmse = np.append(model_rmse, 6)



x = np.arange(len(model_name))
width = 0.35  # the width of the bars

#fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8, 10))
# fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8, 10))
#
# ax1.bar(x, model_mae, width, label='mae')
# ax2.bar(x, model_r2, width, label='mae')
#ax2.bar(x, model_rmse, width, label='rmse')
#ax3.bar(x, model_r2, width, label='r2')


# Add some text for labels, title and custom x-axis tick labels, etc.
# ax1.set_ylabel('Scores')
# ax1.set_title('Scores by group and gender')
# ax1.set_xticks(x, model_name)
# ax1.legend()


#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)
#
# fig.tight_layout()
#
# plt.show()



#df.reset_index(drop=True, inplace=True)

# Chart: Histogram
# fig, ax = plt.subplots(figsize=(8, 6))
# ax = sns.histplot(df['price'])
# plt.xlabel('Price (million rubles)')
# plt.ylabel('Real estate objects')
# plt.show()
# plt.close()
#
# Chart: Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.heatmap(df.corr(), cmap="YlGnBu", linewidth=0.2, annot=True, cbar_kws={"shrink": .6})
plt.xticks(rotation="45")
ax.set_title('Correlation matrix', fontsize=18, pad=20)
fig.tight_layout()
plt.show()

# Preparation for test and train data
x = df.drop('price', axis=1).values
y = df['price'].values.reshape(-1,1)



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

# Print function
def print_results(model, name):
    print(name + ' cv_results: ', model.cv_results_)
    print(name + ' best_params: ', model.best_params_)
    print(name + ' best score: ', model.best_score_)
    print(name + '  best estimator: ', model.best_estimator_)


linear = LinearRegression()
linear.fit(X_train, y_train)
print(linear.score(X_test,y_test))

lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
print(lasso.score(X_test,y_test))

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(ridge.score(X_test,y_test))
y_pred = ridge.predict(X_test)

print(pd.DataFrame(y_pred).describe())

print(r2_score(y_test,y_pred), mean_squared_error(y_test,y_pred))


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
# model2 = XGBRegressor(verbosity=2)
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


#Best estimator. Todo: Check why there are negative predicted values when gbtree or dart booster were used.
# xg = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
#              gamma=0.1, gpu_id=-1, importance_type=None,
#              interaction_constraints='', learning_rate=0.300000012,
#              max_delta_step=0, max_depth=10, min_child_weight=1,
#              monotone_constraints='()', n_estimators=100, n_jobs=8,
#              num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
#              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1)
# xg.fit(X_train, y_train)
# print(xg.score(X_test, y_test))
# prediction = xg.predict(X_test)
# #print(pd.DataFrame({'col1':prediction, 'col2':y_test},).reset_index(drop=True).head(20))
# #pd.DataFrame({'col1':prediction, 'col2':y_test}).reset_index(drop=True).to_csv('test_prediction.csv')
# print(r2_score(y_test,prediction).mean(), np.sqrt(mean_squared_error(y_test,prediction)).mean(), mean_squared_error(y_test,prediction) )
# print(pd.DataFrame(prediction).describe())



# fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8, 10))
# ax1 = sns.histplot(y_test, color='blue')
# ax2 = sns.histplot(prediction, color='red')
# plt.show()
# #
