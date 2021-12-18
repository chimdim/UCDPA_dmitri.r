import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
#plt.show()

# reduce max price to 65 million rubel which reflects most of the properties in dataset
df_moscow.drop(df_moscow[df_moscow['price'] > 65000000].index, inplace=True)

# display prices in millions
df_moscow['price'] = df_moscow['price']/1000000

# drop properties with less or equal rooms then 0
df_moscow.drop(df_moscow[df_moscow['rooms'] < 0].index, inplace=True)

# drop columns
drop_columns = ['time', 'Unnamed: 0', 'region', 'date']
df_moscow.drop(columns=drop_columns, inplace=True)
df_moscow.astype('float64')
print(df_moscow.info())
# Charts<

plt.figure(figsize=(10,6))
sns.histplot(df_moscow['price'])
plt.xlabel('price (million rubel)')
plt.ylabel('real estate objects')
#plt.show()

#print(df_moscow.corr())

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df_moscow.corr(), cmap="YlGnBu", linewidth=0.2, cbar_kws={"shrink": .6})
ax.xaxis.tick_top()
plt.xticks(rotation="30")
ax.set_title('Correlation matrix: Real estate in Moscow', fontsize=18, pad=20)
#plt.show()

#print(df_moscow['price'].corr(df_moscow['area']))

x = df_moscow.drop('price', axis=1)
y = df_moscow['price']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

lin_regression = LinearRegression()
lin_regression.fit(X_train, y_train)

y_pred= lin_regression.predict(X_test)


print('training data:', lin_regression.score(X_train, y_train))
print('test data:', lin_regression.score(X_test, y_test))
print('intercept:', lin_regression.intercept_)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))

cv_results = cross_val_score(lin_regression, x,y, cv=3)
print('cv', cv_results)

print(pd.DataFrame(data=lin_regression.coef_, index=X_train.columns, columns=['coef']))

#Ridge
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print('ridge', ridge.score(X_test, y_test))
print('ridge pred', ridge_pred)

#Lasso
plt.close()
lasso = Lasso(alpha=0.1, normalize=True)
lasso_coef = lasso.fit(X_train, y_train).coef_
_ = plt.plot(range(len(x.columns)), lasso_coef)
_ = plt.xticks(range(len(x.columns)), x.columns, rotation=60)
_ = plt.ylabel('Coeff')
plt.show()
print('lasso', lasso.score(X_test, y_test))




#SMF
ml = smf.ols('area~geo_lat+rooms+geo_lon+kitchen_area+building_type', data=df_moscow).fit()

print(ml.summary())

X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()
print(pd.DataFrame({'coef': results.params, 'p-value': round(results.pvalues, 6)}))

print(variance_inflation_factor(exog=X_incl_const.values, exog_idx=1))
vif = [variance_inflation_factor(exog=X_incl_const.values, exog_idx=1) for i in range(X_incl_const.shape[1])]

print(X_incl_const.shape[1])


#print(pd.DataFrame({'coef_name': X_incl_const.columns, 'vif': np.around(vif, 6)}))


feature = df_moscow.drop('price', axis=1)
target = df_moscow['price']
