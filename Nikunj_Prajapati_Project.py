#
# Course: Programming for Big Data
# Project: COVID-19 prediction.
# creation date: 04/04/2020
# Author's name: Nikunj Prajapati
#

####################################
## Import Libraries               ##
####################################

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

####################################
## Read dataset                   ##
# ##################################
# Here, The Last Update column do not has the data in string format. So, I am not going to convert in date object format.
corona_df = pd.read_csv('./Corona-dataset/covid_19_data.csv',
                        parse_dates=['Last Update'])
# Earliest Cases
print(corona_df.head())

# Latest Cases
print(corona_df.tail())

####################################
## Data Transformation            ##
# ##################################
# renaming ObservationDate to Date and Country/Region to Country.
corona_df.rename(columns={'ObservationDate': 'Date',
                          'Country/Region': 'Country'}, inplace=True)


# Here, I am verifying the datatype of the corona_df. Before making the model,
# all the independent variables should be in the numerical format except date.
print(corona_df.dtypes)
print(corona_df.shape)

# Normalize the data.
# Here, I am checking any null values in any columns. If there are, I need to replace those values.
# Only Province column has the null values.
corona_df.isnull().sum().to_frame('nulls')
corona_df["Province/State"].fillna("NA", inplace=True)

####################################
## Descriptive Analysis           ##
# ##################################
print(corona_df.describe())

####################################
## Exploratory Analysis           ##
# ##################################
# Let's get the total Confirmed, Deaths and Recovered cases in the world(till 04/14).

# Here, I have done groupby date and country.
corona_df = corona_df.groupby(["Date", "Country"])[
    ['Date', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
corona_sorted_df = corona_df.sort_values('Date', ascending=False)
corona_sorted_df = corona_sorted_df.drop_duplicates('Country')

# Now, I have all the unique date in the date column with country.
print(corona_sorted_df.head(80))

corona_confirmed_total_df = corona_sorted_df['Confirmed'].sum()
corona_deaths_total_df = corona_sorted_df['Deaths'].sum()
corona_recovered_total_df = corona_sorted_df['Recovered'].sum()

corona_confirmed_total_df = round(corona_confirmed_total_df, 2)
corona_deaths_total_df = round(corona_deaths_total_df, 2)
corona_recovered_total_df = round(corona_recovered_total_df, 2)

corona_dict = {'Total Confirmed cases  in the world': corona_confirmed_total_df,
               'Total Deaths cases in the world': corona_deaths_total_df, 'Total Recovered cases in the world': corona_recovered_total_df}
corona_dict = pd.DataFrame.from_dict(
    corona_dict, orient='index', columns=['Total'])

corona_dict.style.background_gradient(cmap='Blues')

# Graphic illustrates Total cases in the world.
corona_dict = corona_dict.head(3)
x = corona_dict.index
y = corona_dict['Total'].values

plt.rcParams['figure.figsize'] = (10, 6)
plt.bar(x, y, color=['blue', 'red', 'green'])
plt.title('Total Confirmed / Deaths / Recovered cases worldwide(until 04/14)')
plt.show()

# COVID-19 cases with numbers.
# Confirmed : Cumulative number of confirmed cases till that date <br>
# Deaths : Cumulative number of of deaths till that date <br>
# Recovered : Cumulative number of recovered cases till that date <br>

# Let's have a look at difference in the number of the corona cases between 04/14 and 04/13
corona_diff_df = corona_df.groupby(["Date", "Country"])[
    ['Date', 'Country', 'Confirmed']].sum().reset_index()
corona_confirmed_sorted_df = corona_diff_df.sort_values(
    'Country', ascending=False)

corona_13_df = corona_confirmed_sorted_df[corona_confirmed_sorted_df.Date ==
                                          '04/13/2020'].reset_index().drop('index', axis=1)
corona_14_df = corona_confirmed_sorted_df[corona_confirmed_sorted_df.Date ==
                                          '04/14/2020'].reset_index().drop('index', axis=1)

corona_diff = pd.merge(corona_14_df, corona_13_df, on='Country')
corona_diff['cases_difference_13_14'] = corona_diff['Confirmed_x'] - \
    corona_diff['Confirmed_y']
corona_diff.sort_values('cases_difference_13_14', ascending=False).head(
    60).style.background_gradient(cmap='Reds')

# Top 10  infected Countries
sorted_By_Confirmed1 = corona_sorted_df.sort_values(
    'Confirmed', ascending=False)
sorted_By_Confirmed1 = sorted_By_Confirmed1.head(10)
x = sorted_By_Confirmed1.Country
y = sorted_By_Confirmed1.Confirmed
plt.rcParams['figure.figsize'] = (12, 10)
plt.bar(x, y, color=(0.2, 0.4, 0.6, 0.6))
plt.title('Top 10 infected countries(until 04/14)')
plt.show()

# Table that illustrates increasing infections cases in the world per day .
corona_cases_per_day = corona_df.groupby(
    ["Date"])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
sorted_By_Confirmed1 = corona_cases_per_day.sort_values(
    'Date', ascending=False)

sorted_By_Confirmed1.style.background_gradient(cmap='Reds')

# Line graph that illustrates increasing infections cases per day.
x = corona_cases_per_day.Date
y = corona_cases_per_day.Confirmed
y1 = corona_cases_per_day.Deaths
y2 = corona_cases_per_day.Recovered
ax = plt.gca()
corona_cases_per_day.plot(kind='line', x='Date',
                          y='Confirmed', color='blue', ax=ax)
corona_cases_per_day.plot(kind='line', x='Date',
                          y='Deaths', color='red', ax=ax)
corona_cases_per_day.plot(kind='line', x='Date',
                          y='Recovered', color='green', ax=ax)
print("Blue : Confirmed Cases ")
print("Red : Death Cases ")
print("Green : Recovered Cases ")
plt.ylabel('Total cases(#)')
plt.title('Total confirmed, death and recovered case report per day.')
plt.show()

# Now, Let's prepare the data for the prediction.
# I am converting Date column of dataset to numeric for ensuring the independent variable should be on the same scale as other.
# In the date, I assume the start date as 0 (first day) when the COVID-19 had started and increasing by 1 over increament in date.
# Splitting corona data into Date and confirmed cases.
corona_index_df = pd.DataFrame(corona_cases_per_day.index)
corona_confirmed_df = pd.DataFrame(corona_cases_per_day.Confirmed)
corona_recovered_df = pd.DataFrame(corona_cases_per_day.Recovered)
corona_death_df = pd.DataFrame(corona_cases_per_day.Deaths)

####################################
## Prepare dataset for model      ##
###################################

# Splitting the data.
# Here, I am preparing the train, validate and test dataset.
# Using 90% of the data to train the model.
x_days_train = corona_index_df[0:78]
x_days_validate = corona_index_df[78:82]
x_days_test = corona_index_df[82:84]

# Confirmed
y_confirmed_train = corona_confirmed_df[0:78]
y_confirmed_validate = corona_confirmed_df[78:82]
y_confirmed_test = corona_confirmed_df[82:84]

####################################
##  Regression Model             ##
###################################

# Let's apply Linear regression model on the dataset. As I can see, all the data are continuous, I think that Linear regression will be perfect.
# Linear regression
ln_model = LinearRegression()
ln_model.fit(x_days_train, y_confirmed_train)

plt.scatter(x_days_train, y_confirmed_train, color='red')
plt.plot(x_days_train, ln_model.predict(x_days_train), color='blue')
plt.title('COVID-19 confimed cases of worldwide(Linear Regression)')
plt.xlabel('Days')
plt.ylabel('Confirmed(#)')
plt.show()

# # After applying the linear regression, I can see the predicted line do not fit with the independent variables.
# # It means that the line is unable to capture the patterns in the data.This may be an example of under-fitting.
# # For confirmation, lets check the rmse and r2 score for the model.
print('Linear Regession  R2 Score   : ',
      r2_score(y_confirmed_train, ln_model.predict(x_days_train)))
rmse = np.sqrt(mean_squared_error(
    y_confirmed_train, ln_model.predict(x_days_train)))
print('RMSE of Linear regression is :', rmse)

# R2 score is 63%. It means that 63% of variance is covered by the model.
# Now, to overcome the under-fitting problem, I have to increase the complexity of the model.
# Here, I have used Polynomial linear regression technique because it best fit for the prediction of how deceases spread accross the territory or the world.
# Polynomial linear regression to predict future confirmed cases.

# Polynomal linear Regression (degree=10)
poly_reg = PolynomialFeatures(degree=10)
# Model training with the train data.
x_poly = poly_reg.fit_transform(x_days_train)
# Applying linear regression.
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y_confirmed_train)

# Graphical summary of the model.
plt.scatter(x_days_train, y_confirmed_train, color='red')
plt.plot(x_days_train, lin_reg.predict(
    poly_reg.fit_transform(x_days_train)), color='blue')
plt.title("COVID-19 confimed cases of worldwide(Polynomial Regression Model)")
plt.xlabel('Days')
plt.ylabel('Total confirmed cases(#)')
plt.show()

# It seems that the polynomial line connects all the dots.
# Now, let's validate the model with the x_days_validate which contains index. For the index, the model
# will predict the confirmed cases of the data.
y_pred = lin_reg.predict(poly_reg.fit_transform(x_days_validate))
result = pd.DataFrame(y_pred)
result['Real Value'] = y_confirmed_validate.iloc[:, :].values
result['Predicted Value'] = pd.DataFrame(y_pred)
result = result[['Real Value', 'Predicted Value']]
print(result)

# Now, as per the model,let's check the accuracy of the model on validate data set.
print('Polynomial Regession  R2 Score   : ',
      r2_score(y_confirmed_validate, y_pred))
rmse = np.sqrt(mean_squared_error(y_confirmed_validate, y_pred))
print('RMSE of polynomial regression is :', rmse)

# However, let's apply the accurate model for test data.
y_confirmed_test_pred = lin_reg.predict(poly_reg.fit_transform(x_days_test))
predicted_result = pd.DataFrame(y_confirmed_test_pred)
predicted_result['Real Test Value'] = y_confirmed_test.iloc[:, :].values
predicted_result['Predicted Test Value'] = pd.DataFrame(y_confirmed_test_pred)
predicted_result = predicted_result[[
    'Real Test Value', 'Predicted Test Value']]
print(predicted_result)
