# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 17:36:39 2020

@author: admin
"""

# %% [markdown]
#  # COVID-19 Analysis ,visualization & Prediction

# %% [code]
# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# %% [code]
# Reading Data
data = pd.read_csv('./Corona-dataset/covid_19_data.csv',
                   parse_dates=['Last Update'])
data.rename(columns={'ObservationDate': 'Date',
                     'Country/Region': 'Country'}, inplace=True)

# Earliest Cases
data.head()

# %% [code]
data.shape

# %% [code]
data.dtypes

# %% [code]
# Missing Values
data.isnull().sum().to_frame('nulls')

# %% [markdown]
# # Total Confirmed , Deaths and Recovered cases in the world 21/03/2020

# %% [code]
df = data.groupby(["Date", "Country"])[
    ['Date', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
sorted_By_Confirmed = df.sort_values('Date', ascending=False)
sorted_By_Confirmed = sorted_By_Confirmed.drop_duplicates('Country')
sorted_By_Confirmed.head(80)

world_Confirmed_Total = sorted_By_Confirmed['Confirmed'].sum()
world_Deaths_Total = sorted_By_Confirmed['Deaths'].sum()
world_Recovered_Total = sorted_By_Confirmed['Recovered'].sum()

world_Deaths_rate = (world_Deaths_Total*100)/world_Confirmed_Total
world_Recovered_rate = (world_Recovered_Total*100)/world_Confirmed_Total

China = sorted_By_Confirmed[sorted_By_Confirmed['Country'] == 'Mainland China']
China_Recovered_rate = (
    int(China['Recovered'].values)*100)/int(China['Confirmed'].values)


veri = {'Total Confirmed cases  in the world': world_Confirmed_Total, 'Total Deaths cases in the world': world_Deaths_Total, 'Total Recovered cases in the world': world_Recovered_Total,
        'rate of Recovered Cases %': world_Recovered_rate, 'rate of death Cases %': world_Deaths_rate, 'rate of Recovered China cases %': China_Recovered_rate}
veri = pd.DataFrame.from_dict(veri, orient='index', columns=['Total'])

veri.style.background_gradient(cmap='Blues')

# %% [markdown]
# ### Graphic illustrates Total cases in the world

# %% [code]
veri = veri.head(3)
x = veri.index
y = veri['Total'].values
plt.rcParams['figure.figsize'] = (10, 6)
sns.barplot(x, y, order=x).set_title(
    'Total Cases / Deaths / Recovered')  # graf çizdir (Most popular)

# %% [markdown]
# # Coron Virus With Numbers
#
# Confirmed --              Cumulative number of confirmed cases till that date <br>
# Deaths --               Cumulative number of of deaths till that date <br>
# Recovered :              Cumulative number of recovered cases till that date <br>
# Recovered Cases Rate % -- rate of Recovered Cases from total of Confirmed cases in same Country <br>
# Deaths Cases Rate % --    rate of death Cases from total of Confirmed cases in same Country <br>
# Total Cases Rate % --     rate of total cases from Total cases in the world <br>

# %% [code]
Recovered_rate = (
    sorted_By_Confirmed['Recovered']*100)/sorted_By_Confirmed['Confirmed']
Deaths_rate = (sorted_By_Confirmed['Deaths']
               * 100)/sorted_By_Confirmed['Confirmed']
cases_rate = (sorted_By_Confirmed.Confirmed*100)/world_Confirmed_Total

sorted_By_Confirmed['Recovered Cases Rate %'] = pd.DataFrame(Recovered_rate)
sorted_By_Confirmed['Deaths Cases Rate %'] = pd.DataFrame(Deaths_rate)
sorted_By_Confirmed['Total Cases Rate %'] = pd.DataFrame(cases_rate)

print("Sorted By Confirmed Cases")
sorted_By_Confirmed.head(60).style.background_gradient(cmap='Reds')


# %% [markdown]
# ### Difference in the number of cases between 20/03 and 21/03

# %% [code]
df_Difference = data.groupby(["Date", "Country"])[
    ['Date', 'Country', 'Confirmed']].sum().reset_index()
sorted_By_Confirmed_Difference = df_Difference.sort_values(
    'Country', ascending=False)

x1 = sorted_By_Confirmed_Difference[sorted_By_Confirmed_Difference.Date ==
                                    '04/03/2020'].reset_index().drop('index', axis=1)
x2 = sorted_By_Confirmed_Difference[sorted_By_Confirmed_Difference.Date ==
                                    '04/04/2020'].reset_index().drop('index', axis=1)

h = pd.merge(x2, x1, on='Country')
h['cases_difference_03_04'] = h['Confirmed_y']-h['Confirmed_x']
h.sort_values('cases_difference_03_04', ascending=False).head(
    60).style.background_gradient(cmap='Greens')

# %% [markdown]
# # Top 10  infected Countries

# %% [code]
sorted_By_Confirmed1 = sorted_By_Confirmed.sort_values(
    'Confirmed', ascending=False)
sorted_By_Confirmed1 = sorted_By_Confirmed1.head(10)
x = sorted_By_Confirmed1.Country
y = sorted_By_Confirmed1.Confirmed
plt.rcParams['figure.figsize'] = (12, 10)
sns.barplot(x, y, order=x, palette="rocket").set_title(
    'Total Cases / Deaths / Recovered')  # graf çizdir (Most popular)

# %% [markdown]
# ### Cases Rate per country of total cases in the world
#

# %% [code]
Top7 = sorted_By_Confirmed.iloc[0:8, -1].values
others = sorted_By_Confirmed.iloc[8:, -1].sum()
x = np.array(Top7)
x2 = np.array(others)
rates = np.concatenate((x, x2), axis=None)

rate_perCountry = pd.DataFrame(
    data=rates, index=[sorted_By_Confirmed['Country'].head(9)], columns=['rate'])
rate_perCountry.rename(index={'Switzerland': "other Countries"}, inplace=True)


labels = rate_perCountry.index
sizes = rate_perCountry['rate'].values

explode = None  # explode 1st slice
plt.subplots(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels,
        autopct='%1.1f%%', shadow=False, startangle=0)
plt.axis('equal')
print("cases rate per country of total cases in the world ")
plt.show()

# %% [markdown]
# ## Table that illustrates Increasing infections cases in the world per day .

# %% [code]
cases_per_Day = data.groupby(
    ["Date"])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
sorted_By_Confirmed1 = cases_per_Day.sort_values('Date', ascending=False)

sorted_By_Confirmed1.style.background_gradient(cmap='Reds')

# %% [markdown]
# ## Graphic that illustrates Increasing infections cases per day

# %% [code]
x = cases_per_Day.index

y = cases_per_Day.Confirmed
y1 = cases_per_Day.Deaths
y2 = cases_per_Day.Recovered

plt.scatter(x, y, color='blue', label='Confirmed Cases')
plt.scatter(x, y1, color='red', label="Deaths Cases")
plt.scatter(x, y2, color='green', label="Recovered Cases")
print("Blue : Confirmed Cases ")
print("Red : Deaths Cases ")
print("Green : Recovered Cases ")
plt.show()

# %% [markdown]
# # Prediction Future cases

# %% [code]
# Train & Test Data
x_data = pd.DataFrame(cases_per_Day.index)
x_data = x_data.drop([21, 22, 23, 24, 25, 26, 27], axis=0)
y_data = pd.DataFrame(cases_per_Day.Confirmed)
y_data = y_data.drop([21, 22, 23, 24, 25, 26, 27], axis=0)

# %% [code]
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.1, random_state=0)

# %% [markdown]
# # Polynomial Regression to predict future cases .

# %% [code]
# Polynomal Regression (degree=5)

poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x_train)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y_train)

# %% [markdown]
# ### Model Graphic

# %% [code]
plt.scatter(x, y, color='red')
plt.plot(x_test, lin_reg2.predict(
    poly_reg.fit_transform(x_test)), color='blue')
plt.title("Polynomial Regression Model ")
plt.show()

# %% [markdown]
# ## Test algorithm

# %% [code]
y_pred = lin_reg2.predict(poly_reg.fit_transform(x_test))

result = pd.DataFrame(y_pred)
result['Real Value'] = y_test.iloc[:, :].values
result['Predicted Value'] = pd.DataFrame(y_pred)
result = result[['Real Value', 'Predicted Value']]
result

# %% [code]


print('Polynomial Regession  R2 Score   : ', r2_score(y_test, y_pred))

# %% [markdown]
# # Make Predict For Future Days

# %% [code]
# today is 03/21/2020
print("After {0} day will be {1} case in the world".format(
    (76-len(cases_per_Day)), lin_reg2.predict(poly_reg.fit_transform([[76]]))))
print("After {0} day will be {1} case in the world".format(
    (67-len(cases_per_Day)), lin_reg2.predict(poly_reg.fit_transform([[67]]))))
print("After {0} day will be {1} case in the world".format(
    (77-len(cases_per_Day)), lin_reg2.predict(poly_reg.fit_transform([[77]]))))
print("After {0} day will be {1} case in the world".format(
    (87-len(cases_per_Day)), lin_reg2.predict(poly_reg.fit_transform([[87]]))))
print("After {0} day will be {1} case in the world".format(
    (97-len(cases_per_Day)), lin_reg2.predict(poly_reg.fit_transform([[97]]))))
print("After {0} day will be {1} case in the world".format(
    (107-len(cases_per_Day)), lin_reg2.predict(poly_reg.fit_transform([[107]]))))

# %% [markdown]
# ## Linear regression to predict future cases

# %% [code]
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

# %% [code]
y_pred2 = lin_reg.predict(x_test)

result2 = pd.DataFrame(y_pred2)
result2['Real Value'] = y_test.iloc[:, :].values
result2['Predicted Value'] = pd.DataFrame(y_pred2)
result2 = result2[['Real Value', 'Predicted Value']]

# Accuracy

print('Linear regession  R2 Score : ', r2_score(y_test, y_pred2))
result.head(len(y_pred2))

# %% [code]
plt.scatter(x, y, color='red')
plt.plot(x_test, y_pred2, color='blue')
plt.show()

# %% [markdown]
#  If you like my Kernel,vote it please ... Thanks
