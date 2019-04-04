# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] {"pycharm": {}}
# # Data Exploration and Clean up
# Looking at Vaccination rates per county and county cross-referenced with demographic data to look for any correlations.
# ## Exploration Questions
# What does the vaccination rate look like accross California
#
# What relationship (if any) do the folowing variables have on the vaccination rates accross countries.
# - Median Income
# - County Education Level
# - Unemployment Level
# - Percent Uninsured
# - County Population
#
# ## Data being used
# - Kindergarten Immunization records from Kaggle [link](https://www.kaggle.com/broach/california-kindergarten-immunization-rates)
# - CA census estimates currated by the state [link](http://www.dof.ca.gov/Reports/Demographic_Reports/American_Community_Survey/#ACS2017x5)

# %% [markdown] {"pycharm": {}}
# # Imports

# %% [markdown] {"pycharm": {}}
# ## Modules and Environment

# %% {"pycharm": {"is_executing": false}}
# Calculation
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Additional Modules
import subprocess
from functools import reduce

get_ipython().run_line_magic('config', 'IPCompleter.greedy = True')

# %% [markdown] {"pycharm": {}}
# ## Import data and basic cleaning of Data

# %% [markdown] {"pycharm": {}}
# ### Immunization Records

# %% [markdown] {"pycharm": {}}
# #### Key for the table
# - schoolType - Public/Private indicator
# - COUNTY - name of county in CA
# - SCHOOL - String label of school (not always consistent across years)
# - school_code - Unique integer code for each school (consistent across years)
# - n - Number of students
# - nMMR - Number of students reporting complete MMR vaccination
# - nDTP - Number of students reporting DTP vaccination
# - nPolio - Number of students reporting Polio vaccination
# - nPBE - Number of students reporting personal beliefs exemption*
# - nPME - Number of students reporting permanent medical exemption
# - year - Calendar year (2000:2014) where 2000=2000-2001 school year, 2001=2001-2002 school year, etc.

# %% {"pycharm": {"is_executing": false}}
# Import immunization records
student_df = pd.read_csv('data/california-kindergarten-immunization-rates/StudentData.csv', sep=None, engine='python')
student_df.head()

# %% {"pycharm": {}}
# Narrow down to the years we have demo data
student_df = student_df[(student_df['year'] > 2009) & (student_df['year'] < 2015)]
student_df['year'].unique()

# %% {"pycharm": {"is_executing": false}}
# Aggregate the student_df
student_df['exemptions'] = student_df['nPBE'] + student_df['nPME']
student_agg = student_df.groupby('COUNTY')['n', 'exemptions'].sum()

# Format it
student_agg = student_agg.reset_index()
student_agg = student_agg.rename(columns={'COUNTY': 'County'})
student_agg['County'] = student_agg['County'].str.title()

# Find the percent vacinated
student_agg['Percent Vaccinated'] = round((student_agg['n'] - student_agg['exemptions']) / student_agg['n'] * 100, 2)

display(student_agg.head())
display(student_agg.describe())


# %% [markdown] {"pycharm": {}}
# ### Demographic Data

# %% {"pycharm": {}}
def filter_by_county(df):
    new_df = df.loc[df['Geography'].str.contains('County')]
    new_df = new_df.loc[~new_df['Geography'].str.contains('\(')]
    new_df['Geography'] = new_df['Geography'].str.replace(' County', '')
    new_df = new_df.rename(columns={'Geography': 'County'})
    return new_df


# %% {"pycharm": {}}
# Import Education Data
education_df = pd.read_excel('data/Web_ACS2014_10_Educ.xlsx', sheet_name='Educational Attainment', header=[4,5])

# Clean up the cols
education_df = education_df.rename(columns={'Unnamed: 0_level_0': 'Geography'})
education_df = education_df[["Geography", "Percent Less than 9th grade", "Percent 9th to 12th grade, no diploma", "Percent High school graduate (includes equivalency)",
                             "Percent Bachelor's degree", "Percent Graduate or professional degree",
                             "Percent high school graduate or higher", "Percent bachelor's degree or higher"]]
education_df = education_df.drop(columns=['Margin of Error', 'Margin of Error.1', 'Summary Level', 'County', 'Place'], level=1)

education_df.columns = ['Geography', '9th or less', '9th-12th', "High School Graduate", 'Bachelors', 'Graduate Degree', 'High School or Higher', 'Bachelors or Higher']
education_df['No High School Diploma'] = education_df['9th or less'] + education_df['9th-12th']
education_df = education_df.drop(columns=['9th or less', '9th-12th'])
# Filter down to the county level
education_df = filter_by_county(education_df)

education_df.head()

# %% {"pycharm": {}}
education_df.info()

# %% {"pycharm": {}}
# Import Health Insurance Data
health_ins_df = pd.read_excel('data/Web_ACS2014_10_HealthIns.xlsx', sheet_name='Health Insurance', header=[4,5])
health_ins_df = health_ins_df.drop(columns=['Estimate Margin of Error', 'Percent Margin of Error'], level=1)

health_ins_df = health_ins_df.iloc[:, :4]
health_ins_df.columns = ['Geography', 'Population', 'Number Insured', 'Percent Insured']
health_ins_df = health_ins_df.drop(columns='Number Insured')
health_ins_df = filter_by_county(health_ins_df)

health_ins_df.shape

# %% {"pycharm": {}}
# Import Income Data
income_df = pd.read_excel('data/Web_ACS2014_10_Inc-Pov-Emp.xlsx', sheet_name='Income', header=[3,4])
income_df = income_df[[('Unnamed: 0_level_0', 'Geography'),
                                     ('Median household income (dollars)', 'Estimate'),
                                     ('Mean household income (dollars)', 'Estimate'),
                                     ('Per capita income (dollars)', 'Estimate')]]
income_df.columns = ['Geography', 'Median Income', 'Mean Income', 'Per capita income']
income_df = filter_by_county(income_df)
income_df['Mean - Median'] = income_df['Mean Income'] - income_df['Median Income']
income_df.head()

# %% {"pycharm": {}}
# Import Unemployment Data
unemployment_df_raw = pd.read_excel('data/Web_ACS2014_10_Inc-Pov-Emp.xlsx', sheet_name='Employment Status', header=[3,4,5,6])
unemployment_df = pd.DataFrame()
unemployment_df= unemployment_df_raw.iloc[:,[0, 17]]
unemployment_df.columns = ['Geography', 'Unemployment Percentage']

unemployment_df = filter_by_county(unemployment_df)
unemployment_df.shape

# %% [markdown]
# ### Merge data into one frame for analysis

# %%
dfs = [student_agg, education_df, health_ins_df, income_df, unemployment_df]

merged_df = reduce(lambda left, right: pd.merge(left, right, on='County'), dfs)

# Convert non-numerical types that should be numbers
merged_df.loc[:, merged_df.columns != 'County'] = merged_df.loc[:, merged_df.columns != 'County'].apply(pd.to_numeric)
merged_df = merged_df.sort_values(by='Percent Vaccinated')

merged_df.head()

# %%
# merged_df['High School or Higher'] = merged_df['High School or Higher'].astype('float64')

merged_df.info()

# %% [markdown]
# # Analysis

# %% [markdown]
# ## Overview of Data

# %%
# Get some info about the data set
merged_df.info()

# %%
# Get some basic stats about the data
merged_df.describe()

# %%
# Some visualizations of the vacination rates
sns.boxplot(x='Percent Vaccinated', data=merged_df)

# %%
# Some Descriptive data and Probabability density function, looking for outliers
num_of_bins = 20
print(merged_df['Percent Vaccinated'].describe())
sns.distplot(merged_df['Percent Vaccinated'], bins=num_of_bins)

# %%
# Look at the least vaccinated counties
merged_df.head()

# %%
fig, ax = plt.subplots(figsize=(10,8))
sns.barplot(x='County', y='Population', data=merged_df, ax=ax)
ax.tick_params(rotation=90)
ax.set_title('County Populations')
fig.tight_layout()

# %% [markdown]
# ## Visualizations Explorations

# %%
# Look at bubble plots for different independent variables
ind_cols = merged_df.columns[4:]
ind_cols

for idx in range(0, len(ind_cols), 4):
    sns.pairplot(merged_df, y_vars='Percent Vaccinated', x_vars=ind_cols[idx: idx+4], kind='reg',height=5)

# %% [markdown]
# ## Education Analysis

# %%
# Look at linear regression for the High Schol or Higher group
X = merged_df['High School or Higher']
Y = merged_df['Percent Vaccinated']

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predication = model.predict(X)

print_model = model.summary()
print(print_model)

# %%
# Look at Bachelors or higher
X = merged_df['Bachelors or Higher']
Y = merged_df['Percent Vaccinated']

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predication = model.predict(X)

print_model = model.summary()
print(print_model)

# %% [markdown]
# #### Findings
# What does it mean to have a low p-value and a low r-squared value for High School or Higher Education regression model
#     - There seems to be a correlation but there's also a lot of variance hurting the predictive capabilities of the model

# %% [markdown]
# ### Considering Bachelors or Higher has no correlation, but High School or Higher does. Let's look at them on a more individual basis.

# %%
# High School Diploma
X = merged_df['No High School Diploma']
Y = merged_df['Percent Vaccinated']

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predication = model.predict(X)

print_model = model.summary()
print(print_model)

# %%
# High School Diploma
X = merged_df['High School Graduate']
Y = merged_df['Percent Vaccinated']

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predication = model.predict(X)

print_model = model.summary()
print(print_model)

# %%
# Bachelors
X = merged_df['Bachelors']
Y = merged_df['Percent Vaccinated']

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predication = model.predict(X)

print_model = model.summary()
print(print_model)

# %%
# Graduate
X = merged_df['Graduate Degree']
Y = merged_df['Percent Vaccinated']

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predication = model.predict(X)

print_model = model.summary()
print(print_model)

# %% [markdown]
# There doesn't appear to be any correlation with any of the individual education ranges.

# %% [markdown]
# #### Findings
# There is a significant **positive** correlation between the percentage of the population no High School Diploma and percent vacinated.

# %% [markdown]
# ## Insurance

# %%
# Percent Insured
X = merged_df['Percent Insured']
Y = merged_df['Percent Vaccinated']

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predication = model.predict(X)

print_model = model.summary()
print(print_model)

# %%
# Unemployment Rate
X = merged_df['Unemployment Percentage']
Y = merged_df['Percent Vaccinated']

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predication = model.predict(X)

print_model = model.summary()
print(print_model)

# %% [markdown]
# #### Finding: Suprisingly enough, there appears to be no correlation between the uninsured rate and percent vaccinated

# %%
# Unemployment Rate
X = merged_df['Mean - Median']
Y = merged_df['Percent Vaccinated']

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predication = model.predict(X)

print_model = model.summary()
print(print_model)

# %% [markdown]
# ## Multi-variant Analysis

# %%
# Lets look at median income coupled with no High School Diploma
# Unemployment Rate
X = merged_df[['No High School Diploma', 'Median Income']]
Y = merged_df['Percent Vaccinated']

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predication = model.predict(X)

print_model = model.summary()
print(print_model)


# %% [markdown]
# # Findings and Next Steps
# ## Findings
# The following demographic indicators appear to have not correlation or predictive capability for the level of vaccination:
# - Income
# - Employment Rate
# - Insured Rate
#
# Education seems to have some correlation with the uninsured rate. Interestingly though the only significant indicator seems to be the percent of the population with *No High School Diploma*.
#
