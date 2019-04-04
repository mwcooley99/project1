def helper():
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

    return(merged_df)

merged_df = helper()