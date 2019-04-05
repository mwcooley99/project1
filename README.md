
# Final Analysis of Vaccination Rates
Vaccination rates have begun to fall in areas of Califonia. This has led to outbreaks of previously erradicated diseases. This isn't just a problem in California so it would be helpful to identify demographic indicators that could help us target where outbreaks are more likely to occur, allowing targeted educaitonal campaigns to combat misinformation about vaccinations.
## Exploration Questions
What does the vaccination rate look like accross California?

What relationship (if any) do the folowing variables have on the vaccination rates?accross countries.
- Income
- Unemployment
- County Education Level
- Insured Rate
- County Population

## Data being used
- Kindergarten Immunization records from Kaggle [link](https://www.kaggle.com/broach/california-kindergarten-immunization-rates)
- CA census estimates currated by the state [link](http://www.dof.ca.gov/Reports/Demographic_Reports/American_Community_Survey/#ACS2017x5)

# Findings

## What does the vaccination rate look like around California?
- There is a relatively large range of vaccination rates accross California Counties. The difference from the lowest to highest county is about 20%
- The mean and median are in the 95-96% range. This is good for herd immunity, but the Std Dev of 3 means that there are a significant number of counties that are likely below the threshholds for herd immunity needed for contagious diseases like the measles.

## Education level
- There appears to be correlation between a counties percentage of residents with 'No High School Diploma' and the vaccination rate.
    - The high r-squared value implies that this isn't predictive. 
    - It could also be an artifact.
    - We're looking at a .38% increase in percent vacinated for every 1% increase in the percent of high school dropouts.
- Interestingly this correlation seems to dissapear at any other educational milestone.


## Other Demographic indicators
- None of the other indicators I looked at appear to correlate with the vaccination rate.

# Next Steps

## Disaggregate data by city
Noting the large variance in county size, it would be interesting to see if looking at the same data at the city level would lend any new insghts.

## Other demographic indicators
While dissagregating the data might yields some interesting insights, we might just be missing the relevant indicators. Some interesting data to include for further analysis might be:
- Number of members in households
- Ethnicity
- Religion

# Work
## Calculations
- exporatory_analysis.ipnyb
## Final Anaylsis - A writeup
- final_anaylsis.ipynb