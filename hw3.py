# Sejuti Mannan
# NYU Tandon Data Science Bootcamp
# Week 3 (10/16) Exploratory Data Analysis and Data Visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Filter the data to include only weekdays (Monday to Friday) and
# plot a line graph showing the pedestrian counts for each day of the
# week.

url = "https://data.cityofnewyork.us/api/views/6fi9-q3ta/rows.csv?accessType=DOWNLOAD"
df = pd.read_csv(url)
df = df.sort_values(by='hour_beginning')
pd.set_option('display.max_columns', None)

# Add day of the week
df['hour_beginning'] = pd.to_datetime(df['hour_beginning'], errors='coerce')
df['weekday_index'] = df['hour_beginning'].dt.dayofweek

weekdays = df[(df['hour_beginning'].dt.dayofweek >= 0) & (df['hour_beginning'].dt.dayofweek <= 4)]
ped_counts_per_day = weekdays.groupby(weekdays['hour_beginning'].dt.day_name())['Pedestrians'].sum()

# Reindex to ensure Monday to Friday order
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
ped_counts_per_day = ped_counts_per_day.reindex(weekday_order)

plt.figure(figsize=(12, 6))
ped_counts_per_day.plot(kind='line', marker='o')
plt.title('Pedestrian Counts During Weekdays')
plt.xlabel('Weekday')
plt.ylabel('Pedestrian Count')
plt.grid(True)
plt.tight_layout()
plt.savefig('pedestrian_count_during_weekday.png', dpi=300, bbox_inches='tight')

# 2. Track pedestrian counts on the Brooklyn Bridge for the year 2019
# and analyze how different weather conditions influence pedestrian
# activity in that year. Sort the pedestrian count data by weather
# summary to identify any correlations( with a correlation matrix)
# between weather patterns and pedestrian counts for the selected year.
#
# -This question requires you to show the relationship between a
# numerical feature(Pedestrians) and a non-numerical feature(Weather
# Summary). In such instances we use Encoding. Each weather condition
# can be encoded as numbers( 0,1,2..). This technique is called One-hot
# encoding.
#
# -Correlation matrices may not always be the most suitable
# visualization method for relationships involving categorical
# datapoints, nonetheless this was given as a question to help you
# understand the concept better.

df['date'] = df['hour_beginning'].dt.date
df_2019 = df[(df['hour_beginning'].dt.year == 2019) & (df['location'] == 'Brooklyn Bridge')]
weather_summary_encoded = pd.get_dummies(df_2019['weather_summary'])

# Concatenate the one-hot encoded columns with the original DataFrame (axis=1 means column-wise)
df_2019_encoded = pd.concat([df_2019[['Pedestrians']], weather_summary_encoded], axis=1)

correlation_matrix = df_2019_encoded.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Encoded Weather Summary and Pedestrian Counts')
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')

# 3. Implement a custom function to categorize time of day into morning,
# afternoon, evening, and night, and create a new column in the
# DataFrame to store these categories. Use this new column to analyze
# pedestrian activity patterns throughout the day.
def time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'

df['time_of_day'] = df['hour_beginning'].dt.hour.apply(time_of_day)
ped_counts_per_time_of_day = df.groupby('time_of_day')['Pedestrians'].sum()
plt.figure(figsize=(12, 6))

ped_counts_per_time_of_day.plot(kind='bar', color='orange')
plt.title('Total Pedestrian Counts by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Pedestrian Count')
plt.tight_layout()
plt.savefig('time_of_day.png', dpi=300, bbox_inches='tight')