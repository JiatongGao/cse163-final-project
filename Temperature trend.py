import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("weatherAUS.csv")
# set up which city we want to study
# and what temperature is considered as extreme temperature
high_temp = 30
city = 'CoffsHarbour'

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df = df[(df['Location'] == city)]
grouped = df.groupby('Year')['MaxTemp'].agg(['count', lambda x: (x > high_temp).sum()])
grouped['Percentage'] = grouped['<lambda_0>'] / grouped['count'] * 100

grouped.rename(columns={'<lambda_0>': 'AboveTemp'}, inplace=True)
grouped.reset_index(inplace=True)
df = pd.merge(df, grouped[['Year', 'Percentage']], on='Year', how='left')
df = df.loc[:, ['Year', 'Percentage']]
df_unique = df.drop_duplicates()
print(df_unique)
sns.relplot(data=df_unique, x="Year", y="Percentage",kind="line")
plt.ylabel('Percentage')
plt.title("Percentage of extreme temperature for " + city + " weather over years")
plt.savefig('plot_test.png', bbox_inches='tight')

# Merge the aggregated values back into the original DataFrame
#df = df.merge(df_agg, on='Year', suffixes=('', '_aggregated'))
#df_unique = df.drop_duplicates()
#print(df_unique)
#sns.relplot(data=df_unique, x="Year", y="MaxTemp_aggregated",kind="line")
#plt.ylabel('Maximum temperature')
#plt.title("Maximum temperature for" + city + " weather over years")
#plt.savefig('plot_test.png', bbox_inches='tight')
