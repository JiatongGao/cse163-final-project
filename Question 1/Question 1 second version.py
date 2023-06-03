import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("weatherAUS.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
avg_temp = data.groupby(['Location', 'Year'])['MaxTemp'].mean().reset_index()



# Calculate the percentage bigger than 30C 
hot_days = data[data['MaxTemp'] > 30].groupby(['Location', 'Year']).size().reset_index(name='HotDays')
total_days = data.groupby(['Location', 'Year']).size().reset_index(name='TotalDays')
hot_days_percent = hot_days.merge(total_days, on=['Location', 'Year'])
hot_days_percent['Percentage'] = (hot_days_percent['HotDays'] / hot_days_percent['TotalDays']) * 100

cities = ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Canberra', 'Adelaide']

# Plot average temperature for each city
fig, axs = plt.subplots(len(cities), 1, figsize=(10, 8), sharex=True)

for i, city in enumerate(cities):
    city_avg_temp = avg_temp[avg_temp['Location'] == city]
    ax = axs[i]
    ax.bar(city_avg_temp['Year'], city_avg_temp['MaxTemp'])
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"{city} - Average Temperature")

axs[-1].set_xlabel("Year")
fig.tight_layout()
plt.savefig("average_temperature_plot.png")

# Save the average temperature plot to a file
plt.savefig("average_temperature_plot.png")

# Plot percentage of hot days for each city
fig, axs = plt.subplots(len(cities), 1, figsize=(10, 8), sharex=True)

for i, city in enumerate(cities):
    city_hot_days_percent = hot_days_percent[hot_days_percent['Location'] == city]
    ax = axs[i]
    ax.plot(city_hot_days_percent['Year'], city_hot_days_percent['Percentage'])
    ax.set_ylabel("Percentage")
    ax.set_title(f"{city} - Percentage of Hot Days")

axs[-1].set_xlabel("Year")
fig.tight_layout()
plt.savefig("percentage_hot_days_plot.png")
