import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chart_studio.plotly as py
import chart_studio.tools as tls

data = pd.read_csv("weatherAUS.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
avg_temp = data.groupby(['Location', 'Year'])['MaxTemp'].mean().reset_index()

# Calculate the percentage bigger than 30C 
hot_days = data[data['MaxTemp'] > 30].groupby(['Location', 'Year']).size().reset_index(name='HotDays')
total_days = data.groupby(['Location', 'Year']).size().reset_index(name='TotalDays')
hot_days_percent = hot_days.merge(total_days, on=['Location', 'Year'])
hot_days_percent['Percentage'] = (hot_days_percent['HotDays'] / hot_days_percent['TotalDays']) * 100

cities = avg_temp['Location'].unique()

# Calculate the number of rows and columns
num_rows = (len(cities) + 2) // 3  # Round up division, with a minimum of one row
num_cols = 3

# Create subplots with two subplots per city
fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=cities, specs=[[{'secondary_y': True}, {'secondary_y': True}, {'secondary_y': True}]] * num_rows)

# Traverse each city
row = 1
col = 1
scatter_visible = []
bar_visible = []
for city in cities:
    # Get the average temperature data for the current city
    city_avg_temp = avg_temp[avg_temp['Location'] == city]

    # Get the percentage of days with temperature above 30 degrees for the current city
    city_hot_days_percent = hot_days_percent[hot_days_percent['Location'] == city]

    # Add bar chart for average temperature
    fig.add_trace(go.Bar(x=city_avg_temp['Year'], y=city_avg_temp['MaxTemp'], name='Avg Temp'), row=row, col=col, secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", row=row, col=col, secondary_y=False)
    bar_visible.append(True)
    scatter_visible.append(False)

    # Add scatter chart for percentage of hot days
    fig.add_trace(go.Scatter(x=city_hot_days_percent['Year'], y=city_hot_days_percent['Percentage'], mode='lines', name='Hot Days %'), row=row, col=col, secondary_y=True)
    fig.update_yaxes(title_text="Percentage", row=row, col=col, secondary_y=True)
    scatter_visible.append(True)
    bar_visible.append(False)

    # Set y-axis titles for both subplots
   
    

    col += 1
    if col > num_cols:
        col = 1
        row += 1

buttons = [
    dict(
        label="Percentage",
        method="update",
        args=[{"visible": scatter_visible}, {"title": "Percentage of days temperature larger than 30 in a year"}]
    ),
    dict(
        label="Temperature",
        method="update",
        args=[{"visible": bar_visible}, {"title": "Averge temperature by year"}]
    )
]

updatemenu = [
    dict(
        buttons=buttons,
        direction="down",
        showactive=True,
        x=0.5,  # Adjust the horizontal position of the buttons
        xanchor="center",
        y=1.05,  # Adjust the vertical position of the buttons
        yanchor="top"
    ),
]

# Update the layout of the figure
# Update the layout of the figure
fig.update_layout(
    height=3500,
    width=2000,
    title_text='Temperature Analysis',
    updatemenus=updatemenu,
    plot_bgcolor='rgb(255, 228, 225)',  # Set plot background color to light blue
    paper_bgcolor='rgb(204, 229, 255)',  # Set paper background color to light blue
    font=dict(family='Arial', size=18, color='black'),  # Set font family, size, and color
    margin=dict(t=200, b=100),  # Adjust top and bottom margins
    legend=dict(font=dict(size=16)),  # Set legend font size
    xaxis=dict(
        title='Year',
        title_font=dict(size=8),
        tickfont=dict(size=16),
    ),
    yaxis=dict(
        title='Temperature (°C)',
        title_font=dict(size=8),
        tickfont=dict(size=8),  
    ),
    yaxis2=dict(
        title='Percentage',
        title_font=dict(size=8),
        tickfont=dict(size=8),  
    )
)

# Show the figure


username = 'JiatongGao'
api_key = '0VSlIry9JOD1DGbpsPWD'

tls.set_credentials_file(username=username, api_key=api_key)
url = py.plot(fig, filename='weather_analysis', auto_open=False)
plot_websit = "https://plotly.com/~JiatongGao/1/"
print("website of plot", url)
with open("Question1_plot", "w") as file:
    file.write(plot_websit)