import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
from plotly.subplots import make_subplots
import chart_studio.plotly as py
import chart_studio.tools as tls

def plot_line(dates, true_values, predicted_values):
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           subplot_titles=('True Values', 'Predicted Values'))

    fig.add_trace(go.Scatter(x=dates, y=true_values, mode='lines', name='True Values', line=dict(color='blue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=predicted_values, mode='lines', name='Predicted Values',
                             line=dict(color='red')), row=2, col=1)

    fig.update_layout(xaxis=dict(title='Date'), title='True Values vs Predicted Values')
    fig.show()

