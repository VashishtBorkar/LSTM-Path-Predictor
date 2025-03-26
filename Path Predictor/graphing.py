import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from data_prcoessing import create_sequences_from_trial

# Function to graph the actual trial compared to the prediction
def graph_trial(model, test_sequence, sequence_length):
    
    # Create sequences of the data
    x_data, y_data = create_sequences_from_trial(test_sequence, sequence_length)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Make a prediction with untrained data
    prediction = model.predict(x_data)
    values = y_data

    # Set x and z values to be graphed for prediction and actual
    prediction = np.array(prediction)
    prediction_x = prediction[:, 0]
    prediction_z = prediction[:, 1]

    values = np.array(values)
    values_x = values[:, 0]
    values_z = values[:, 1]

    graph_df = pd.DataFrame()
    graph_df['prediction_x'] = prediction_x
    graph_df['prediction_z'] = prediction_z
    graph_df['values_x'] = values_x
    graph_df['values_z'] = values_z

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prediction_x, y=prediction_z, name='Predicted Path', mode='markers'))
    fig.add_trace(go.Scatter(x=values_x, y=values_z, name='Actual Path', mode='markers'))
    fig.update_xaxes(range=[-2,2])
    fig.update_yaxes(range=[-1,6])
    fig.update_layout(
        title='Predicted Path vs Actual Path',
        xaxis_title='X Position',
        yaxis_title='Z Position',
        height= 1000
    )
    fig.show()