import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split    
# K folds cross validation
def load_data(filename):
    raw_data = pd.read_csv(filename)
    # List of available variables
    available_variables =   [   
                                'timeSinceStart', 'trial_id', 'room_id', 'pos_x', 'pos_z', 'Participant_ID', 
                                'gate', 'distanceToGate', 'front_row_pos_x', 'back_row_pos_x', 'distance_to_front_1', 
                                'distance_to_front_2', 'distance_to_front_3', 'distance_to_front_4', 'distance_to_front_5', 
                                'distance_to_back_1', 'distance_to_back_2', 'distance_to_back_3', 'distance_to_back_4', 
                                'distance_to_back_5', 'row_direction', 'WorldPosX', 'WorldPosY', 'WorldPosZ' 
                            ]
    raw_data = raw_data[available_variables]
    
    raw_data.rename(columns={'timeSinceStart' : 'time_since_start', 'Participant_ID' : 'participant_id', 
                                    'distanceToGate' : 'distance_to_gate'}, inplace=True)
    return raw_data

def preprocess_data(raw_data):
    # Set the categorical and continuous variables by creating list of names of variables being used for input and output
    cat_vars_list = ['gate', 'row_direction'] # List of categorical variables (Features can be added when more data is gathered)
    cont_vars_list = ['time_since_start', 'distance_to_gate'] # List of continuous variables (Features can be added when more data is gathered)
    pos_labels = ['pos_x', 'pos_z'] 

    # Information used for labels (coordinates)
    label_vars = raw_data[pos_labels]

    # Use one hot encodng for categorical variables
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    categorical_vars = one_hot_encoder.fit_transform(raw_data[cat_vars_list]) # Can add categorical variables if necessary

    # Normalize path data 
    scaler = MinMaxScaler()
    continuous_vars = scaler.fit_transform(raw_data[cont_vars_list]) # Can add continuous variables if necesssary

    # Combine the continous and categorical features
    combined_features = np.hstack([categorical_vars, continuous_vars, label_vars])

    # Create df and add participant and trial ids
    combined_df = pd.DataFrame(combined_features)
    combined_df['trial_id'] = raw_data['trial_id']
    combined_df['participant_id'] = raw_data['participant_id']

    # Create df with data split up into each trial
    combined_trial_data = [] # Will become a 3D array of the form, trial, sequence, features

    # Split df by participant id and trial id to get each individual trial
    # Must be split up this way in order to make sequences based on each trial
    grouped = combined_df.groupby(['participant_id', 'trial_id'])

    for (p_id, t_id), data in grouped:
        trial_data = data.drop(['participant_id', 'trial_id'], axis=1)
        seq = np.array(trial_data)
        combined_trial_data.append(seq)
        
    for index,trials in enumerate(combined_trial_data):
        combined_trial_data[index] = np.squeeze(trials)

    print("Number of trials of data: ", len(combined_trial_data))
    print("Example number of sequences (changes per trial): ",len(combined_trial_data[0]))
    print("Number of features: ",len(combined_trial_data[0][0]))

    return combined_trial_data

def create_sequences_from_trial(trial_data, seq_length):
    features, labels = [], []
    filler_data = trial_data[0]
    filler_list = [filler_data]*seq_length
    filler_list = np.array(filler_list)
    trial_data = np.vstack([filler_list, trial_data])
    for i in range(seq_length, len(trial_data)):
        feat = np.array(trial_data[i-seq_length:i])
        feat = feat[:, :-2]
        label = np.array(trial_data[i])
        label = label[-2:]
        features.append(feat)
        labels.append(label)
    return features, labels

def create_list_of_sequences(preprocessed_data, sequence_length):
    # Create an array of all of the sequences and their labels
    x_data, y_data = [], []
    for trials in preprocessed_data: 
        features, labels = create_sequences_from_trial(trials, sequence_length)
        for feat in features:
            x_data.append(feat)
        for label in labels:
            y_data.append(label)

    return np.array(x_data),np.array(y_data)

def create_training_testing_split(x_data, y_data, test_size):

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size)

    # Print shape of data
    print("X Training Data Shape: ", x_train.shape)
    print("Y Training Data Shape: ", y_train.shape)

    print("X Test Data Shape: ", x_test.shape)
    print("Y Test Data Shape: ", y_test.shape)

    return x_train, x_test, y_train, y_test


