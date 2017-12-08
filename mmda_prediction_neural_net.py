import pandas as pd
import numpy as np
import math
from random import uniform
from neural_network import neural_network as net

TRAIN_DATA_X = 'data/x_train.csv'
TRAIN_DATA_Y = 'data/y_train.csv'
TEST_DATA_X = 'data/x_test.csv'
TEST_DATA_Y = 'data/y_test.csv'

NORMALIZE_COLUMNS = ['NumDeath', 'NumInjured', 'NumVehInteraction']

def normalize_df(df_base, columns):
    df = df_base.copy()
    for column in columns:
        df[column] = normalize_max_unknown(df[column])
    
    return df

def normalize_max_unknown(vector):
    min = np.min(vector)
    max = np.max(vector)
        
    return [(x - float(min)) / (float(max) - float(min)) for x in vector]

train_x_df = pd.read_csv(TRAIN_DATA_X)
train_y_df = pd.read_csv(TRAIN_DATA_Y)

df_y = train_y_df['Classification']
df_x = train_x_df[['NumDeath', 'NumInjured', 'NumPedestrianVictim', 'NumVehInteraction', 'PassengerInjured',
                   'PassengerKilled', 'PedestrianInjured','PedestrianKilled','DriversInjured','DriversKilled']]

df_x = normalize_df(df_x, NORMALIZE_COLUMNS)

dataset_df = pd.concat([df_x, df_y], axis=1)

#Partition dataset based on classification
class_one_df = dataset_df.loc[dataset_df['Classification'] == 1]
class_two_df = dataset_df.loc[dataset_df['Classification'] == 2]
class_three_df = dataset_df.loc[dataset_df['Classification'] == 3]

#Extract and one hot classification
class_one_y = pd.get_dummies(class_one_df['Classification'])
class_two_y = pd.get_dummies(class_two_df['Classification'])
class_three_y = pd.get_dummies(class_three_df['Classification'])

#Remove classification column
class_one_df = class_one_df.drop('Classification', axis=1)
class_two_df = class_two_df.drop('Classification', axis=1)
class_three_df = class_three_df.drop('Classification', axis=1)

#Convert dataframes to lists
df_values_arr = [class_one_df.values, class_two_df.values, class_three_df.values]

#Create and train autoencoder model for each dataset partition
activation_scheme = ['Sigmoid', 'Sigmoid']

#Classification 1 Auto Encoder
data_arr = class_one_df.values

features_len = len(data_arr[0])
latent_len = features_len - 2
topology = [features_len, latent_len, features_len]

class_one_encoder = net.NeuralNetwork(topology, activation_scheme)

class_one_encoder.train(data_arr, data_arr, epochs=300)
class_one_encoder.run(data_arr[1], data_arr[1])

#Classification 2 Auto Encoder
data_arr = class_two_df.values

features_len = len(data_arr[0])
latent_len = features_len - 2
topology = [features_len, latent_len, features_len]

class_two_encoder = net.NeuralNetwork(topology, activation_scheme)

class_two_encoder.train(data_arr, data_arr, epochs=300)
class_two_encoder.run(data_arr[1], data_arr[1])

#Classification 3 Auto Encoder
data_arr = class_three_df.values

features_len = len(data_arr[0])
latent_len = features_len - 2
topology = [features_len, latent_len, features_len]

class_three_encoder = net.NeuralNetwork(topology, activation_scheme)

class_three_encoder.train(data_arr, data_arr, epochs=300)
class_three_encoder.run(data_arr[1], data_arr[1])

# print('Latent Layers: %s' % str(neural_net.latent_layers))

# Experiment B Classification Neural Net
latent_layers = class_one_encoder.latent_layers + class_two_encoder.latent_layers + class_three_encoder.latent_layers
