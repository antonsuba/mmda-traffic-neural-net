import pandas as pd
import numpy as np
import math
from random import uniform
from neural_network import neural_network as net

TRAIN_DATA_X = 'data/x_train.csv'
TRAIN_DATA_Y = 'data/y_train.csv'
TEST_DATA_X = 'data/x_test.csv'
TEST_DATA_Y = 'data/y_test.csv'

train_x_df = pd.read_csv(TRAIN_DATA_X)
train_y_df = pd.read_csv(TRAIN_DATA_Y)

df_y = train_y_df['Classification']
df_x = train_x_df[['NumDeath', 'NumInjured', 'NumPedestrianVictim', 'NumVehInteraction', 'PassengerInjured',
                   'PassengerKilled', 'PedestrianInjured','PedestrianKilled','DriversInjured','DriversKilled']]

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
for data_arr in df_values_arr:
    features_len = len(data_arr[0])
    latent_len = features_len - 6
    topology = [features_len, latent_len, features_len]

    activation_scheme = ['Sigmoid', 'Sigmoid']

    neural_net = net.NeuralNetwork(topology, activation_scheme)

    neural_net.train(data_arr, data_arr, epochs=600)
    neural_net.run(data_arr[2],data_arr[2])
