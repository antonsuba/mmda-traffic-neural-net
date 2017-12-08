# MMDA Traffic Accident Severity Neural Net Classifier
MMDA traffic accident data classification using neural networks and autoencoders

## Raw Features Used:
* NumDeath
* NumInjured
* NumPdestrianVictim
* NumVehInteraction
* PassengerInjured
* PassengerKilled
* PedestrianInjured
* PedestrianKilled
* DriversInjured
* DriversKilled

Features were selected based on a preliminary analysis of the dataset.
Categorical features such as junction type and collision type were discarded.
Columns representing specific vehicle types and their quantity were removed in favor of a more generalized
variable in the form of NumVehInteraction.

## How to Run
Open mmda_neural_net.ipynb using jupyter. Running all the cells will produce results and graphs for analysis.

An alternative is to run mmda_neural_net.py (~python mmda_neural_net.py) instead to view results on a terminal.

## Configurations

### NeuralNetwork Main Class
* topology - specify neural network topology
* activation_scheme - activation scheme for feed forward and back propagation. *Limited to __Relu__ or __Sigmoid__*
* momentum - momentum rate for new weights. *Defaults to __1__*
* learning_rate - learning rate for new weights. *Defaults to __1__*
* weight_seed - set numerical seed for uniform weights. *Defaults to __None__*
* custom_weights - list of custom weights. Length of list must be number of layers - 1. *Defaults to __None__*

### Train method
* inputs - input layer
* outputs - output / actual layer
* epochs - set number of epochs. *Defaults to __600__*
