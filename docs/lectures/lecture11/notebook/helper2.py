import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
np.random.seed(0)
tf.random.set_seed(0)
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


# Use the helper code below to generate the data
def unregularized_model():

	# Defines the number of data points to generate
	num_points = 30 

	# Generate predictor points (x) between 0 and 5
	x = np.linspace(0,5,num_points)

	# Generate the response variable (y) using the predictor points
	y = x * np.sin(x) + np.random.normal(loc=0, scale=1, size=num_points)

	# Generate data of the true function y = x*sin(x) 
	# x_b will be used for all predictions below 
	x_b = np.linspace(0,5,100)
	y_b = x_b*np.sin(x_b)

	# Split the data into train and test sets with .33 and random_state = 42
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

	# Building an unregularized NN. 
	# Initialise the NN, give it an appropriate name for the ease of reading
	# The FCNN has 5 layers, each with 100 nodes
	model_1 = models.Sequential(name='Unregularized')

	# Add 5 hidden layers with 100 neurons each
	model_1.add(layers.Dense(100,  activation='tanh', input_shape=(1,)))
	model_1.add(layers.Dense(100,  activation='relu'))
	model_1.add(layers.Dense(100,  activation='relu'))
	model_1.add(layers.Dense(100,  activation='relu'))
	model_1.add(layers.Dense(100,  activation='relu'))

	# Add the output layer with one neuron 
	model_1.add(layers.Dense(1,  activation='linear'))
	# Load with the weights already provided for the unregularized network

	# Compile the model
	model_1.compile(loss='MSE',optimizer=optimizers.Adam(learning_rate=0.001)) 

	# Save the history about the model
	history_1 = model_1.fit(x_train, y_train,  validation_data=(x_test,y_test), epochs=200, batch_size=10, verbose=0)

	# Use the model above to predict for x_b (used exclusively for plotting) 
	y_pred = model_1.predict(x_b)

	# Use the model above to predict on the test data
	y_pred_test = model_1.predict(x_test)

	# Compute the MSE on the test data
	mse = mean_squared_error(y_test,y_pred_test)

	# Plot the MSE of the model
	plt.rcParams["figure.figsize"] = (10,8)
	plt.title("Unregularized model")
	plt.semilogy(history_1.history['loss'], label='Train Loss', color='#FF9A98')
	plt.semilogy(history_1.history['val_loss'],  label='Validation Loss', color='#75B594')
	plt.legend()

	# Set the axes labels
	plt.xlabel('Epochs')
	plt.ylabel('Log MSE Loss')
	plt.legend()
	plt.show()

	return x_b, x_train, x_test, y_train, y_test, y_pred, mse


