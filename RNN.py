"""
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
https://www.freecodecamp.org/news/the-ultimate-guide-to-recurrent-neural-networks-in-python/
"""


import pandas
import matplotlib.pyplot as plt
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# fix random seed for reproducibility
numpy.random.seed(7)


# load and plot dataset
dataset = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python')
dataset = dataset.values
dataset = dataset.astype('float32')
# plt.plot(dataset)
# plt.show()


# LSTM sensitive to scale of input data (esp. for tanh and sigmoid activation functions)
# Rescle data to range (0, 1) i.e. normalize
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)



# split into train and test sets
train_size = int(len(dataset) * 0.67)        # index of split with 67% of data in training dataset
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# convert an array of values into a dataset matrix
def create_dataset(dataset, N=1):
	"""
	Takes two arguments: 
	- dataset = NumPy time-series array that we want to convert into a dataset
	- N number of previous time steps to use as input variables to predict the next time step 

	Default N=1 creates a dataset where X is the value at time (t) and Y the value at time (t + 1).
	"""
	dataX, dataY = [], []
	for i in range(len(dataset) -  N):            # for all but the last N values in the data set
		a = dataset[i:(i + N), 0]                 # values used to predict next value
		dataX.append(a)
		dataY.append(dataset[i + N, 0])           # predicted next value
	return numpy.array(dataX), numpy.array(dataY)



# reshape into X=t and Y=t+1
N = 3
trainX, trainY = create_dataset(train, N)
testX, testY = create_dataset(test, N)
print(len(trainX), len(testX))

# LSTM expects inputs as 3D matrix with dimensions [samples, timesteps, features]

# Model past observations as separate input features 
# trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX  = numpy.reshape(testX,  (testX.shape[0], 1, testX.shape[1]))

# Model past observations as time steps of the one input feature 
# (allows number of features to vary with each sample)
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

batch_size = 1 # number of training samples to work through before the model’s internal parameters are updated.
n_epochs = 100 # number of complete passes through the training dataset.

# create and fit the LSTM network
model = Sequential()

# model.add(LSTM(4,                        # 4 LSTM blocks or neurons, one layer
# 	      #input_shape=(1, N))           # N timesteps, N predictors/features  
#           input_shape=(N, 1))           # N timesteps, N predictors/features 
#           ) 

# statfeul LSTM
# stacked LSTMs (return sequence must be set to true for all but last layer)
model.add(LSTM(4, batch_input_shape=(batch_size, N, 1), stateful=True, return_sequences=True))  
model.add(LSTM(4, batch_input_shape=(batch_size, N, 1), stateful=True, return_sequences=True)) 
model.add(LSTM(4, batch_input_shape=(batch_size, N, 1), stateful=True))                 

model.add(Dense(1))                      # output a single prediction 

model.compile(loss='mean_squared_error', # compile the network 
	          optimizer='adam')


# Fit model to data 
# Normally, the state within the network is reset after each training batch when fitting the model (and each call to the predict function)
# model.fit(trainX, trainY, 
#           epochs=n_epochs, batch_size=1,      # train for 100 epochs, batch size 1
#           verbose=2) 


# LSTM with Memory Between Batches
# Fine control over when the internal state of the LSTM network is cleared 
# Can build state over the entire training sequence and even maintain that state if needed to make predictions.
# batch_size = 1
# Loop manually gives number of epochs 
for i in range(n_epochs):                                                                     
	model.fit(trainX, trainY, 
	          epochs=1, batch_size=batch_size,     # fit takes one epoch each iteration 
		      verbose=2, shuffle=False) 
	model.reset_states()                           # reset states after each exposure to training data 


# make predictions for Y, based on X
# trainPredict = model.predict(trainX) 
# testPredict = model.predict(testX)
trainPredict = model.predict(trainX, batch_size=batch_size) # stateful LSTM, memory between batches 
testPredict = model.predict(testX, batch_size=batch_size)


# invert predictions to convert back to same units as original, unscaled input
trainPredict = scaler.inverse_transform(trainPredict) # unscaled predictions, training
trainY = scaler.inverse_transform([trainY])           # unscaled labels, training
testPredict = scaler.inverse_transform(testPredict)   # unscaled predictions, test
testY = scaler.inverse_transform([testY])             # unscaled labels, test

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# Aseemble set of predicted values (training and test) and plot
# shift train predictions forward N timesteps for plotting (first N values cannot be predicted)
trainPredictPlot = numpy.empty_like(dataset)               # empty array, size of original data set 
trainPredictPlot[:, :] = numpy.nan                         # all elements are nan
train_start_idx = N
train_stop_idx = len(trainPredict)+N
trainPredictPlot[train_start_idx:train_stop_idx, :] = trainPredict  # populate element N to end of training data predictions 

# shift test predictions forward training_data + N timesteps for plotting (first N values cannot be predicted)
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
test_start_idx = train_stop_idx + N
test_stop_idx = test_start_idx+len(testPredict) 
testPredictPlot[test_start_idx : test_stop_idx] = testPredict

print(len(dataset), len(trainPredict), len(testPredict), N, len(trainPredict)+N, test_start_idx, test_stop_idx)

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset)) # original data
plt.plot(trainPredictPlot)                  # training set prediction
plt.plot(testPredictPlot)                   # test set prediction 
plt.show()






