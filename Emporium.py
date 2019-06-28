# Recurrent Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def main():
	# Importing the training set
	dataset_train = pd.read_csv('Apple_Stock_Price.csv')
	training_set = dataset_train.iloc[:, 1:2].values

	# Part 1 - Data Preprocessing

	# Feature Scaling

	sc = MinMaxScaler(feature_range = (0, 1))
	training_set_scaled = sc.fit_transform(training_set)

	# Creating a data structure with 60 timesteps and 1 output
	X_train = []
	y_train = []
	for i in range(60, 1258):
	    X_train.append(training_set_scaled[i-60:i, 0])
	    y_train.append(training_set_scaled[i, 0])
	X_train, y_train = np.array(X_train), np.array(y_train)

	# Reshaping
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

	# Part 2 - Building the RNN

	# Initialising the RNN
	regressor = Sequential()

	# Adding the first LSTM layer and some Dropout regularisation
	regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
	regressor.add(Dropout(0.2))

	# Adding a second LSTM layer and some Dropout regularisation
	regressor.add(LSTM(units = 50, return_sequences = True))
	regressor.add(Dropout(0.2))

	# Adding a third LSTM layer and some Dropout regularisation
	regressor.add(LSTM(units = 50, return_sequences = True))
	regressor.add(Dropout(0.2))

	# Adding a fourth LSTM layer and some Dropout regularisation
	regressor.add(LSTM(units = 50))
	regressor.add(Dropout(0.2))

	# Adding the output layer
	regressor.add(Dense(units = 1))

	# Compiling the RNN
	regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])

	# Fitting the RNN to the Training set
	regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



	# Part 3 - Making the predictions and visualising the results

	while True:
		data_list = dataset_train.values.tolist()  # convert dataframe into a list
		dataset_of_month = []
		while(True):
			year = int(input("Enter the year: "))
			month = int(input("Enter the month: "))
			if month < 10:
				start_date = str(year)+"-0"+str(month)
				break
			elif month <= 12:
				start_date = str(year)+"-"+str(month)
				break

		#data of the month selected by the user
		for row in data_list:
			if row[0][:7] == start_date:
				dataset_of_month.append(row)
		
		#change in the stock price in the month selected by the user
		difference_lsit = []
		for i in range(len(dataset_of_month)-1):
			test = []
			for j in range (1, 7):
				test.append(dataset_of_month[i+1][j] - dataset_of_month[i][j])
			difference_lsit.append(test)	
		
		#training set prepared by adding values of difference list to current day
		dataset_test = []
		dataset_test.append(data_list[-1][1:])
		for i in range(len(difference_lsit)):
			test = []
			for j in range (0, 6):
				test.append(dataset_test[i][j] + difference_lsit[i][j])
			dataset_test.append(test)
		dataset_test = dataset_test[1:]
		dataset_test = pd.DataFrame(np.array(dataset_test), columns = ("Open","High","Low","Close","Adj Close","Volume"))	
		print(dataset_test)




		# Getting the predicted stock price of next month
		#Predsicting market price using previous data
		'''dataset_total = dataset_train['Open']
								inputs = dataset_total[len(dataset_total) - 80:].values
								inputs = inputs.reshape(-1,1)
								inputs = sc.transform(inputs)
								X_test = []
								for i in range(60, 80):
								    X_test.append(inputs[i-60:i, 0])
								X_test = np.array(X_test)
								X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
						
								recent_month_stock_price = regressor.predict(X_test)
								recent_month_stock_price = sc.inverse_transform(recent_month_stock_price)'''

		dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
		inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
		inputs = inputs.reshape(-1,1)
		inputs = sc.transform(inputs)
		X_test = []
		for i in range(60, 80):
		    X_test.append(inputs[i-60:i, 0])
		X_test = np.array(X_test)
		X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


		analysed_stock_price = regressor.predict(X_test)
		analysed_stock_price = sc.inverse_transform(analysed_stock_price)
		print(analysed_stock_price)
		x_data = []
		for i in range(59,60+len(analysed_stock_price)):
			x_data.append(i)
		x_data = np.array(x_data)

		analysed_stock_price = np.concatenate([training_set[-1:],analysed_stock_price])

		# Visualising the results
		plt.plot(training_set[-60:], color = 'red', label = 'Analysed Stock Price')
		plt.plot(x_data, analysed_stock_price, color = 'blue', label = 'User given')
		#plt.plot(x_data, recent_month_stock_price, color = 'green', label = 'Recent driven')
		plt.title('Stock Price Analysis')
		plt.xlabel('Time')
		plt.ylabel('Stock Price')
		plt.legend()
		plt.show()
						

if __name__ == "__main__":
    main()