
#importing essential libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#function to generate the dataset
def generate_dataset(filename,testing_samples):
	dataset = pd.read_excel(filename)

	dataset = (dataset - dataset.mean())/dataset.std()	

	#splitting the dataset into training data and testing data
	training_data = dataset[: -1 * testing_samples]
	testing_data = dataset[ -1 * testing_samples : ]

	training_x = np.mat(training_data.drop(columns = ['PE']))
	training_y = np.mat(training_data['PE']).T

	testing_x = np.mat(testing_data.drop(columns = ['PE']))
	testing_y = np.mat(testing_data['PE']).T

	# print(training_x.shape) #7568 X 4
	# print(training_y.shape) #7568 X 1
	# print(testing_x.shape)	#2000 X 4
	# print(testing_y.shape)  #2000 X 1


	#Adding extra ones for the weights
	training_x = np.concatenate( (	np.full ((training_x.shape[0],1),1) , training_x), axis = 1)
	testing_x = np.concatenate( ( np.full ((testing_x.shape[0], 1),1) , testing_x) , axis = 1)
	
	return training_x, training_y, testing_x, testing_y


#function to generate the least squares error 
def square_error(x,y,w):
	# print("shapes are:")
	# print(x.shape)
	# print(y.shape)
	# print(w.shape)
	m = x.shape[0]
	diff = np.dot(x, w.T) - y
	error = 0.5 * np.dot(diff.T, diff)/ (m)
	return error


#the linear regression class which holds all the important functions for the three parts
#by default learning rate is 0.001
#number of iterations is 10,000 
class LinearRegression():
	def __init__(self, training_x, training_y, testing_x, testing_y, learning_rate = 0.01, n_iters = 10000, tolerance = 0.0005):
		self.learning_rate = learning_rate
		self.n_iters = n_iters
		self.training_x = training_x
		self.training_y = training_y
		self.testing_x = testing_x
		self.testing_y = testing_y
		self.predictions = None #to store the predicted values when the weights are multiplied by the testing_x
		self.initial_cost = 0
		self.w = np.random.rand(self.training_x.shape[1] ,1).T
		self.num_features = training_x.shape[1]
		

	def normal_equations_method(self):
		xTx = self.training_x.T  * self.training_x

		#if not a square matrix inverse cannot exist
		if( np.linalg.det(xTx) == 0.0 ) : raise(Exception("Inverse cannot exist"))

		else:
			weights = xTx.I * (self.training_x.T * self.training_y) #calulating the weights using inv(xTx).x.y
			error = 0.0
			predictions = np.dot(self.testing_x, weights) #predicting the values
			error = square_error(self.testing_x, self.testing_y,weights.T)
			print("Testing error is:", error)


	#function for calculating the weights for gradient descent without regularization
	def fit(self):
		#m is the number of training samples
		m = self.training_x.shape[0]
		num_features = self.training_x.shape[1]

		for i in range(self.n_iters):

			print("Iteration number is:",i)
			print("Current cost is:", square_error(self.training_x, self.training_y,self.w))
			print("W is:")
			print(self.w)

			#calcuating the difference between predicted values and actual values
			diff = np.dot(self.training_x, self.w.T) - self.training_y  #7568 X 1

			#the gradient(partial derivative) which is (x.diff)/m
			gradient = np.dot(self.training_x.T, diff) / (m)

			#calculating the updated weights
			self.w = self.w - (self.learning_rate * gradient.T)



	#predicting the values for the test data and showing the testing error
	def predict(self):
		error = 0.0
		self.predictions = np.dot(self.testing_x, self.w.T) 
		diff = np.dot(self.testing_x, self.w.T) - self.testing_y
		error = 0.5 * np.dot(diff.T, diff)/ (self.training_x.shape[0])
		print("Testing error is:", error)



	#function for calculating the gradient descent with l1-regularization
	def l1_regularization(self, coefficient):
		w = np.random.rand(5,1).T
		m = self.training_x.shape[0]

		for i in range(self.n_iters):
			print("Iteration number is L1:",i)
			diff = np.dot(self.training_x, w.T) - self.training_y
			gradient = (np.dot(self.training_x.T, diff) + (coefficient * np.divide(w.T, np.abs(w.T))) ) / (m)
			w = w - self.learning_rate * gradient.T
			
			#print("New cost is:", square_error(self.training_x, self.training_y,w))


		cv_err = square_error(self.validation_x, self.validation_y, w)
		test_err =  square_error(self.testing_x, self.testing_y,w)
		return cv_err, test_err


	#function for calculating the gradient descent with l1-regularization
	def l2_regularization(self,coefficient):
		w = np.random.rand(5,1).T
		# print("initial_cost",w.shape)
		m = self.training_x.shape[0]

		for i in range(self.n_iters):
			print("Iteration number is:",i)
			diff = np.dot(self.training_x, w.T) - self.training_y
			gradient = (np.dot(self.training_x.T, diff) + (2 * coefficient * w.T) ) / (m)
			w = w - self.learning_rate * gradient.T
			#print("New cost is:", square_error(self.training_x, self.training_y,w))

		cv_err = square_error(self.validation_x, self.validation_y, w)
		test_err =  square_error(self.testing_x, self.testing_y,w)
		return cv_err, test_err

	#function for applying linear regression with regularization
	#returns both l1 and l2 errors
	def fit_with_regularization(self):
		#splitting the training data to train and cross validation data
		#we are going to make a 60-20-20 split in total (training, validation, testing)
		self.validation_x = self.training_x[int(-1 * self.training_x.shape[0] * 0.2):]
		self.training_x = self.training_x[: int(-1 * self.training_x.shape[0] * 0.2)]

		self.validation_y = self.training_y[int(-1 * self.training_y.shape[0] * 0.2):]
		self.training_y = self.training_y[: int(-1 * self.training_y.shape[0] * 0.2)]

		#the coefficient value - i.e.lambda values
		coefficients = np.mat([0.00001, 0.0001, 0.001, 0.01 ,0.1 ,1 , 10, 100, 1000]).T
		#print(coefficient.shape)
		l1_err_cv = np.zeros(coefficients.shape[0])
		l2_err_cv = np.zeros(coefficients.shape[0])
		l1_err_test = np.zeros(coefficients.shape[0])
		l2_err_test = np.zeros(coefficients.shape[0])

		for i in range(coefficients.shape[0]):
			print("Calculating for coeffi")
			l1_err_cv[i], l1_err_test[i] = self.l1_regularization(float(coefficients[i][0]))
			l2_err_cv[i], l2_err_test[i] = self.l2_regularization(float(coefficients[i][0]))
			

		print("L1 error is:",l1_err_cv)
		print("L2 error is:",l2_err_cv)

		print("L1 reg Error for test:", l1_err_test)
		print("L2 reg Error for test:", l2_err_test)

		#function for plotting the errors obtained with l1 and l2 regularization
		coefficients = np.array([-5, -4, -3, -2 ,-1 ,0 , 1, 2, 3])

		plt.plot(coefficients, l1_err_cv, 'r--', label = 'l1 regression errors')
		plt.plot(coefficients, l2_err_cv, 'b--', label = 'l2 regression errors')
		plt.legend()
		plt.xlabel('coefficient value')
		plt.ylabel('cross-validation error')
		plt.title('Cross validation errors with different lambda values')

		#save the graph
		# plt.savefig('Errors_lambda_new.png', format = "png")

		#FOR HUNDRED ITERATIONS
		# [1.68652868 1.53296998 1.45960332 2.07007336 0.93953382 1.13365495
		#  0.97757305 1.00738326 0.87149   ]
		# [1.30285894 2.18976079 1.4884043  1.44491661 0.85906633 1.88885805
		#  1.28409482 1.04905928 0.90099086]

		#FOR TEN THOUSAND ITERATIONS
		# [0.03926908 0.03932065 0.04538471 0.04319545 0.03877882 0.03808641
		#  0.03793691 0.04340717 0.05721345]
		# [0.04078567 0.0422648  0.0478478  0.03737022 0.04210545 0.0404004
		#  0.04592231 0.04573241 0.05560283]





def main():
	training_x, training_y, testing_x, testing_y =generate_dataset("../dataset.xlsx",2000)
	
	# calculating for Part A : Normal equations method
	# nom = LinearRegression(training_x, training_y, testing_x, testing_y)
	# nom.normal_equations_method()


	# # calculating for  Part B :gradient descent with regularization
	# gd = LinearRegression(training_x, training_y, testing_x, testing_y)
	# gd.fit()
	# gd.predict()

	# Calculating for Part C: gradient descent with Regularization 
	gd_reg = LinearRegression(training_x, training_y, testing_x, testing_y)
	gd_reg.fit_with_regularization()



if __name__ == '__main__':
	main()
