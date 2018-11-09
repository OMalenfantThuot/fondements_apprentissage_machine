#!/usr/bin/env python

"""
Python code for the hw3 of class IFT6390.
"""

__authors__ = "Jimmy Leroux, Nicolas Laliberte, Olivier Malenfant"
__version__ = "1.0"
__maintainer__ = "Jimmy Leroux"
__email__ = "jim.leroux1@gmail.com"
__studentid__ = "1024610"

# If cuda
#import cupy as np
#import numpy as npp

# if cpu
import numpy as np
import matplotlib.pyplot as plt
import utils.mnist_reader as mnist_reader
import time

class NeuralNetwork:
	"""
	Implement the neural network with one hidden layer
	"""

	def __init__(self, layers=[], lams=[0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
			minibatch_size=16):
		"""
		Initialize the class attributes
		"""

		self.layers = layers # Number of nodes per layers
		self.parameters = {} # Dictionnary of all the model parameters
		self.lams = lams # Regularization hyperparameters lams=[lam11,lam12,lam21,lam22]
		self.K = minibatch_size # Minibatch size
		self.initialize_parameters() # Parameters initialization

	def accuracy(self, X, Y):
		"""
		Method evaluating the accuracy of the model.

		Parameters:
		-----------
		X: Input we wish to evaluate the performance on. Shape: (dim, num_exemple)
		Y: The respective target for each of the exemples.
			Shape: (num_class, num_exemple)
		
		Returns:
		--------
		acc: The accuracy of the model to predict the input X
		
		"""

		pred = self.prediction(X)
		num_correct = float(np.sum(pred==np.argmax(Y, axis=0)))
		acc = num_correct/X.shape[1]
		return acc

	def bprop(self, cache, Y):
		"""
		Method performing the backpropagation for the model.
		
		Parameters:
		-----------
		cache: Stored intermediate cache of the forward propagation pass.
		Y: Target cache (of the training set). Shape: (num_class, num_exemple)

		Returns:
		--------
		grads: Dictionary containing the gradients of the parameters.

		"""

		grads = {}
		dO_a = cache["O_s"] - Y
		dW2 = np.mean(dO_a[:,None,:] * cache["h_s"][None,:,:], axis=2) +\
			2 * self.lams[3] * self.parameters["W2"] +\
			self.lams[2] * np.sign(self.parameters["W2"])
		db2 = np.mean(dO_a, axis=1, keepdims=True)
		dh_s = np.sum(dO_a[:,None,:] * self.parameters["W2"][:,:,None], axis=0)
		dh_a = dh_s * 1. * (cache["h_s"]>0)
		dW1 = np.mean(dh_a[:,None,:] * cache["X"][None,:,], axis=2) +\
			2 * self.lams[1] * self.parameters["W1"] +\
			self.lams[0] * np.sign(self.parameters["W1"])
		db1 = np.mean(dh_a, axis=1, keepdims=True)
		dx = np.sum(dh_a[:,None,:]*self.parameters["W1"][:,:,None], axis=0)
		grads = {"dW2":dW2, "dW1":dW1, "db2":db2, "db1":db1, "dx":dx}
		return grads

	def fprop(self, X):
		"""
		Forward propagation method. It propagated X through the network.

		Parameters:
		-----------
		X: Input matrix we wish to propagate. Shape: (dim, num_exemple)

		Returns:
		cache: Dictionary of the intermediate cache at each step of the propagation.
		"""

		h_a = np.dot(self.parameters["W1"], X) + self.parameters["b1"]
		h_s = self.relu(h_a)
		O_a = np.dot(self.parameters["W2"], h_s) + self.parameters["b2"]
		O_s = self.softmax(O_a)
		cache = {"h_a":h_a, "h_s":h_s, "O_a":O_a, "O_s":O_s, "X":X}
		return cache

	def grad_check(self, X, Y, totest):
		cache = self.fprop(X)
		grad = self.bprop(cache, Y)		
		epsilon = 0.000001
		dtest = np.zeros(grad["d"+totest].shape)

		for i in range(dtest.shape[0]):
			for j in range(dtest.shape[1]):
				self.parameters[totest][i,j] = self.parameters[totest][i,j] + epsilon
				cache = self.fprop(X)
				loss1 = self.loss(X, Y, cache)
				self.parameters[totest][i,j] = self.parameters[totest][i,j] - 2 * epsilon
				cache = self.fprop(X)
				loss2 = self.loss(X, Y, cache)
				dtest[i,j] = (loss1-loss2)/(2*epsilon)/X.shape[1]
				self.parameters[totest][i,j] = self.parameters[totest][i,j] + epsilon
		return dtest, grad["d"+totest]
	
	def stupid_loop_grad_check(self, X, Y, toprint = "Y"):
		n = X.shape[1]
		dtest_W1 = np.zeros(self.parameters["W1"].shape)
		dtest_b1 = np.zeros(self.parameters["b1"].shape)
		dtest_W2 = np.zeros(self.parameters["W2"].shape)
		dtest_b2 = np.zeros(self.parameters["b2"].shape)
		
		grad_W1 = np.zeros(self.parameters["W1"].shape)
		grad_b1 = np.zeros(self.parameters["b1"].shape)
		grad_W2 = np.zeros(self.parameters["W2"].shape)
		grad_b2 = np.zeros(self.parameters["b2"].shape)
		
		for i in range(n):
			it_dtest_W1, it_grad_W1 = self.grad_check(X[:,[i]],Y[:,[i]], "W1")
			it_dtest_b1, it_grad_b1 = self.grad_check(X[:,[i]],Y[:,[i]], "b1")
			it_dtest_W2, it_grad_W2 = self.grad_check(X[:,[i]],Y[:,[i]], "W2")
			it_dtest_b2, it_grad_b2 = self.grad_check(X[:,[i]],Y[:,[i]], "b2")

			dtest_W1 += it_dtest_W1
			dtest_b1 += it_dtest_b1
			dtest_W2 += it_dtest_W2
			dtest_b2 += it_dtest_b2
			
			grad_W1 += it_grad_W1
			grad_b1 += it_grad_b1
			grad_W2 += it_grad_W2
			grad_b2 += it_grad_b2
			
		dtest_W1 /= n
		dtest_b1 /= n
		dtest_W2 /= n
		dtest_b2 /= n
	
		grad_W1 /= n
		grad_b1 /= n
		grad_W2 /= n
		grad_b2 /= n
			
		if toprint == "Y":
			print("Compare the finite difference with the direct computation of the gradient on a minibatch of size " + str(n) + ":\n")
	
			print("Gradient with respect to W1:")
			print("Gradient: " + str(grad_W1.flatten()))
			print("Finite difference: " + str(dtest_W1.flatten()))
			print("\n")
	
			print("Gradient with respect to b1:")
			print("Gradient: " + str(grad_b1.flatten()))
			print("Finite difference: " + str(dtest_b1.flatten()))
			print("\n")	
	
			print("Gradient with respect to W2:")
			print("Gradient: " + str(grad_W2.flatten()))
			print("Finite difference: " + str(dtest_W2.flatten()))
			print("\n")
	
			print("Gradient with respect to b2:")
			print("Gradient: " + str(grad_b2.flatten()))
			print("Finite difference: " + str(dtest_b2.flatten()))	   
			print("\n")
		
		return grad_W1, grad_b1, grad_W2, grad_b2
		
	def initialize_parameters(self):
		"""
		Initialization of the model's parameters
		"""

		num_layer = len(self.layers)
		for i in range(1, num_layer):
			n_c = 1./np.sqrt(self.layers[i])
			self.parameters["W" + str(i)] = np.ones((self.layers[i],
				self.layers[i-1])) * np.random.uniform(-n_c, n_c,
					(self.layers[i],self.layers[i-1]))
			self.parameters["b" + str(i)] = np.zeros((self.layers[i],1))

	def loss(self, X, Y, cache): 
		"""
		Compute the loss value.
		"""

		loss = np.sum(-np.log(cache["O_s"]+0.000001)*Y)
		loss += (self.lams[0] * np.sum(np.abs(self.parameters["W1"])) +\
			self.lams[2] * np.sum(np.abs(self.parameters["W2"])) +\
			self.lams[1] * np.sum(self.parameters["W1"]**2) +\
			self.lams[3] * np.sum(self.parameters["W2"]**2)) * X.shape[1]
		return loss

	def prediction(self, X):
		pred = self.fprop(X)["O_s"]
		return np.argmax(pred, axis=0)

	def relu(self, X):
		return np.maximum(0, X)

	def softmax(self, X):
		max_ = X.max(axis=0)
		O_s = np.exp(X - max_) / np.sum(np.exp(X - max_), axis=0)
		return O_s

	def to_minibatch(self, X, Y, seed = 10):
		np.random.seed(seed)
		inds = np.arange(X.shape[1])
		np.random.shuffle(inds)		
		random_x = X[:,inds] 
		random_y = Y[:,inds]
		complete_mini = X.shape[1] // self.K

		minibatch = []
		for i in range(complete_mini):
			mini_x = random_x[:,i * self.K:(i + 1) * self.K]
			mini_y = random_y[:,i * self.K:(i + 1) * self.K]
			minibatch.append((mini_x, mini_y))

		if X.shape[1]%self.K!=0:
			mini_x = random_x[:,complete_mini * self.K:]
			mini_y = random_y[:,complete_mini * self.K:]
			minibatch.append((mini_x, mini_y))
		return minibatch

	def train(self, dataset, num_epoch, lr=0.01, comp_err = "N"):
		acc_train = []
		acc_valid = []
		acc_test = []
		err_train = []
		err_valid = []
		err_test = []
		loss_train = 0.
		loss_valid = 0.
		loss_test = 0.
		for epoch in range(num_epoch):
			minibatch = self.to_minibatch(dataset.train_x, dataset.train_y, seed = epoch)			
			for mini in minibatch:
				mini_x = mini[0]
				mini_y = mini[1]
				cache = self.fprop(mini_x)
				grad = self.bprop(cache, mini_y)
				self.update_param(grad,(lr / (1 + 4 * epoch / num_epoch)))
			if comp_err == "Y":
				cache_train = self.fprop(dataset.train_x)
				cache_valid = self.fprop(dataset.valid_x)
				cache_test = self.fprop(dataset.test_x)
				loss_train = self.loss(dataset.train_x, dataset.train_y, cache_train)
				loss_valid = self.loss(dataset.valid_x, dataset.valid_y, cache_valid)
				loss_test 	= self.loss(dataset.test_x, dataset.test_y, cache_test)
				acc_train.append((1-self.accuracy(dataset.train_x, dataset.train_y))*100)
				acc_valid.append((1-self.accuracy(dataset.valid_x, dataset.valid_y))*100)
				acc_test.append((1-self.accuracy(dataset.test_x, dataset.test_y))*100)
				err_train.append(loss_train/float(dataset.train_x.shape[1]))
				err_valid.append(loss_valid/float(dataset.valid_x.shape[1]))
				err_test.append(loss_test/float(dataset.test_x.shape[1]))
		return acc_train, acc_valid, acc_test, err_train, err_valid, err_test
		
	def stupid_hyper_loop(self, dataset, num_epoch, lr=0.05):
		acc_train = []
		acc_test = []
		for epoch in range(num_epoch):
			minibatch = self.to_minibatch(dataset.train_x, dataset.train_y, seed = epoch)
			loss = 0			
			for mini in minibatch:
				grad = {}
				grad["dW1"] = np.zeros(self.parameters["W1"].shape)
				grad["db1"] = np.zeros(self.parameters["b1"].shape)
				grad["dW2"] = np.zeros(self.parameters["W2"].shape)
				grad["db2"] = np.zeros(self.parameters["b2"].shape)
				mini_x = mini[0]
				mini_y = mini[1]
				for i in range(mini_x.shape[1]):
					cache = self.fprop(mini_x[:,[i]])
					grad["dW1"] += self.bprop(cache, mini_y[:,[i]])["dW1"]
					grad["db1"] += self.bprop(cache, mini_y[:,[i]])["db1"]
					grad["dW2"] += self.bprop(cache, mini_y[:,[i]])["dW2"]
					grad["db2"] += self.bprop(cache, mini_y[:,[i]])["db2"]					
					loss += self.loss(mini_x[:,[i]], mini_y[:,[i]], cache)
				grad["dW1"] /= mini_x.shape[1]
				grad["db1"] /= mini_x.shape[1]
				grad["dW2"] /= mini_x.shape[1]
				grad["db2"] /= mini_x.shape[1]
				self.update_param(grad,(lr / (1 + 4 * epoch / num_epoch)))
			acc_train.append(1-self.accuracy(dataset.train_x, dataset.train_y))
			acc_test.append(1-self.accuracy(dataset.test_x, dataset.test_y))
		return grad
		
	def update_param(self, grad, lambda_):
		self.parameters["W1"] = self.parameters["W1"] - lambda_ * grad["dW1"]
		self.parameters["W2"] = self.parameters["W2"] - lambda_ * grad["dW2"]
		self.parameters["b1"] = self.parameters["b1"] - lambda_ * grad["db1"]
		self.parameters["b2"] = self.parameters["b2"] - lambda_ * grad["db2"]
		
class dataset:
	def __init__(self,X,Y,numclass):
		self.X = X.T		
		self.Y = Y
		self.train_x = 0
		self.train_y = 0
		self.valid_x = 0
		self.valid_y = 0
		self.test_x = 0
		self.test_y = 0		
		self.numclass = numclass		
		self.toonehot()
		self.split_and_randomize()

	def toonehot(self):	
		onehot = np.zeros((self.numclass,len(self.Y)))
		for j in range(len(self.Y)):
			onehot[int(self.Y[j]),j] = 1.
		self.Y = onehot

	def split_and_randomize(self):
		n_train = int(0.70 * self.X.shape[1])
		n_valid = int(0.15 * self.X.shape[1])
		inds = np.arange(self.X.shape[1])
		np.random.shuffle(inds)
		train_inds = inds[:n_train]
		valid_inds = inds[n_train:n_train+n_valid]
		test_inds = inds[n_train+n_valid:]		
		self.train_x = self.X[:,train_inds]
		self.train_y = self.Y[:,train_inds]
		mean_train = self.train_x.mean(axis=1, keepdims=True)
		std_train = self.train_x.std(axis=1, keepdims=True)
		self.train_x = (self.train_x - mean_train) / std_train
		self.valid_x = (self.X[:,valid_inds] - mean_train) / std_train
		self.valid_y = self.Y[:,valid_inds]
		self.test_x = (self.X[:,test_inds] - mean_train) / std_train
		self.test_y = self.Y[:,test_inds]
		
	def grid(self, precision = 0.01):
		min_x1 = np.min(self.train_x, axis = 1)[0]-precision
		min_x2 = np.min(self.train_x, axis = 1)[1]-precision
		max_x1 = np.max(self.train_x, axis = 1)[0]+precision
		max_x2 = np.max(self.train_x, axis = 1)[1]+precision

		x1_mesh = np.arange(min_x1, max_x1, precision)
		x2_mesh = np.arange(min_x2, max_x2, precision)
		mesh_grid = np.meshgrid(x1_mesh, x2_mesh)
		grid_x1 = mesh_grid[0].ravel()
		grid_x2 = mesh_grid[1].ravel()
	
		grid = np.array([grid_x1, grid_x2])
		
		return grid
 
if __name__ == "__main__":
	np.random.seed(10)
	#Experiments ---------------------------------------------------------------
	
	data = np.loadtxt("cercle.txt")
	data = dataset(data[:,:-1], 1. * (data[:,-1]>0), 2)
    
	exp_NN = NeuralNetwork(layers=[2,2,2],
		lams=[0.001,0.001,0.001,0.001])

	"""
	We compute the gradient and finite difference for a single exemple.
	"""
	print("Question 2: \n")
	
	exp_NN.stupid_loop_grad_check(data.train_x[:,[1]], data.train_y[:,[1]])
	
	"""
	We compute the gradient and finite difference for a minibatch of size K = 10.
	"""
	print("Question 4: \n")
	
	exp_NN.K = 10
	
	minibatch = exp_NN.to_minibatch(data.train_x, data.train_y)
	mini_x = minibatch[0][0]
	mini_y = minibatch[0][1]
	exp_NN.stupid_loop_grad_check(mini_x, mini_y)

	"""
	We plot decision region
	"""
	print("Question 5: \n")
	
	min_x1 = np.min(data.train_x, axis = 1)[0]-0.1
	min_x2 = np.min(data.train_x, axis = 1)[1]-0.1
	max_x1 = np.max(data.train_x, axis = 1)[0]+0.1
	max_x2 = np.max(data.train_x, axis = 1)[1]+0.1
	precision_mesh = 0.01

	x1_mesh = np.arange(min_x1, max_x1, precision_mesh)
	x2_mesh = np.arange(min_x2, max_x2, precision_mesh)
	mesh_grid = np.meshgrid(x1_mesh, x2_mesh)
	grid_x1 = mesh_grid[0].ravel()
	grid_x2 = mesh_grid[1].ravel()
	
	grid = np.array([grid_x1, grid_x2])

	exp_NN = NeuralNetwork(layers=[2,128,2], lams=[0.0001, 0.0001, 0.0001, 0.0001], minibatch_size = 16)
		
	exp_NN.train(data, num_epoch = 30, lr = 0.5)
			
	y = np.array(exp_NN.prediction(grid))
	prediction_grid = np.array([grid_x1, grid_x2, y])
	data_0 = np.array([prediction_grid[:-1,i] for i in range(prediction_grid.shape[1]) if prediction_grid[2,i] == 0.])
	data_1 = np.array([prediction_grid[:-1,i] for i in range(prediction_grid.shape[1]) if prediction_grid[2,i] == 1.])

	plt.figure()
	plt.title("Decision boundary: num_epoch = 30, K = 16, weight = 0.0001, 128 hidden neurons")
	plt.xlabel("x1")
	plt.ylabel("x2")
	plt.xlim((min_x1, max_x1))
	plt.ylim((min_x1, max_x2))
	if len(data_0) > 0:
		plt.plot(data_0[:,0], data_0[:,1],"o", markersize = 4, color = 'b', alpha = 0.008)
	if len(data_1) > 0:
		plt.plot(data_1[:,0], data_1[:,1],"o", markersize = 4, color = 'orange', alpha = 0.008)
	plt.scatter(data.train_x[0], data.train_x[1], data.train_y[0] + 10)
	plt.axis("equal")
	
	exp_NN = NeuralNetwork(layers=[2,8,2], lams=[0.001, 0.001, 0.001, 0.001], minibatch_size = 10)
		
	exp_NN.train(data, num_epoch = 10, lr = 0.5)
	
	y = np.array(exp_NN.prediction(grid))
	prediction_grid = np.array([grid_x1, grid_x2, y])
	data_0 = np.array([prediction_grid[:-1,i] for i in range(prediction_grid.shape[1]) if prediction_grid[2,i] == 0.])
	data_1 = np.array([prediction_grid[:-1,i] for i in range(prediction_grid.shape[1]) if prediction_grid[2,i] == 1.])
	
	plt.figure()
	plt.title("Decision boundary: num_epoch = 10, K = 10, weight = 0.001, 8 hidden neurons")
	plt.xlabel("x1")
	plt.ylabel("x2")
	plt.xlim((min_x1, max_x1))
	plt.ylim((min_x1, max_x2))
	if len(data_0) > 0:
		plt.plot(data_0[:,0], data_0[:,1],"o", markersize = 4, color = 'b', alpha = 0.008)
	if len(data_1) > 0:
		plt.plot(data_1[:,0], data_1[:,1],"o", markersize = 4, color = 'orange', alpha = 0.008)
	plt.scatter(data.train_x[0], data.train_x[1], data.train_y[0] + 10)
	plt.axis("equal")
	

	exp_NN = NeuralNetwork(layers=[2,2,2], lams=[0.001, 0.001, 0.001, 0.001], minibatch_size = 5)
		
	exp_NN.train(data, num_epoch = 5, lr = 0.5)
	
	y = np.array(exp_NN.prediction(grid))
	prediction_grid = np.array([grid_x1, grid_x2, y])
	data_0 = np.array([prediction_grid[:-1,i] for i in range(prediction_grid.shape[1]) if prediction_grid[2,i] == 0.])
	data_1 = np.array([prediction_grid[:-1,i] for i in range(prediction_grid.shape[1]) if prediction_grid[2,i] == 1.])
	
	plt.figure()
	plt.title("Decision boundary: num_epoch = 5, K = 5, weight = 0.001, 2 hidden neurons")
	plt.xlabel("x1")
	plt.ylabel("x2")
	plt.xlim((min_x1, max_x1))
	plt.ylim((min_x1, max_x2))
	if len(data_0) > 0:
		plt.plot(data_0[:,0], data_0[:,1],"o", markersize = 4, color = 'b', alpha = 0.008)
	if len(data_1) > 0:
		plt.plot(data_1[:,0], data_1[:,1],"o", markersize = 4, color = 'orange', alpha = 0.008)
	plt.scatter(data.train_x[0], data.train_x[1], data.train_y[0] + 10)
	plt.axis("equal")
	plt.show()
	
	print("\n")

	"""
	m1 = len(x1_mesh)
	m2 = len(x2_mesh)

	plt.figure()
	plt.imshow(exp_NN.prediction(grid).reshape(m2,m1), alpha = 0.5, extent = (min_x1, max_x1, min_x2, max_x2))
	plt.scatter(data.train_x[0], data.train_x[1], data.train_y[0] + 1)
	plt.axis("equal")
	"""
	
	# 6 to 10
	
	"""
	We compare both implementation with K = 1 and K = 10.
	"""
	
	print("Question 7: \n")
	
	exp_NN = NeuralNetwork(layers=[2,2,2], lams=[0.001,0.001,0.001,0.001], minibatch_size = 1)
	
	minibatch = exp_NN.to_minibatch(data.train_x, data.train_y)
	mini_x = minibatch[0][0]
	mini_y = minibatch[0][1]
	
	grad_W1, grad_b1, grad_W2, grad_b2 = exp_NN.stupid_loop_grad_check(mini_x, mini_y, toprint = "N")
	cache = exp_NN.fprop(mini_x)
	mat_grad = exp_NN.bprop(cache, mini_y)
	
	print("Gradient with loop implementation for minibatch of size K = " + str(exp_NN.K) + ": \n")
	print("Gradient with respect to W1:")
	print(grad_W1.flatten())
	print("\n")
	
	print("Gradient with respect to b1:")
	print(grad_b1.flatten())
	print("\n")
	
	print("Gradient with respect to W2:")
	print(grad_W2.flatten())
	print("\n")
	
	print("Gradient with respect to b2:")
	print(grad_b2.flatten())
	print("\n")
	
	print("Gradient with matrix implementation for minibatch of size K = " + str(exp_NN.K) + ": \n")
	print("Gradient with respect to W1:")
	print(mat_grad["dW1"].flatten())
	print("\n")
	
	print("Gradient with respect to b1:")
	print(mat_grad["db1"].flatten())
	print("\n")
	
	print("Gradient with respect to W2:")
	print(mat_grad["dW2"].flatten())
	print("\n")
	
	print("Gradient with respect to b2:")
	print(mat_grad["db2"].flatten())
	print("\n")
	
	exp_NN = NeuralNetwork(layers=[2,2,2], lams=[0.001,0.001,0.001,0.001], minibatch_size = 10)
	
	minibatch = exp_NN.to_minibatch(data.train_x, data.train_y)
	mini_x = minibatch[0][0]
	mini_y = minibatch[0][1]
	
	grad_W1, grad_b1, grad_W2, grad_b2 = exp_NN.stupid_loop_grad_check(mini_x, mini_y, toprint = "N")
	cache = exp_NN.fprop(mini_x)
	mat_grad = exp_NN.bprop(cache, mini_y)
	
	print("Gradient with loop implementation for minibatch of size K = " + str(exp_NN.K) + ": \n")
	print("Gradient with respect to W1:")
	print(grad_W1.flatten())
	print("\n")
	
	print("Gradient with respect to b1:")
	print(grad_b1.flatten())
	print("\n")
	
	print("Gradient with respect to W2:")
	print(grad_W2.flatten())
	print("\n")
	
	print("Gradient with respect to b2:")
	print(grad_b2.flatten())
	print("\n")
	
	print("Gradient with matrix implementation for minibatch of size K = " + str(exp_NN.K) + ": \n")
	print("Gradient with respect to W1:")
	print(mat_grad["dW1"].flatten())
	print("\n")
	
	print("Gradient with respect to b1:")
	print(mat_grad["db1"].flatten())
	print("\n")
	
	print("Gradient with respect to W2:")
	print(mat_grad["dW2"].flatten())
	print("\n")
	
	print("Gradient with respect to b2:")
	print(mat_grad["db2"].flatten())
	print("\n")
	
	print("Question 8: \n")
	
	X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
	X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
	X = np.concatenate((np.array(X_train), np.array(X_test)), axis=0)
	Y = np.concatenate((np.array(y_train), np.array(y_test)), axis=0)
	mnist = dataset(X/255., Y, 10)
	
	exp_NN = NeuralNetwork(layers=[784,64,10] , lams=[0.001, 0.001, 0.001, 0.001], minibatch_size = 20)
	
	t1 = time.time()

	#exp_NN.stupid_hyper_loop(mnist, num_epoch = 1)
	
	print("It tooks " + str(time.time() - t1) + " seconds to train the NN with the loop implementation \n")

	t2 = time.time()
	
	exp_NN.train(mnist, num_epoch = 1)
	
	print("It tooks " + str(time.time() - t2) + " seconds to train the NN with the matrix implementation \n")

	plt.style.use('ggplot')	
	plt.rc('xtick', labelsize=15)
	plt.rc('ytick', labelsize=15)
	plt.rc('axes', labelsize=15)
	
	print("Question 9&10: \n")
		
	NN = NeuralNetwork(layers=[784, 64, 10], minibatch_size=32,
		lams=[0.0001,0.0001,0.0001,0.0001])

	acc_train, acc_valid, acc_test, err_train, err_valid, err_test = NN.train(
		mnist, num_epoch=10, lr=0.05, comp_err = "Y")
	pred = NN.prediction(mnist.train_x)

	plt.figure()
	plt.title("Error vs Epoch")
	plt.plot(range(1,11), acc_train, label="Train")
	plt.plot(range(1,11), acc_valid, label="Valid")
	plt.plot(range(1,11), acc_test, label="Test")
	plt.xlabel("Epoch")
	plt.ylabel("Error (%)")	
	plt.legend()
	
	plt.figure()
	plt.title("Average loss vs Epoch")
	plt.plot(range(1,11), err_train, label="Train")
	plt.plot(range(1,11), err_valid, label="Valid")
	plt.plot(range(1,11), err_test, label="Test")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.show()
	# for cuda
	#NN = NeuralNetwork(layers=[784,64,10], minibatch_size=512)
	#a_train, a_test = NN.train(mnist, num_epoch=15, lr=0.25)
	
	#for cpu

	
