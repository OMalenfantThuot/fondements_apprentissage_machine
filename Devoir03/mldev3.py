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

	def bprop(self, values, Y):
		"""
		Method performing the backpropagation for the model.
		
		Parameters:
		-----------
		values: Stored intermediate values of the forward propagation pass.
		Y: Target values (of the training set). Shape: (num_class, num_exemple)

		Returns:
		--------
		grads: Dictionary containing the gradients of the parameters.

		"""

		grads = {}
		dO_a = values["O_s"] - Y
		dW2 = np.mean(dO_a[:,None,:] * values["h_s"][None,:,:], axis=2) +\
			2 * self.lams[3] * self.parameters["W2"] +\
			self.lams[2] * np.sign(self.parameters["W2"])
		db2 = np.mean(dO_a, axis=1, keepdims=True)
		dh_s = np.sum(dO_a[:,None,:] * self.parameters["W2"][:,:,None], axis=0)
		dh_a = dh_s * 1. * (values["h_s"]>0)
		dW1 = np.mean(dh_a[:,None,:] * values["X"][None,:,], axis=2) +\
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
		values: Dictionary of the intermediate values at each step of the propagation.
		"""

		h_a = np.dot(self.parameters["W1"], X) + self.parameters["b1"]
		h_s = self.relu(h_a)
		O_a = np.dot(self.parameters["W2"], h_s) + self.parameters["b2"]
		O_s = self.softmax(O_a)
		values = {"h_a":h_a, "h_s":h_s, "O_a":O_a, "O_s":O_s, "X":X}
		return values

	def grad_check(self, X, Y, totest):
		values = self.fprop(X)
		grad = self.bprop(values, Y)		
		epsilon = 0.000001
		dtest = np.zeros(grad["d"+totest].shape)

		for i in range(dtest.shape[0]):
			for j in range(dtest.shape[1]):
				self.parameters[totest][i,j] = self.parameters[totest][i,j] + epsilon
				values = self.fprop(X)
				loss1 = self.loss(X, Y, values)
				self.parameters[totest][i,j] = self.parameters[totest][i,j] - 2 * epsilon
				values = self.fprop(X)
				loss2 = self.loss(X, Y, values)
				dtest[i,j] = (loss1-loss2)/(2*epsilon)/X.shape[1]
				self.parameters[totest][i,j] = self.parameters[totest][i,j] + epsilon
		return dtest/grad["d"+totest]

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

	def loss(self, X, Y, values): 
		"""
		Calculate the loss value.
		"""

		loss = np.sum(-np.log(values["O_s"])*Y)
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

	def to_minibatch(self, X, Y, seed):
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

	def train(self, dataset, num_epoch, lr=0.01):
		acc_train = []
		acc_valid = []
		acc_test = []
		loss_train = []
		loss_valid = []
		loss_test = []
		for epoch in range(num_epoch):
			minibatch = self.to_minibatch(dataset.train_x, dataset.train_y, epoch)
			loss = 0			
			for mini in minibatch:
				mini_x = mini[0]
				mini_y = mini[1]
				values = self.fprop(mini_x)
				grad = self.bprop(values, mini_y)
				self.update_param(grad,(lr / (1 + 4 * epoch / num_epoch)))
				loss += self.loss(mini_x, mini_y, values)
			acc_train.append(1-self.accuracy(dataset.train_x, dataset.train_y))
			acc_test.append(1-self.accuracy(dataset.test_x, dataset.test_y))
			print(loss)
		return acc_train, acc_test

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
 
if __name__ == "__main__":
	np.random.seed(10)
	#plt.style.use('ggplot')	
	plt.rc('xtick', labelsize=15)
	plt.rc('ytick', labelsize=15)
	plt.rc('axes', labelsize=15)
	
	# Load circle datas for cpu
	data = np.loadtxt("cercle.txt")
	# Load circle data for gpu
	#data = npp.loadtxt("cercle.txt")
	#data = np.array(data)

	data = dataset(data[:,:-1], 1. * (data[:,-1]>0), 2)
	
	# Question 1/2
	NN = NeuralNetwork(layers=[2, 10, 2], minibatch_size=16,
		lams=[0.001,0.001,0.001,0.001])
	#print(NN.grad_check(data.train_x, data.train_y, "W2"))
	a_train, a_test = NN.train(data, num_epoch=30, lr=0.5)
	pred=NN.prediction(data.train_x)
	
	col=["sr","sk"]
	plt.figure()
	for i in range(len(pred)):
		plt.plot(float(data.train_x[0,i]), float(data.train_x[1,i]), col[int(pred[i])])
	plt.axis("equal")

	plt.figure()
	plt.plot(range(30),a_train,"r")
	plt.plot(range(30),a_test,"k")
	
	X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
	X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
	X = np.concatenate((np.array(X_train), np.array(X_test)), axis=0)
	Y = np.concatenate((np.array(y_train), np.array(y_test)), axis=0)
	mnist = dataset(X/255., Y, 10)
	
	# for cuda
	#NN = NeuralNetwork(layers=[784,64,10], minibatch_size=512)
	#a_train, a_test = NN.train(mnist, num_epoch=15, lr=0.25)
	
	#for cpu
	NN = NeuralNetwork(layers=[784,64,10], minibatch_size=32)
	a_train, a_test = NN.train(mnist, num_epoch=5, lr=0.01)
	
	#plt.figure()
	#plt.plot(range(5),a_train,"r")
	#plt.plot(range(5),a_test,"k")
	plt.show()