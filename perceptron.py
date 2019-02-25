import matplotlib,sys
from matplotlib import pyplot as plot 
import numpy as np

class Perceptron(object):

	def __init__(self, numInputs, threshold = 100, learningRate = 0.01):
		self.threshold = threshold
		self.learningRate = learningRate
		self.weights = np.zeros(numInputs + 1)


	def prediction(self, input):
		sum = np.dot(input, self.weights[1:]) + self.weights[0]
		# return activation based on sum
		if sum > 0:
			return 1
		else:
			return -1
	
	# calculate prediction accuracy	
	def checkAccuracy(matrix, weights):
		correct = 0.0
		predictions = []
		for i in range(len(matrix)):
			predict = predict(matrix[i][:1], weights)
			predictions.append(predict)
			# checks if prediction is accurate 
			if predict == matrix[i][-1]: correct += 1.0
			return correct/float(len(matrix))

	def main():








