import numpy as np 
from sklearn import svm
from sklearn import ensemble
from sklearn import neural_network
from sklearn import linear_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt

class PostClassification:


	def __init__(self, modelType):

		self.modelType = modelType

		if(modelType == "LogReg"):
			self.model = linear_model.LogisticRegression(class_weight = 'balanced')
		if(modelType == "LinearSVM"):
			self.model = svm.SVC(class_weight = 'balanced',kernel='linear')
		if(modelType == "RbfSVM"):
			self.model = svm.SVC(class_weight = 'balanced',kernel='rbf')
		if(modelType == "AdaBoost"):
			self.model = ensemble.AdaBoostClassifier(n_estimators=1000)
		if(modelType == "RandomForest"):
			self.model = ensemble.RandomForestClassifier(n_estimators=200, max_depth = 100, min_samples_leaf = 1)
		if(modelType == "MLP"):
			self.model = neural_network.MLPClassifier(hidden_layer_sizes=(64,64,64), max_iter = 500, activation = 'relu', verbose=True)		



	# parameters
	#
	# X: numpy array of training data: (num_observations, num_features)
	# y: numpy array of ground truth values: (num_observations)
	def train(self,X,y, print = False):

		if(self.modelType == "LogReg"):
			self.model.fit(X,y)
		if(self.modelType == "LinearSVM"):
			self.model.fit(X,y)
		if(self.modelType == "RbfSVM"):
			self.model.fit(X,y)
		if(self.modelType == "AdaBoost"):
			self.model.fit(X,y)
		if(self.modelType == "RandomForest"):
			self.model.fit(X,y)
		if(self.modelType == "MLP"):
			self.model.fit(X,y)

	def test(self,X):

		if(self.modelType == "LogReg"):
			y_pred = self.model.predict(X)
		if(self.modelType == "LinearSVM"):
			y_pred = self.model.predict(X)
		if(self.modelType == "RbfSVM"):
			y_pred = self.model.predict(X)
		if(self.modelType == "AdaBoost"):
			y_pred = self.model.predict(X)
		if(self.modelType == "RandomForest"):
			y_pred = self.model.predict(X)
		if(self.modelType == "MLP"):
			y_pred = self.model.predict(X)


		return y_pred


	def get_metrics(self,y_true, y_pred):

		accuracy = accuracy_score(y_true,y_pred)
		precision = precision_score(y_true,y_pred)
		recall = recall_score(y_true,y_pred)
		f1 = f1_score(y_true,y_pred)

		return {"accuracy": accuracy,"precision":precision,"recall":recall, "f1":f1}






