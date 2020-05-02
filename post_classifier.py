import numpy as np 
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class PostClassification:


	def __init__(self, modelType):

		self.modelType = modelType

		if(modelType == "LinearSVM"):
			self.model = svm.SVC(class_weight = 'balanced',kernel='linear')
		if(modelType == "RbfSVM"):
			self.model = svm.SVC(class_weight = 'balanced',kernel='rbf')



	# parameters
	#
	# X: numpy array of training data: (num_observations, num_features)
	# y: numpy array of ground truth values: (num_observations)
	def train(self,X,y):

		if(self.modelType == "LinearSVM"):
			self.model.fit(X,y)
		if(self.modelType == "RbfSVM"):
			self.model.fit(X,y)

	def test(self,X):

		if(self.modelType == "LinearSVM"):
			y_pred = self.model.predict(X)
		if(self.modelType == "RbfSVM"):
			y_pred = self.model.predict(X)


		return y_pred


	def get_metrics(self,y_true, y_pred):

		accuracy = accuracy_score(y_true,y_pred)
		precision = precision_score(y_true,y_pred)
		recall = recall_score(y_true,y_pred)
		f1 = f1_score(y_true,y_pred)

		return {"accuracy": accuracy,"precision":precision,"recall":recall, "f1":f1}






