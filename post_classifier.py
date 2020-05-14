import numpy as np 
from sklearn import svm
from sklearn import ensemble
from sklearn import neural_network
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt

class PostClassification:


	def __init__(self, modelType, gridSearch = False):

		self.modelType = modelType

		if(modelType == "LogReg"):
			self.model = linear_model.LogisticRegression(class_weight = 'balanced', C = 0.2)
		if(modelType == "LinearSVM"):
			self.model = svm.SVC(class_weight = 'balanced',kernel='linear', probability = True, C = 1)
		if(modelType == "RbfSVM"):
			self.model = svm.SVC(class_weight = 'balanced',kernel='rbf', probability = True,  C=1)
		if(modelType == "AdaBoost"):
			self.model = ensemble.AdaBoostClassifier(n_estimators=1000)
		if(modelType == "RandomForest"):
			self.model = ensemble.RandomForestClassifier(n_estimators=200, max_depth = 100, min_samples_leaf = 1)
		if(modelType == "MLP"):
			self.model = neural_network.MLPClassifier(hidden_layer_sizes=(32,32,10), max_iter = 500, activation = 'relu', verbose=True)		



	# parameters
	#
	# X: numpy array of training data: (num_observations, num_features)
	# y: numpy array of ground truth values: (num_observations)
	def train(self,X,y):

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


	def test_probability(self,X):

		if(self.modelType == "LogReg"):
			y_prob = self.model.predict_proba(X)
		if(self.modelType == "LinearSVM"):
			y_prob = self.model.predict_proba(X)
		if(self.modelType == "RbfSVM"):
			y_prob = self.model.predict_proba(X)
		if(self.modelType == "AdaBoost"):
			y_prob = self.model.predict_proba(X)
		if(self.modelType == "RandomForest"):
			y_prob = self.model.predict_proba(X)
		if(self.modelType == "MLP"):
			y_prob = self.model.predict_proba(X)

		return y_prob


	def train_grid_search_CV(self,X,y, param_dict, groups):

		if(self.modelType == "LogReg"):
			self.model= model_selection.GridSearchCV(self.model, param_dict,verbose=2)
			self.model.fit(X,y)
		if(self.modelType == "RbfSVM"):
			self.model= model_selection.GridSearchCV(self.model, param_dict,verbose=2)
			self.model.fit(X,y)
		if(self.modelType == "LinearSVM"):
			self.model= model_selection.GridSearchCV(self.model, param_dict,verbose=2)
			self.model.fit(X,y)
		if(self.modelType == "AdaBoost"):
			self.model= model_selection.GridSearchCV(self.model, param_dict,verbose=2)
			self.model.fit(X,y)
		if(self.modelType == "RandomForest"):
			self.model= model_selection.GridSearchCV(self.model, param_dict,verbose=2)
			self.model.fit(X,y)
		if(self.modelType == "MLP"):
			self.model= model_selection.GridSearchCV(self.model, param_dict,verbose=2)
			self.model.fit(X,y)

		print(self.model.best_params_)



	def get_metrics(self,y_true, y_pred):

		accuracy = accuracy_score(y_true,y_pred)
		precision = precision_score(y_true,y_pred)
		recall = recall_score(y_true,y_pred)
		f1 = f1_score(y_true,y_pred)

		return {"accuracy": accuracy,"precision":precision,"recall":recall, "f1":f1}






