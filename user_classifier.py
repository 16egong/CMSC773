from collections import Counter
from collections import defaultdict
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

class UserClassification:
    def __init__(self, user_to_post_label):
        self.user_to_post_label = user_to_post_label
        self.user_to_label=defaultdict(list)
    # Takes the most common label
    def argmax(self):
        for user_id in self.user_to_post_label:
            occurence_count = Counter(self.user_to_post_label[user_id])
            self.user_to_label[user_id]=occurence_count.most_common(1)[0][0]
        return self.user_to_label
    
    def minimum(self, num=1):
        for user_id in self.user_to_post_label:
            occurence_count = Counter(self.user_to_post_label[user_id])
            if occurence_count[1] >= num:
                self.user_to_label[user_id]=1
            else:
                self.user_to_label[user_id]=0
        return self.user_to_label
    
    # Classify based on a threshold
    def threshold(self, percent=5):
        for user_id in self.user_to_post_label:
            occurence_count = Counter(self.user_to_post_label[user_id])
            if (float(occurence_count[1])/float(occurence_count[1] + occurence_count[0])) > percent:
                self.user_to_label[user_id]=1
            else:
                self.user_to_label[user_id]=0
                
    def find_threshold(self, user_to_y_test):
        f_score = []
        percent = np.linspace(.5,1,51)
        for p in percent:
            self.threshold(percent=p)
            user_y_test = []
            user_y_pred_test = []
            for user_id in self.user_to_label:
                user_y_test.append(user_to_y_test[user_id])
                user_y_pred_test.append(self.user_to_label[user_id])
            results = self.get_metrics(user_y_test, user_y_pred_test)
            f_score.append(results['f1'])
            
        print('percent: ', percent)
        print('f_score: ', f_score)
        plt.plot(percent, f_score)
        plt.ylabel('f_score')
        
        print('max: ', percent[np.argmax(np.array(f_score))])
        plt.show()
        p = percent[np.argmax(np.array(f_score))]
        self.threshold(percent=p)
        user_y_test = []
        user_y_pred_test = []
        for user_id in self.user_to_label:
            user_y_test.append(user_to_y_test[user_id])
            user_y_pred_test.append(self.user_to_label[user_id])
        results = self.get_metrics(user_y_test, user_y_pred_test)
        return results
        
    def get_metrics(self, y_true, y_pred):

        accuracy = accuracy_score(y_true,y_pred)
        precision = precision_score(y_true,y_pred)
        recall = recall_score(y_true,y_pred)
        f1 = f1_score(y_true,y_pred)

        return {"accuracy": accuracy,"precision": precision,"recall":recall, "f1":f1}
    
# Example Use
# import user_classification 
# FOLDERPATH = './processed/'
# user_to_post_label = aggregate_posts(FOLDERPATH, sample_data)
# test = user_classification.UserClassification(user_to_post_label)
# user_to_label = test.argmax()
# returns: defaultdict(list, {'1234': 'a', '3456': 'a'})
