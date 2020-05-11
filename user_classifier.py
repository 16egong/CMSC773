from collections import Counter
from collections import defaultdict
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class UserClassification:
    def __init__(self, user_to_post_label):
        self.user_to_post_label = user_to_post_label
    
    # Takes the most common label
    def argmax(self):
        user_to_label=defaultdict(list)
        for user_id in self.user_to_post_label:
            occurence_count = Counter(self.user_to_post_label[user_id])
            user_to_label[user_id]=occurence_count.most_common(1)[0][0]
        return user_to_label
    
    def get_metrics(self, y_true, y_pred):

        accuracy = accuracy_score(y_true,y_pred)
        precision = precision_score(y_true,y_pred)
        recall = recall_score(y_true,y_pred)
        f1 = f1_score(y_true,y_pred)

        return {"accuracy": accuracy,"precision":precision,"recall":recall, "f1":f1}
    
# Example Use
# import user_classification 
# FOLDERPATH = './processed/'
# user_to_post_label = aggregate_posts(FOLDERPATH, sample_data)
# test = user_classification.UserClassification(user_to_post_label)
# user_to_label = test.argmax()
# returns: defaultdict(list, {'1234': 'a', '3456': 'a'})
