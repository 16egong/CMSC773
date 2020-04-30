from collections import Counter
from collections import defaultdict

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
    
    
    
# Example Use
# import user_classification 
# FOLDERPATH = './processed/'
# user_to_post_label = aggregate_posts(FOLDERPATH, sample_data)
# test = user_classification.UserClassification(user_to_post_label)
# user_to_label = test.argmax()
# returns: defaultdict(list, {'1234': 'a', '3456': 'a'})
