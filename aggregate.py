import sys
import pickle

from collections import defaultdict

# takes the classifcations from post level and groups them by user
def aggregate_posts(FOLDERPATH: str, post_classifications: defaultdict) -> defaultdict:
    user_to_post_label = defaultdict(list)
    
    # assumes post_classications contains user_id and the classification as y_prime
    for post_id in post_classifications:
        post = post_classifications[post_id]
        user_to_post_label[post[0]].append(post[1])
    save_aggregate(FOLDERPATH, user_to_post_label)
    return user_to_post_label

def save_aggregate(FOLDERPATH, user_to_post_label):
    with open(FOLDERPATH + 'utpl.pickle', 'wb') as handle:
        pickle.dump(user_to_post_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved aggreagation of user to post labels...")
        
def load_aggregate(FOLDERPATH):
    with open(FOLDERPATH + 'utpl.pickle', 'rb') as handle:
        user_to_post_label = pickle.load(handle)
    print("Loaded aggreagation of user to post labels...")
    return user_to_post_label


# Usage example
# sample_data = {}



# sample_data['2j6par'] = (user_id, y_pred)
# sample_data['2j6qn6'] = (user_id, y_pred)
# sample_data['2daejg '] = (user_id, y_pred)
# sample_data['wrfw1'] = (user_id, y_pred)

# FOLDERPATH = './processed/'

# fuser_to_post_label = aggregate(FOLDERPATH, sample_data)

# Loading users with post labels from a folder
# uuser_to_post_label = load_aggregate(FOLDERPATH)