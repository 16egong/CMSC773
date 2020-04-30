import sys
import pickle

from collections import defaultdict

# takes the classifcations from post level and groups them by user
def aggregate_posts(FOLDERPATH: str, post_classifications: defaultdict) -> defaultdict:
    user_to_post_label = defaultdict(list)
    
    # assumes post_classications contains user_id and the classification as y_prime
    for p in post_classifications:
        post = post_classifications[p]
        user_to_post_label[post['user_id']].append(post['y_prime'])
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
# post_abcde = {}
# post_abcde['text'] = "hey yo"
# post_abcde['user_id'] = "1234"
# post_abcde['y'] = "d"
# post_abcde['y_prime'] = "d"

# post_qwert = {}
# post_qwert['text'] = "yo"
# post_qwert['user_id'] = "1234"
# post_qwert['y'] = "d"
# post_qwert['y_prime'] = "a"

# post_efghi = {}
# post_efghi['text'] = "life poop"
# post_efghi['user_id'] = "1234"
# post_efghi['y'] = "d"
# post_efghi['y_prime'] = "d"

# post_zxcvb = {}
# post_zxcvb['post'] = "hungry"
# post_zxcvb['user_id'] = "3456"
# post_zxcvb['y'] = "a"
# post_zxcvb['y_prime'] = "a"


# sample_data['2j6par'] = post_abcde
# sample_data['2j6qn6'] = post_efghi
# sample_data['2daejg '] = post_qwert
# sample_data['wrfw1'] = post_zxcvb

# FOLDERPATH = './processed/'

# fuser_to_post_label = aggregate(FOLDERPATH, sample_data)

# Loading users with post labels from a folder
# uuser_to_post_label = load_aggregate(FOLDERPATH)