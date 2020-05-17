import pandas as pd
import nltk
import tqdm
import string
import math
import sys
import pickle
import random

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
from itertools import chain
from collections import defaultdict

def load_posts(PATH, user_subset = None, append_title = False):

    data = pd.read_csv(PATH)

    #post_id,user_id,timestamp,subreddit,post_title,post_body
    index = data.index
    post_ids = data['post_id']
    user_ids = data['user_id']
    timestamps = data['timestamp']
    subreddits = data['subreddit']
    post_title = data['post_title']
    posts = data['post_body']
    
    print("Flagging empty posts")
    flag = [type(post) == type("") for post in tqdm.tqdm(posts)]
    
    if append_title:
        combined_posts = [None] * len(posts)
        print("Appending Titles")
        for i in tqdm.tqdm(index):
            combined_posts[i] = str(post_title[i]) + ((" . " + posts[i]) if flag[i] else "")
    
        posts = combined_posts
    
    if user_subset is not None:
        print("Filtering subset...")
        index = [i for i in tqdm.tqdm(index) if user_ids[i] in user_subset]

        # Tokenize speech into sentences
        print("Tokenizing sentences...")
        sents = dict((i,([sent for sent in sent_tokenize(posts[i])] if type(posts[i]) == type("") else [])) for i in tqdm.tqdm(index))
            
        # remove punctuation and normalize to lowercase
        print("Normalizing...")
        translate_dict = dict((ord(char), None) for char in string.punctuation)  
        sents = dict((i,[sent.translate(translate_dict).lower() for sent in sents[i]]) for i in tqdm.tqdm(index))

        # tokenize each sentence into words
        print("Tokenizing sentences into words...")
        words = dict((i,[nltk.word_tokenize(sent) for sent in sents[i]]) for i in tqdm.tqdm(index))
    
    else:
        # Tokenize speech into sentences
        print("Tokenizing sentences...")
        sents = [([sent for sent in sent_tokenize(post)] if type(post) == type("") else []) for post in tqdm.tqdm(posts)]
            
        # remove punctuation and normalize to lowercase
        print("Normalizing...")
        translate_dict = dict((ord(char), None) for char in string.punctuation)  
        sents = [[sent.translate(translate_dict).lower() for sent in sent_group] for sent_group in sents]

        # tokenize each sentence into words
        print("Tokenizing sentences into words...")
        words = [[nltk.word_tokenize(sent) for sent in sent_group] for sent_group in tqdm.tqdm(sents)]
    
    user_to_post = defaultdict(list)
    for i in index:
        user_to_post[user_ids[i]].append(post_ids[i])
    
    post_to_words = {}
    for i in index:
        post_to_words[post_ids[i]] = words[i]
        
    post_to_metadata = {}
    for i in index:
        post_to_metadata[post_ids[i]] = (timestamps[i], subreddits[i], post_title[i], flag[i])
    
    return user_to_post, post_to_words, post_to_metadata
    
# Creates a post -> (user, tokens, label) dataset using a user -> label data file and the three lists created in load_posts
def load_classification(PATH, user_to_post, post_to_words, post_to_metadata, user_subset = None):
    data = pd.read_csv(PATH)
    index = data.index
    user_ids = data['user_id']
    labels = data['label']
    
    print(len(user_to_post))
    
    post_to_label = {}
    for i in index:
        if (user_subset is None or user_ids[i] in user_subset):
            label = labels[i] if type(labels[i]) == type("") else "None"
            for j in user_to_post[user_ids[i]]:
                
                post_to_label[j] = (user_ids[i], post_to_words[j], label)
    
    return post_to_label
    
# Filters posts from mental health related subreddits
# Also returns a dict of users to timestamps of their SW posts and a dict of only SW posts
def filter_posts(post_to_label, post_to_metadata, filter_images=False):
    subreddits_to_filter = ["Anger", "BPD", "EatingDisorders", "MMFB", "StopSelfHarm", "SuicideWatch", "addiction", 
                            "alcoholism", "depression", "feelgood", "getting over it", "hardshipmates", "mentalhealth", 
                            "psychoticreddit", "ptsd", "rapecounseling", "schizophrenia", "socialanxiety", "survivorsofabuse", "traumatoolbox"]
    
    filtered_dict = {}
    SW_dict = {}
    users_to_SWtimestamps = defaultdict(list)
    for post in post_to_label.keys():
        user = post_to_label[post][0]
        subreddit = post_to_metadata[post][1]
        
        if subreddit == "SuicideWatch":
            users_to_SWtimestamps[user].append(post_to_metadata[post][0])
            SW_dict[post] = post_to_label[post]
            
        if subreddit not in subreddits_to_filter:
            img_flag = post_to_metadata[post][3]
            if(not filter_images or img_flag):
                filtered_dict[post] = post_to_label[post]
    return filtered_dict, SW_dict, users_to_SWtimestamps
    
def filter_near_SW(post_to_label, post_to_metadata, sw_timestamps, thresh = 604800 * 2, filter_control = False, filter_direction = False, filter_first = False):
    filtered_dict = {}
    print("Filtering posts far away from SW posts...")
    
    control_user_times = {}
    keys = post_to_label.keys()
    if(filter_control):
        keys = list(keys)
        random.shuffle(keys)
        
    for post in tqdm.tqdm(keys):
        user, words, label = post_to_label[post]
        time = post_to_metadata[post][0]
        SWtimes = sw_timestamps[user]
        
        if (filter_control):
            if len(SWtimes) == 0:
                if user in control_user_times:
                    SWtimes = control_user_times[user]
                else:
                    control_user_times[user] = [time]
                    SWtimes = [time]
        
        if len(SWtimes) > 0:
            near = False
            if filter_first:
                SWtimes = [min(SWtimes)]
            for time_SW in SWtimes:
                diff = time_SW - time
                if not filter_direction:
                    if abs(diff) < thresh:
                        near = True
                else:
                    if diff > 0 and diff < thresh:
                        near = True
            if near:
                filtered_dict[post] = (user, words, label)
        else:
            filtered_dict[post] = (user, words, label)
    return filtered_dict
    
# Filters post -> (user, tokens, label) dicts for English stopwords
def filter_stopwords(post_to_label):
    stop_words = set(stopwords.words('english')) 
    
    filtered_dict = {}
    for key in post_to_label.keys():
        tokens = post_to_label[key][1]
        new_tokens = [[word for word in sent if word not in stop_words] for sent in tokens]
        
        filtered_dict[key] = (post_to_label[key][0], new_tokens, post_to_label[key][2])
    return filtered_dict
    
def save_to_folder(FOLDERPATH, user_to_post, post_to_metadata, post_to_label, sw_posts, sw_timestamps):
    with open(FOLDERPATH + 'utp.pickle', 'wb') as handle:
        pickle.dump(user_to_post, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(FOLDERPATH + 'ptm.pickle', 'wb') as handle:
        pickle.dump(post_to_metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(FOLDERPATH + 'data.pickle', 'wb') as handle:
        pickle.dump(post_to_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(FOLDERPATH + 'swd.pickle', 'wb') as handle:
        pickle.dump(sw_posts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(FOLDERPATH + 'swt.pickle', 'wb') as handle:
        pickle.dump(sw_timestamps, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_from_folder(FOLDERPATH):
    with open(FOLDERPATH + 'utp.pickle', 'rb') as handle:
        user_to_post = pickle.load(handle)
    with open(FOLDERPATH + 'ptm.pickle', 'rb') as handle:
        post_to_metadata = pickle.load(handle)
    with open(FOLDERPATH + 'data.pickle', 'rb') as handle:
        post_to_label = pickle.load(handle)
    with open(FOLDERPATH + 'swd.pickle', 'rb') as handle:
        sw_posts = pickle.load(handle)
    with open(FOLDERPATH + 'swt.pickle', 'rb') as handle:
        sw_timestamps = pickle.load(handle)
    return user_to_post, post_to_metadata, post_to_label, sw_posts, sw_timestamps
    
def load_user_subset_from_train(PATH, subset = 100):
    data = pd.read_csv(PATH)
    index = data.index
    user_ids = data['user_id']
    
    user_list = list(set(user_ids))
    
    return set(user_list[:min(subset, len(user_list))])
    
# Usage example
# POSTPATH = './expert/expert_posts.csv'
# LABELPATH = './expert/expert.csv'
# user_to_post, post_to_words, post_to_metadata = load_posts(POSTPATH, append_title=True)
# post_to_label = load_classification(LABELPATH, user_to_post, post_to_words, post_to_metadata)
# filtered_data, sw_posts, sw_timestamps = filter_posts(post_to_label, post_to_metadata)
# print(sw_posts["2il6xf"])

# Filtering posts far away from SW_posts:
# filtered_data = filter_near_SW(filtered_data, post_to_metadata, sw_timestamps)
 
# Saving all data structures to a folder (make sure the folder exists and you pass this method these 5 data structures)
# FOLDERPATH = './processed/'
# save_to_folder(FOLDERPATH, user_to_post, post_to_metadata, filtered_data, sw_posts, sw_timestamps)

# Loading all data structures from a folder
# user_to_post, post_to_metadata, filtered_data, sw_posts, sw_timestamps = load_from_folder(FOLDERPATH)

# Filter for stop words:
# filter_stopwords(filtered_data)

# For crowd data use:
# POSTPATH = './crowd/train/shared_task_posts.csv'
# LABELPATH = './crowd/train/crowd_train.csv'

# Loading a subset of data, the subset size is 100, which means a subset of 100 users from given classification file.
# POSTPATH = './Data/crowd/train/shared_task_posts.csv'
# LABELPATH = './Data/crowd/train/crowd_train.csv'
# USERPATH = './Data/crowd/train/task_C_train.posts.csv'
# users = load_user_subset_from_train(USERPATH, subset = 100)
# user_to_post, post_to_words, post_to_metadata = load_posts(POSTPATH, user_subset = users)
# post_to_label = load_classification(LABELPATH, user_to_post, post_to_words, post_to_metadata, user_subset = users)
# filtered_data, sw_posts, sw_timestamps = filter_posts(post_to_label, post_to_metadata, filter_images=True)
# print(filtered_data)