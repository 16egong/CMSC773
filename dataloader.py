import pandas as pd
import nltk
import tqdm
import string
import math
import sys

from nltk.tokenize import sent_tokenize, word_tokenize
from itertools import chain
from collections import defaultdict

def load_posts(PATH):

    data = pd.read_csv(PATH)

    #post_id,user_id,timestamp,subreddit,post_title,post_body
    index = data.index
    post_ids = data['post_id']
    user_ids = data['user_id']
    timestamps = data['timestamp']
    subreddits = data['subreddit']
    post_title = data['post_title']
    posts = data['post_body']

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
        post_to_metadata[post_ids[i]] = (timestamps[i], subreddits[i], post_title[i])
    
    return user_to_post, post_to_words, post_to_metadata
    
# Creates a post -> (user, tokens, label) dataset using a user -> label data file and the three lists created in load_posts
def load_classification(PATH, user_to_post, post_to_words, post_to_metadata):
    data = pd.read_csv(PATH)
    index = data.index
    user_ids = data['user_id']
    labels = data['label']
    
    post_to_label = {}
    for i in index:
        label = labels[i] if type(labels[i]) == type("") else "None"
        for j in user_to_post[user_ids[i]]:
            
            post_to_label[j] = (user_ids[i], post_to_words[j], label)
    
    return post_to_label
    
# Filters posts from mental health related subreddits
# Also returns a dict of users to timestamps of their SW posts and a dict of only SW posts
def filter_posts(post_to_label, post_to_metadata):
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
            filtered_dict[post] = post_to_label[post]
    return filtered_dict, SW_dict, users_to_SWtimestamps
    
# Usage example
#POSTPATH = './expert/expert_posts.csv'
#LABELPATH = './expert/expert.csv'
    
#user_to_post, post_to_words, post_to_metadata = load_posts(POSTPATH)
#post_to_label = load_classification(LABELPATH, user_to_post, post_to_words, post_to_metadata)
#filtered_data, sw_posts, sw_timestamps = filter_posts(post_to_label, post_to_metadata)

# For crowd data use:
# POSTPATH = './crowd/train/shared_task_posts.csv'
# LABELPATH = './crowd/train/crowd_train.csv'