import dataloader
import tomotopy as tp
from itertools import chain
import tqdm
import pandas as pd
import numpy as np

def train_slda_model(POSTPATH, LABELPATH, USERPATH, FOLDERPATH, topics=30, load_existing=False):
    if load_existing:
        user_to_post, post_to_metadata, filtered_data, sw_posts, sw_timestamps = dataloader.load_from_folder(FOLDERPATH)
        filtered_data = dataloader.filter_stopwords(filtered_data)
        sw_posts = dataloader.filter_stopwords(sw_posts)
        
    else:
        users = dataloader.load_user_subset_from_train(USERPATH, subset = 1000)
            
        user_to_post, post_to_words, post_to_metadata = dataloader.load_posts(POSTPATH, user_subset = users)
        post_to_label = dataloader.load_classification(LABELPATH, user_to_post, post_to_words, post_to_metadata)
        filtered_data, sw_posts, sw_timestamps = dataloader.filter_posts(post_to_label, post_to_metadata)

        filtered_data = dataloader.filter_stopwords(filtered_data)
        sw_posts = dataloader.filter_stopwords(sw_posts)

        dataloader.save_to_folder(FOLDERPATH, user_to_post, post_to_metadata, filtered_data, sw_posts, sw_timestamps)

    mdl = tp.SLDAModel(k=topics, vars=['b'])
    for data in tqdm.tqdm(filtered_data.keys()):
        mdl.add_doc(chain.from_iterable(filtered_data[data][1]), [1 if filtered_data[data][2] == 'd' else 0])

    print("Beginning sLDA training...")
    for i in range(0, 1000, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    print("Finished Training")
    return mdl


def print_model_info(mdl):

    slda_coefficients = mdl.get_regression_coef(0)
    data = []
    for k in range(mdl.k):
        top_words = mdl.get_topic_words(k, top_n=40)
        words = [word for (word, float) in top_words]
        words = ", ".join(words)
        data.append([words, slda_coefficients[k]])
        
    indices = np.array(slda_coefficients).argsort()
    data = np.array(data)
    data = data[indices]

    data = pd.DataFrame(data, columns=["Topic", "Suicidality Coefficient"])
    print(data)
    return data

def get_topic_vecs(mdl, post_to_data):
    post_to_topic_vec = {}
    for post in post_to_data.keys():
        words = chain.from_iterable(post_to_data[post][1])
        label = post_to_data[post][2]
        post_to_topic_vec[post] = (mdl.infer(mdl.make_doc(words)), label)
    return post_to_topic_vec
    
# Usage Example
# POSTPATH = './crowd/train/shared_task_posts.csv'
# LABELPATH = './crowd/train/crowd_train.csv'
# USERPATH = './crowd/train/task_C_train.posts.csv'
# FOLDERPATH = './crowd_processed/'
    
# mdl = train_slda_model(POSTPATH, LABELPATH, USERPATH, FOLDERPATH, topics=30, load_existing=True)
# print_model_info(mdl)

# save from file
# mdl.save(FOLDERPATH + 'crowd_slda_model.bin')
# load from file
# mdl = tp.SLDAModel.load(FOLDERPATH + 'crowd_slda_model.bin')
# 

# inference: get vector of topic probabilities from list of words
# vec = mdl.infer(mdl.make_doc(["i", "feel", "very", "depressed"]))
# print(vec[0])
