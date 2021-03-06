import dataloader
import tomotopy as tp
from itertools import chain
import tqdm
import pandas as pd
import numpy as np

def train_slda_model(POSTPATH, LABELPATH, USERPATH, FOLDERPATH, topics=30, load_existing=False):
    if load_existing:
        user_to_post, post_to_metadata, filtered_data, sw_posts, sw_timestamps = dataloader.load_from_folder(FOLDERPATH)
        
    else:
        users = dataloader.load_user_subset_from_train(USERPATH, subset = 1000)
            
        user_to_post, post_to_words, post_to_metadata = dataloader.load_posts(POSTPATH, user_subset = users)
        post_to_label = dataloader.load_classification(LABELPATH, user_to_post, post_to_words, post_to_metadata)
        filtered_data, sw_posts, sw_timestamps = dataloader.filter_posts(post_to_label, post_to_metadata)
        
        filtered_data = dataloader.filter_near_SW(filtered_data, post_to_metadata, sw_timestamps)

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

def train_lda_model_from_data(filtered_data, topics=30):
    mdl = tp.LDAModel(k=topics)
    for data in tqdm.tqdm(filtered_data.keys()):
        mdl.add_doc(chain.from_iterable(filtered_data[data][1]))

    print("Beginning LDA training...")
    for i in range(0, 1000, 10):
        mdl.train(10)
        if(i % 100 == 0):
            print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    print("Finished Training")
    return mdl


def train_slda_model_from_data(filtered_data, topics=30):
    mdl = tp.SLDAModel(k=topics, vars=['b'])
    for data in tqdm.tqdm(filtered_data.keys()):
        mdl.add_doc(chain.from_iterable(filtered_data[data][1]), [1 if filtered_data[data][2] == 'd' else 0])

    print("Beginning sLDA training...")
    for i in range(0, 1000, 10):
        mdl.train(10)
        if(i % 100 == 0):
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
    print("Getting topic distributions...")
    for post in tqdm.tqdm(post_to_data.keys()):
        words = chain.from_iterable(post_to_data[post][1])
        label = post_to_data[post][2]
        post_to_topic_vec[post] = (mdl.infer(mdl.make_doc(words))[0], [1 if label == 'd' else 0])
    return post_to_topic_vec
    
def vectorize_data_set(mdl, FOLDERPATH):
    _, _, post_to_data, _, _ = dataloader.load_from_folder(FOLDERPATH)
    post_to_vec = get_topic_vecs(mdl, post_to_data)
    X = np.array([post_to_vec[post][0] for post in post_to_vec.keys()])
    Y = np.array([post_to_vec[post][1] for post in post_to_vec.keys()])
    return X,Y, post_to_vec 
    

# Usage Example
#POSTPATH = './crowd/train/shared_task_posts.csv'
#LABELPATH = './crowd/train/crowd_train.csv'
#USERPATH = './crowd/train/task_C_train.posts.csv'
#FOLDERPATH = './crowd_processed/'
    
#mdl = train_slda_model(POSTPATH, LABELPATH, USERPATH, FOLDERPATH, topics=30, load_existing=False)
# print_model_info(mdl)

# save to file
#mdl.save(FOLDERPATH + 'crowd_slda_model.bin')

# load from file
# mdl = tp.SLDAModel.load(FOLDERPATH + 'crowd_slda_model.bin')
# 

# inference: get vector of topic probabilities from list of words
# vec = mdl.infer(mdl.make_doc(["i", "feel", "very", "depressed"]))
# print(vec[0])

# use existing model to vectorize existing dataset and save it
# mdl = tp.SLDAModel.load(FOLDERPATH + 'crowd_slda_model.bin')

#X, Y = vectorize_data_set(mdl, FOLDERPATH)
#np.save(FOLDERPATH + "trainX.npy",X)
#np.save(FOLDERPATH + "trainY.npy",Y)
    
# loading existing dataset
# X = np.load(FOLDERPATH + "trainX.npy")
# Y = np.load(FOLDERPATH + "trainY.npy")