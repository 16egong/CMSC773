import dataloader
import tomotopy as tp
from itertools import chain
import tqdm
import pandas as pd
import numpy

POSTPATH = './crowd/train/shared_task_posts.csv'
LABELPATH = './crowd/train/crowd_train.csv'
USERPATH = './crowd/train/task_C_train.posts.csv'

users = dataloader.load_user_subset_from_train(USERPATH, subset = 1000)
    
user_to_post, post_to_words, post_to_metadata = dataloader.load_posts(POSTPATH, user_subset = users)
post_to_label = dataloader.load_classification(LABELPATH, user_to_post, post_to_words, post_to_metadata)
filtered_data, sw_posts, sw_timestamps = dataloader.filter_posts(post_to_label, post_to_metadata)

filtered_data = dataloader.filter_stopwords(filtered_data)
sw_posts = dataloader.filter_stopwords(sw_posts)

FOLDERPATH = './crowd_processed/'
dataloader.save_to_folder(FOLDERPATH, user_to_post, post_to_metadata, filtered_data, sw_posts, sw_timestamps)

#FOLDERPATH = './processed/'
#user_to_post, post_to_metadata, filtered_data, sw_posts, sw_timestamps = dataloader.load_from_folder(FOLDERPATH)
#filtered_data = dataloader.filter_stopwords(filtered_data)
#sw_posts = dataloader.filter_stopwords(sw_posts)

mdl = tp.SLDAModel(k=20, vars=['b'])
for data in tqdm.tqdm(filtered_data.keys()):
    mdl.add_doc(chain.from_iterable(filtered_data[data][1]), [1 if filtered_data[data][2] == 'd' else 0])

for i in range(0, 1000, 10):
    mdl.train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

for k in range(mdl.k):
    print('Top 10 words of topic #{}'.format(k))
    top_words = mdl.get_topic_words(k, top_n=10)
    top_wordsDoc = mdl.get_topic_words(k, top_n=100)
    words = mdl.make_doc([word for (word, float) in top_wordsDoc])

slda_coefficients = mdl.get_regression_coef(0)
data = []
for k in range(mdl.k):
    top_words = mdl.get_topic_words(k, top_n=10)
    words = [word for (word, float) in top_words]
    words = ", ".join(words)
    data.append([words, slda_coefficients[k]])
    
indices = np.array(slda_coefficients).argsort()
data = np.array(data)
data = data[indices]

pd.DataFrame(data, columns=["Topic", "Suicidality Coefficient"])