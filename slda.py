import dataloader
import tomotopy as tp
from itertools import chain
import tqdm

POSTPATH = './expert/expert_posts.csv'
LABELPATH = './expert/expert.csv'
    
user_to_post, post_to_words, post_to_metadata = dataloader.load_posts(POSTPATH)
post_to_label = dataloader.load_classification(LABELPATH, user_to_post, post_to_words, post_to_metadata)
filtered_data, sw_posts, sw_timestamps = dataloader.filter_posts(post_to_label, post_to_metadata)

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
    print(mdl.get_topic_words(k, top_n=10))