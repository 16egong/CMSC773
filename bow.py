import numpy as np 
import nltk
import dataloader
import itertools
import pickle
import tqdm
from sklearn.decomposition import IncrementalPCA

def get_bow_features(POSTPATH, LABELPATH, USERPATH, FOLDERPATH, load_existing=False):

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


	words_to_index, index_to_words = generate_vocabulary(filtered_data)

	save_vocabulary(FOLDERPATH, index_to_words, words_to_index)

	post_to_vecs, pca = get_PCA_vectors_from_post_set(filtered_data, words_to_index)

	return post_to_vecs


def save_vocabulary(FOLDERPATH, index_to_words, words_to_index):

	with open(FOLDERPATH + 'bow_index2word.pickle', 'wb') as handle:
		pickle.dump(index_to_words, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open(FOLDERPATH + 'bow_word2index.pickle', 'wb') as handle:
		pickle.dump(words_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_vocabulary(FOLDERPATH):

	with open(FOLDERPATH + 'bow_word2index.pickle', 'rb') as handle:
		words_to_index = pickle.load(handle)
	with open(FOLDERPATH + 'bow_index2word.pickle', 'rb') as handle:
		index_to_words = pickle.load(handle)

	return words_to_index, index_to_words



def generate_vocabulary(post_to_data):
	
	words_to_index = {}
	index_to_words = {}

	words_to_index['<OOV>'] = 0
	index_to_words[0] = '<OOV>'

	next_index = 1

	for k,v in post_to_data.items():

		post = list(itertools.chain.from_iterable(v[1]))

		for token in post:

			if(token not in words_to_index):

				words_to_index[token] = next_index
				index_to_words[next_index] = token

				next_index += 1

	return words_to_index, index_to_words


def get_PCA_vectors_from_post_set(post_to_data, words_to_index, n_components=40, batch_size = 500):

	post_to_vec = {}


	vocab_len = len(words_to_index)

	i_pca = IncrementalPCA(n_components=n_components)

	vectors = []

	for k,v in tqdm.tqdm(post_to_data.items()):

		if(len(vectors) == batch_size):
			vectors = np.stack(vectors)

			i_pca.partial_fit(vectors)

			vectors = []


		vectors.append(get_vector_from_post(v,words_to_index))


	for k,v in tqdm.tqdm(post_to_data.items()):

		vector = get_vector_from_post(v,words_to_index)

		pca_vector = i_pca.transform(vector.reshape(1,-1))[0]

		post_to_vec[k] = (pca_vector, [1 if v[2] == 'd' else 0] )

	return pca_model, post_to_vec



def get_vector_from_post(post, words_to_index):

	vocab_len = len(words_to_index)

	vector = [0]*vocab_len

	post = [j for i in post[1] for j in i]

	for token in post:

		if(token not in words_to_index):

			vector[0] += 1

		else:

			vector[words_to_index[token]] += 1

		#print(vector)
		#print(sum(vector))

	return np.array(vector)













def main():


	POSTPATH = './Data/crowd/train/shared_task_posts.csv'
	LABELPATH = './Data/crowd/train/crowd_train.csv'
	USERPATH = './Data/crowd/train/task_C_train.posts.csv'
	FOLDERPATH = './Processing/crowd_processed/'
	get_bow_features(POSTPATH, LABELPATH, USERPATH, FOLDERPATH, load_existing=True)


if __name__ == '__main__':
	main()