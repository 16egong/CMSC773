from empath import Empath
import numpy as np 
import tqdm

lexicon = Empath()


def get_empath_vectors_from_post_set(post_to_data):

	post_to_vec = {}


	for k,v in tqdm.tqdm(post_to_data.items()):

		vector = get_vector_from_post(v)

		post_to_vec[k] = (vector, [1 if v[2] == 'd' else 0] )

	return post_to_vec



def get_vector_from_post(post):

	post = " ".join([j for i in post[1] for j in i])

	empath_dict = lexicon.analyze(post, normalize=True)

	if(empath_dict is None):
		return [0]*194

	empath_vect = [v for k,v in empath_dict.items()]


	return empath_vect


if __name__ == "__main__":
	x = [["hi"]] * 200
	print(x)

	get_empath_vectors_from_post_set({"x": x, "y":x, "f": x})
