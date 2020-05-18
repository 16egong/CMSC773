# CMSC773
Hi! Welcome to the H-DEV team. 

## Getting Started
### Create Filepaths
Please make sure to have the following directories created.
FILE_PATH = './Processing/'
```
mkdir Processing
cd Proccessing
```

TRAIN_FOLDERPATH = './Processing/crowd_processed/'
```
mkdir crowd_processed
```

DEV_FOLDERPATH ='./Processing/crowd_processed_test/'
```
mkdir crowd_processed_test
```

TEST_FOLDERPATH = './Processing/crowd_processed_expert/'
```
mkdir crowd_processed_expert
```

### Alter Filepaths
Our data paths are formatted as followed:

```
POSTPATH = './Data/crowd/train/shared_task_posts.csv'

LABELPATH = './Data/crowd/train/crowd_train.csv'
```

The following path is option if you would like to train a subset of the data:

```
USERPATH = './Data/crowd/train/task_C_train.posts.csv'
```
Either replace all of these variables with a path relative to your local computer or change the path to fit this format. 

### Modules
```
python3 -m pip install pandas
python3 -m pip install numpy
python3 -m pip install tqdm
python3 -m pip install sklearn
python3 -m pip install nltk
python3 -m pip install tomotopy
python3 -m pip install imblearn
python3 -m pip install empath

# for deep models
python3 -m pip install torch
python3 -m pip install transformers
```

## Notebeooks
Run the Reorganizd.ipynb notebook for a clean experience of what we did
Run the Wrapper.ipynb to see the original messier experimental notebook


### Example run
Run all the defined methods in the notebook and then scroll to the bottom of the notebook and create a new cell to configure a test. This example below shows creates feature vectors and trains a post classifier. Then the trained model or feature vectors used in testing via aggregating the users posts and creating user-level classifications. You have the ability to control the type of feature extraction (sLDA, LDA, BOW, sLDA and BOW, or empath). Only one of these selections should be true at a time except for sLDA and BOW which can be set to True at the same time. In addition, you can also select the type of post-level classifier as seen in post_clf_types. The first time you run a test you shoudl set train_data_processed to False in order to preprocess the data. However, after the first run, set train_data_processed=True to load the saved preprocessed data. Lastly you can change the number of topics for sLDA and LDA. 

For more advanced options, you may filter the time span of the subset data such as two weeks after or two weeks before etc. More to come

The following example runs an sLDA feature extractor and a Logistc Regression post classifier. There are more examples in the notebook. 
```
post_clf_types = ['LogReg', 'LinearSVM', 'RbfSVM', 'AdaBoost', 'RandomForest', 'MLP']
user_clf_type = ['Max']
model, vectors, pca_model, bow_vectors, word2index, p_clf = wrapped(train_data_processed=False, 
                                                                    sLDA=True, LDA=False, BOW=False, empath=False,
                                                                    user_clf_type=user_clf_type, 
                                                                    num_topics=40, post_clf_type=post_clf_types[0])
```

To run these models on the expert data, run the previous example with desired configuration. Then run something like below with matching configurations.

The following is an example of running the test of expert data after training a model under the sLDA + LogReg combination

```
expert_wrapped(test_data_processed=False, model=model, vectors=vectors, pca_model='', bow_vectors='',
               word2index='', p_clf=p_clf, num_topics=40, sLDA=True, LDA=False, BOW=False, empath=False)
```


# Reference
Shing, Han-Chin & Nair, Suraj & Zirikly, Ayah & Friedenberg, Meir & III, Hal & Resnik, Ps. (2018). Expert, Crowdsourced, and Machine Assessment of Suicide Risk via Online Postings. 25-36. 10.18653/v1/W18-0603. 
