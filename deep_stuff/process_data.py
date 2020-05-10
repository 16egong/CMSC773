import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import torch
from transformers import BertForSequenceClassification, BertTokenizer


def proc(load_dir, save_file='processed_data.npy', spearate_title_body=False):

    post_data = pd.read_csv(POSTPATH)
    label_data = pd.read_csv(LABELPATH)

    data_processed = {}

    for user_id in tqdm(label_data['user_id']):
        df = post_data[post_data['user_id'] == user_id]
        for i in range(len(df)):
            data_dict = {
                'user_id': user_id,
                'post_title': df.iloc[i]['post_title'],
                'post_body': df.iloc[i]['post_body'],
                'label': label_data[label_data['user_id'] == user_id].label.item(),
                'subreddit': df.iloc[i]['subreddit'],
                'timestamp': df.iloc[i]['timestamp'],
            }

            data_processed.update({df.iloc[i]['post_id']: data_dict})


    a_counts, b_counts, c_counts, d_counts = Counter(), Counter(), Counter(), Counter()
    no_label_couts = Counter()

    for key, dict_ in data_processed.items():
        lbl_ = dict_['label']
        uid_ = dict_['user_id']

        if lbl_ == 'a':
            a_counts[uid_] += 1

        elif lbl_ == 'b':
            b_counts[uid_] += 1

        elif lbl_ == 'c':
            c_counts[uid_] += 1

        elif lbl_ == 'd':
            d_counts[uid_] += 1

        else:
            no_label_couts[uid_] += 1


    train_labels = []
    train_docs = []

    for key, dict_ in data_processed.items():
        label = dict_['label']

        if dict_['post_body'] is not np.nan:
            post_body = dict_['post_body']
        else:
            post_body = ""

        if dict_['post_title'] is not np.nan:
            post_title = dict_['post_title']
        else:
            post_title = ""


        if label in ['a', 'b', 'c']:
            train_labels.append(0)
            train_docs.append((post_body, post_title))
        elif label == 'd':
            train_labels.append(1)
            train_docs.append((post_body, post_title))
        else:
            continue


    model_version = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)


    input_ids_all = []
    attention_masks_all = []
    token_type_ids_all = []

    for sentence_a, sentence_b in tqdm(train_docs):
        if spearate_title_body:
            if sentence_a is np.nan:
                inputs = tokenizer.encode_plus(sentence_b[:512], return_tensors='pt',
                                               add_special_tokens=True, pad_to_max_length=True)
                token_type_ids = None
            elif sentence_b is np.nan:
                inputs = tokenizer.encode_plus(sentence_a[:512], return_tensors='pt',
                                               add_special_tokens=True, pad_to_max_length=True)
                token_type_ids = None
            else:
                inputs = tokenizer.encode_plus(sentence_a[:512], sentence_b[:512], return_tensors='pt',
                                               add_special_tokens=True, pad_to_max_length=True)
                token_type_ids = inputs['token_type_ids']
        else:
            if sentence_a is np.nan:
                sent = sentence_b
            elif sentence_b is np.nan:
                sent = sentence_a
            else:
                sent = sentence_a + '. ' + sentence_b
            inputs = tokenizer.encode_plus(sent[:512], return_tensors='pt',
                                           add_special_tokens=True, pad_to_max_length=True)
            token_type_ids = None

        input_ids = inputs['input_ids']
        attention_masks = inputs['attention_mask']

        input_ids_all.append(input_ids)
        attention_masks_all.append(attention_masks)
        token_type_ids_all.append(token_type_ids)

    x = (np.concatenate(input_ids_all), np.concatenate(attention_masks_all))
    if any(token_type_ids_all):
        x += (np.concatenate(token_type_ids_all),)
    y = np.reshape(train_labels, (-1, 1))

    np.save(save_file, (x, y))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load_dir", help="dir where data is stored, default: ~/Documents/CL2/umd_reddit_suicidewatch_dataset_v2",
        type=str, default='Documents/CL2/umd_reddit_suicidewatch_dataset_v2',
        )
    parser.add_argument(
        "--save_dir", help="dir to save processed data: ~/Documents/CL2/umd_reddit_suicidewatch_dataset_v2/processed_data",
        type=str, default='Documents/CL2/umd_reddit_suicidewatch_dataset_v2/processed_data',
        )
    parser.add_argument(
        "--test", help="train/test",
        action="store_true",
        )

    args = parser.parse_args()

    home_dir = os.environ['HOME']
    load_dir = os.path.join(home_dir, args.load_dir)
    save_dir = os.path.join(home_dir, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if args.test:
        POSTPATH = os.path.join(load_dir, 'crowd/test/shared_task_posts_test.csv')
        LABELPATH = os.path.join(load_dir, 'crowd/test/crowd_test_C.csv')
        USERPATH = os.path.join(load_dir, 'crowd/test/task_C_test.posts.csv')

        save_file = os.path.join(save_dir, 'test_data.npy')

    else:
        POSTPATH = os.path.join(load_dir, 'crowd/train/shared_task_posts.csv')
        LABELPATH = os.path.join(load_dir, 'crowd/train/crowd_train.csv')
        USERPATH = os.path.join(load_dir, 'crowd/train/task_C_train.posts.csv')

        save_file = os.path.join(save_dir, 'train_data.npy')

    proc(load_dir, save_file)
