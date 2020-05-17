import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import BertTokenizer


def proc(save_file, mode='post'):
    post_data = pd.read_csv(POSTPATH)
    label_data = pd.read_csv(LABELPATH)

    model_version = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)

    if mode == 'post':
        post_clf_data_processed = {}

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

                post_clf_data_processed.update({df.iloc[i]['post_id']: data_dict})

        labels_all = []
        train_docs = []
        for key, dict_ in post_clf_data_processed.items():
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
                labels_all.append(0)
                train_docs.append((post_body, post_title))
            elif label == 'd':
                labels_all.append(1)
                train_docs.append((post_body, post_title))
            else:
                continue

        input_ids_all = []
        attention_masks_all = []
        for sentence_a, sentence_b in tqdm(train_docs):
            if sentence_a is np.nan:
                sent = sentence_b
            elif sentence_b is np.nan:
                sent = sentence_a
            else:
                sent = sentence_a + '. ' + sentence_b
            inputs = tokenizer.encode_plus(sent[:512], return_tensors='pt',
                                           add_special_tokens=True, pad_to_max_length=True)

            input_ids = inputs['input_ids']
            attention_masks = inputs['attention_mask']

            input_ids_all.append(input_ids)
            attention_masks_all.append(attention_masks)

        x = (np.concatenate(input_ids_all), np.concatenate(attention_masks_all))
        y = np.reshape(labels_all, (-1,))

        np.save(save_file, (x, y))

    elif mode == 'user':
        user_clf_data_processed = {}

        for user_id in tqdm(label_data['user_id']):
            df = post_data[post_data['user_id'] == user_id]
            posts_list = []
            labels_list = []
            for i in range(len(df)):
                post_title = df.iloc[i]['post_title']
                post_body = df.iloc[i]['post_body']
                if post_title is np.nan:
                    doc = post_body
                elif post_body is np.nan:
                    doc = post_title
                elif post_title is np.nan and post_body is np.nan:
                    continue
                else:
                    doc = post_title + '. ' + post_body

                posts_list.append(doc)
                labels_list.append(label_data[label_data['user_id'] == user_id].label.item())

            user_clf_data_processed.update({user_id: (posts_list, labels_list)})

        user_clf_data_all = {}
        for user_id, data_tuple in user_clf_data_processed.items():
            labels_list = data_tuple[1]
            lbl = np.unique(labels_list)

            if lbl[0] in ['a', 'b', 'c']:
                current_label = 0
            elif lbl[0] == 'd':
                current_label = 1
            else:
                continue

            try:
                inputs_list = [tokenizer.encode_plus(
                    item[:512],
                    return_tensors='pt',
                    add_special_tokens=True,
                    pad_to_max_length=True) for item in data_tuple[0]]
            except TypeError:
                print('messy data point. moving on')
                continue

            input_ids_list = [item['input_ids'] for item in inputs_list]
            attention_masks_list = [item['attention_mask'] for item in inputs_list]

            input_ids = torch.cat(input_ids_list)
            attention_masks = torch.cat(attention_masks_list)

            _data_dict = {
                'input_ids': to_np(input_ids),
                'attention_masks': to_np(attention_masks),
                'label': current_label,
            }

            user_clf_data_all.update({user_id: _data_dict})

        np.save(save_file, user_clf_data_all)

    else:
        raise ValueError("Invalid mode")


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", help="classification mode in ['user', 'post']", type=str)

    parser.add_argument(
        "--load_dir", help="dir where data is stored, default: ~/Documents/CL2/umd_reddit_suicidewatch_dataset_v2",
        type=str, default='Documents/CL2/umd_reddit_suicidewatch_dataset_v2',
        )
    parser.add_argument(
        "--save_dir", help="dir to save processed data, default: ~/Documents/CL2/umd_reddit_suicidewatch_dataset_v2/processed_data",
        type=str, default='Documents/CL2/umd_reddit_suicidewatch_dataset_v2/processed_data',
        )
    parser.add_argument(
        "--test", help="train/test",
        action="store_true",
        )

    args = parser.parse_args()

    if args.mode not in ['post', 'user']:
        raise ValueError("Invalid mode entered")

    home_dir = os.environ['HOME']
    load_dir = os.path.join(home_dir, args.load_dir)
    save_dir = os.path.join(home_dir, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if args.test:
        POSTPATH = os.path.join(load_dir, 'crowd/test/shared_task_posts_test.csv')
        LABELPATH = os.path.join(load_dir, 'crowd/test/crowd_test_C.csv')
        USERPATH = os.path.join(load_dir, 'crowd/test/task_C_test.posts.csv')

        save_file = os.path.join(save_dir, '{}_clf_test_data.npy'.format(args.mode))

    else:
        POSTPATH = os.path.join(load_dir, 'crowd/train/shared_task_posts.csv')
        LABELPATH = os.path.join(load_dir, 'crowd/train/crowd_train.csv')
        USERPATH = os.path.join(load_dir, 'crowd/train/task_C_train.posts.csv')

        save_file = os.path.join(save_dir, '{}_clf_train_data.npy'.format(args.mode))

    proc(save_file, args.mode)
