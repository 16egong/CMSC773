import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import BertTokenizer


def proc(save_file, filter_data=True):
    post_data = pd.read_csv(POSTPATH)
    label_data = pd.read_csv(LABELPATH)

    model_version = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)

    subreddits_to_filter = ["Anger", "BPD", "EatingDisorders", "MMFB", "StopSelfHarm", "SuicideWatch", "addiction",
                            "alcoholism", "depression", "feelgood", "getting over it", "hardshipmates", "mentalhealth",
                            "psychoticreddit", "ptsd", "rapecounseling", "schizophrenia", "socialanxiety", "survivorsofabuse", "traumatoolbox"]

    if filter_data:
        print('Filtering out data from the following subreddits:\n\n{}\n'.format(subreddits_to_filter))

    user_clf_data_processed = {}

    for user_id in tqdm(label_data['user_id']):
        df = post_data[post_data['user_id'] == user_id]
        posts_list = []
        labels_list = []
        for i in range(len(df)):
            if df.iloc[i]['subreddit'] in subreddits_to_filter and filter_data:
                continue
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

        try:
            if lbl[0] in ['a', 'b', 'c']:
                current_label = 0
            elif lbl[0] == 'd':
                current_label = 1
            else:
                continue
        except IndexError:
            print('messy labels. moving on')
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


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load_dir", help="dir where data is stored, default: ~/Documents/CL2/umd_reddit_suicidewatch_dataset_v2",
        type=str, default='Documents/CL2/umd_reddit_suicidewatch_dataset_v2',
        )
    parser.add_argument(
        "--save_dir", help="dir to save processed data, default: ~/Documents/CL2/umd_reddit_suicidewatch_dataset_v2/processed_data",
        type=str, default='Documents/CL2/umd_reddit_suicidewatch_dataset_v2/processed_data',
        )
    parser.add_argument(
        "--filter", help="filter data from suicide watch",
        action="store_true",
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

        if args.filter:
            save_file = os.path.join(save_dir, 'filtered_test_data.npy')
        else:
            save_file = os.path.join(save_dir, 'test_data.npy')

    else:
        POSTPATH = os.path.join(load_dir, 'crowd/train/shared_task_posts.csv')
        LABELPATH = os.path.join(load_dir, 'crowd/train/crowd_train.csv')
        USERPATH = os.path.join(load_dir, 'crowd/train/task_C_train.posts.csv')

        if args.filter:
            save_file = os.path.join(save_dir, 'filtered_train_data.npy')
        else:
            save_file = os.path.join(save_dir, 'train_data.npy')

    proc(save_file, args.filter)
