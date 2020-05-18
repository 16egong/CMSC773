import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def load_data(file_name, base_dir=None):
    """
    :param file_name: e.g. train_data.npy
    :param base_dir: where the data is at
    :return: [np.ndarray] input_ids, attention_masks, labels, user_ids
    """
    if base_dir is None:
        base_dir = 'Documents/CL2/umd_reddit_suicidewatch_dataset_v2/processed_data'
    load_dir = os.path.join(os.environ['HOME'], base_dir)

    clf_data_dict = np.load(os.path.join(load_dir, file_name), allow_pickle=True).item()

    user_ids_list = []
    labels_list = []
    input_ids_list = []
    attention_masks_list = []

    for user_id, data_dict in clf_data_dict.items():
        for ii in range(len(data_dict['input_ids'])):
            user_ids_list.append(user_id)
            labels_list.append(data_dict['label'])
            input_ids_list.append(data_dict['input_ids'][[ii]])
            attention_masks_list.append(data_dict['attention_masks'][[ii]])

    user_ids = np.array(user_ids_list)
    labels = np.array(labels_list)
    input_ids = np.concatenate(input_ids_list)
    attention_masks = np.concatenate(attention_masks_list)

    return input_ids, attention_masks, labels, user_ids


def calculate_metrics(user_true_pred_lbls, labels):
    num_post_correct = 0
    num_user_correct = 0

    post_gold = []
    post_pred = []

    user_gold = []
    user_pred = []

    for uid, list_of_label_tuples in user_true_pred_lbls.items():
        for (true, pred) in list_of_label_tuples:
            post_gold.append(true)
            post_pred.append(pred)

            if true == pred:
                num_post_correct += 1

        true_list, pred_list = zip(*list_of_label_tuples)

        user_gold.append(true_list[0])

        if np.mean(pred_list) > 0.5:
            declare_suicidal = True
            user_pred.append(1)
        else:
            declare_suicidal = False
            user_pred.append(0)

        if (true_list[0] == 1 and declare_suicidal) or (true_list[0] == 0 and not declare_suicidal):
            num_user_correct += 1

    post_accuracy = num_post_correct / len(labels)
    user_accuracy = num_user_correct / len(user_true_pred_lbls)

    print("Post classification accuracy:\t{:.2f} {}".format(post_accuracy * 100, '%'))
    print("User classification accuracy:\t{:.2f} {}\n".format(user_accuracy * 100, '%'))

    post_p, post_r, post_f, _ = precision_recall_fscore_support(post_gold, post_pred, average='weighted')
    user_p, user_r, user_f, _ = precision_recall_fscore_support(user_gold, user_pred, average='weighted')

    msg = "Post classification Precision, Recall, and F-score:\t{:.2f} {} \t {:.2f} {} \t {:.2f} {}"
    print(msg.format(post_p * 100, '%', post_r * 100, '%', post_f * 100, '%'))
    msg = "User classification Precision, Recall, and F-score:\t{:.2f} {} \t {:.2f} {} \t {:.2f} {}"
    print(msg.format(user_p * 100, '%', user_r * 100, '%', user_f * 100, '%'))
