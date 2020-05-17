import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from transformers import BertModel, BertTokenizer


class Bert4Clf(nn.Module):
    def __init__(self, use_cuda=False, joint_training=False):
        super(Bert4Clf, self).__init__()

        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

        _model_version = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(_model_version, do_lower_case=True)
        self.bert = BertModel.from_pretrained(_model_version, output_attentions=True).to(self.device)

        self.num_classes = 2

        self.post_dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        self.post_classifier = nn.Linear(
            self.bert.config.hidden_size,
            self.num_classes, bias=True).to(self.device)

        self.joint_training = joint_training

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = None
        self.setup_optimizer()

    def forward(self, input_ids, attention_masks, labels=None):
        pooled_outputs = self.bert(input_ids, attention_mask=attention_masks)[1]
        pooled_outputs = self.post_dropout(pooled_outputs)

        post_logits = self.post_classifier(pooled_outputs)
        outputs = (post_logits,)

        if labels is not None:
            post_loss = self.loss_fn(post_logits.view(-1, self.num_classes), labels.view(-1))
            outputs += (post_loss,)

        return outputs

    def train_model(self, input_ids, attention_masks, labels, batch_size=4, update_feq=32, nb_epochs=5):

        num_samples = len(input_ids)
        num_training_steps_per_epoch = int(np.ceil(num_samples / batch_size))

        post_loss_history = []

        self.train()

        for i in range(nb_epochs):
            self.optimizer.zero_grad()
            shuffle_indices = torch.randperm(num_samples)

            for j in range(num_training_steps_per_epoch):
                batch_indices = slice(j * batch_size, (j + 1) * batch_size)

                batch_inputs = torch.tensor(
                    input_ids[shuffle_indices][batch_indices],
                    dtype=torch.long,
                    device=self.device)
                batch_attention_masks = torch.tensor(
                    attention_masks[shuffle_indices][batch_indices],
                    dtype=torch.long,
                    device=self.device)
                batch_labels = torch.tensor(
                    labels[shuffle_indices][batch_indices],
                    dtype=torch.long,
                    device=self.device)

                post_loss = self(batch_inputs, batch_attention_masks, batch_labels)[0]
                post_loss /= update_feq

                post_loss.backward()
                post_loss_history.append(post_loss.item())

                if (j + 1) % update_feq == 0:
                    print(i, j, post_loss.item())
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        return post_loss_history

    def validate_model(self, input_ids, attention_masks, labels, user_ids, batch_size=8):
        num_incorrect = 0
        # y_pred_all = []
        user_true_pred_lbls = {uid: [] for uid in np.unique(user_ids)}

        self.eval()

        num_samples = len(input_ids)
        for i in tqdm(range(int(np.ceil(num_samples / batch_size)))):
            indices = slice(i * batch_size, (i + 1) * batch_size)

            batch_input_ids = torch.tensor(
                input_ids[indices],
                dtype=torch.long,
                device=self.device
            )
            batch_attention_masks = torch.tensor(
                attention_masks[indices],
                dtype=torch.long,
                device=self.device
            )
            batch_labels = labels[indices]
            batch_user_ids = user_ids[indices]

            with torch.no_grad():
                logits = self(batch_input_ids, batch_attention_masks)[0]

            y_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).tolist()

            num_incorrect += sum([abs(tup[0] - tup[1]) for tup in zip(batch_labels.flatten(), y_pred)])
            # y_pred_all.extend(y_pred)

            for j, user in enumerate(batch_user_ids):
                user_true_pred_lbls[user].append((batch_labels[j], y_pred[j]))

        print("Accuracy: {:.2f} {:s}".format((num_samples - num_incorrect) / num_samples * 100, '%'))

        return user_true_pred_lbls

    def setup_optimizer(self):
        large_lr_parameters_keywords = [
            'post_classifier', 'user_hidden_proj', 'layer_norm', 'user_classifier']

        param_small_lr_list = []  # should be changed into torch.nn.ParameterList()
        param_large_lr_list = []  # should be changed into torch.nn.ParameterList()

        for k, v in self.named_parameters():
            small_lr = True
            for keyword in large_lr_parameters_keywords:
                if keyword in k:
                    param_large_lr_list.append(v)
                    small_lr = False
                    break
            if small_lr:
                param_small_lr_list.append(v)

        param_small_lr_list = torch.nn.ParameterList(param_small_lr_list)
        param_large_lr_list = torch.nn.ParameterList(param_large_lr_list)

        if self.joint_training:
            self.bert.train()
            self.optimizer = torch.optim.Adam([
                {'params': param_small_lr_list, 'lr': 1e-5},
                {'params': param_large_lr_list, 'lr': 1e-3}],
                weight_decay=0.01,
            )
        else:
            self.bert.eval()
            self.optimizer = torch.optim.Adam([
                {'params': param_small_lr_list, 'lr': 0.0},
                {'params': param_large_lr_list, 'lr': 1e-3}],
                weight_decay=0.01,
            )

        print('Total number of classifier parameters: {}'.format(sum(p.numel() for p in param_large_lr_list if p.requires_grad)))
        print('Total number of BERT parameters: {}'.format(sum(p.numel() for p in param_small_lr_list if p.requires_grad)))
