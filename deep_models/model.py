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
        user_logits = self.user_classifier(pooled_outputs)

        outputs = (post_logits, user_logits)

        if labels is not None:
            post_loss = self.loss_fn(post_logits.view(-1, self.num_classes), labels.view(-1))
            outputs += (post_loss,)

        return outputs

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
