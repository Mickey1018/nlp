import paddle
import os
import ast
import argparse
import warnings
import numpy as np
from functools import partial
from seqeval.metrics.sequence_labeling import get_entities
import json
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieTokenizer, ErnieModel, ErnieMTokenizer, ErnieMModel, LinearDecayWithWarmup
from paddlenlp.data import Stack, Pad, Tuple
from collections import Counter
import random
import jieba
import opencc
from configuration import Config


def load_dict(path):
    label2id = {}
    with open(path, "r", encoding="utf-8") as f:
        labels = f.readlines()
        for idx, label in enumerate(labels):
            label = label.strip()
            label2id[label] = idx
    id2label = dict([(idx, label) for label, idx in label2id.items()])
    return label2id, id2label


def read(data_path, lang=None):
    if not lang:
        label_path = 'label'
        seq_in_path = 'seq.in'
        seq_out_path = 'seq.out'
    else:
        lang = lang.lower()
        label_path = 'label_{}'.format(lang)
        seq_in_path = 'seq_{}.in'.format(lang)
        seq_out_path = 'seq_{}.out'.format(lang)
        
    with open(os.path.join(data_path, seq_in_path), "r", encoding="utf-8") as f_input, \
         open(os.path.join(data_path, label_path), "r", encoding="utf-8") as f_intent_label, \
         open(os.path.join(data_path, seq_out_path), "r", encoding="utf-8") as f_slot_label:
        inputs, intent_labels, slot_labels = f_input.readlines(), f_intent_label.readlines(), f_slot_label.readlines()
        converter = opencc.OpenCC('t2s.json')
        for input, intent_label, slot_label in zip(inputs, intent_labels, slot_labels):
            yield {
                "words": converter.convert(input).strip().split(), 
                "slot_labels": slot_label.strip().split(), 
                "intent_labels":[intent_label.strip()]
                }


def convert_example_to_feature(example, tokenizer, slot2id, intent2id, pad_default_tag=0, max_seq_len=512):
    new_words = []
    new_slot_labels = []
    for word, label in zip(example["words"], example["slot_labels"]):
        word_piece = tokenizer.tokenize(word)
        new_words.extend(word_piece)
        new_slot_labels.extend([label]*len(word_piece))
    # print("#"*100)
    # print(new_words)
    # print(new_slot_labels)
    # print(len(new_words), len(new_slot_labels))
    # print(len(new_words)==len(new_slot_labels))

    example["words"] = new_words
    example["slot_labels"] = new_slot_labels

    # features = tokenizer(example["words"], is_split_into_words=True, max_seq_len=max_seq_len)
    # print(features)
    # print(len(features["input_ids"]))

    slot_ids = [slot2id[slot] for slot in example["slot_labels"][:(max_seq_len-2)]]
    slot_ids = [slot2id[pad_default_tag]] + slot_ids + [slot2id[pad_default_tag]]
    input_ids = [1] + tokenizer.convert_tokens_to_ids(new_words) + [2]
    input_ids = input_ids[:max_seq_len]
    token_type_ids = [0] * len(input_ids)
    # print(input_ids)
    # print(len(input_ids))
    # assert len(features["input_ids"]) == len(slot_ids)
    print(example)
    assert len(input_ids) == len(slot_ids)

    # get intent feature
    intent_labels = [0] * len(intent2id)
    for intent in example["intent_labels"]:
        intent_labels[intent2id[intent]] = 1   
    
    # return features["input_ids"], features["token_type_ids"], intent_labels, slot_ids
    return input_ids, token_type_ids, intent_labels, slot_ids



class JointModel(paddle.nn.Layer):
    def __init__(self, ernie, num_slots, num_intents, dropout=None):  # use_history=False
        super(JointModel, self).__init__()
        self.num_slots = num_slots
        self.num_intents = num_intents

        self.ernie = ernie
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ernie.config["hidden_dropout_prob"])

        self.intent_hidden = nn.Linear(self.ernie.config["hidden_size"], self.ernie.config["hidden_size"])
        self.slot_hidden = nn.Linear(self.ernie.config["hidden_size"], self.ernie.config["hidden_size"])

        self.intent_classifier = nn.Linear(self.ernie.config["hidden_size"], self.num_intents)
        self.slot_classifier = nn.Linear(self.ernie.config["hidden_size"], self.num_slots)


    def forward(self, token_ids, token_type_ids=None, position_ids=None, attention_mask=None):  # history_ids=None
        sequence_output, pooled_output = self.ernie(token_ids, 
                                                    token_type_ids=token_type_ids, 
                                                    position_ids=position_ids, 
                                                    attention_mask=attention_mask)

        sequence_output = F.relu(self.slot_hidden(self.dropout(sequence_output)))  # dropout --> linear --> relu
        pooled_output = F.relu(self.intent_hidden(self.dropout(pooled_output)))

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits


class JointModel_M(paddle.nn.Layer):
    def __init__(self, ernie, num_slots, num_intents, dropout=None):  # use_history=False
        super(JointModel_M, self).__init__()
        self.num_slots = num_slots
        self.num_intents = num_intents

        self.ernie = ernie
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ernie.config["hidden_dropout_prob"])

        self.intent_hidden = nn.Linear(self.ernie.config["hidden_size"], self.ernie.config["hidden_size"])
        self.slot_hidden = nn.Linear(self.ernie.config["hidden_size"], self.ernie.config["hidden_size"])

        self.intent_classifier = nn.Linear(self.ernie.config["hidden_size"], self.num_intents)
        self.slot_classifier = nn.Linear(self.ernie.config["hidden_size"], self.num_slots)


    def forward(self, token_ids, token_type_ids=None, position_ids=None, attention_mask=None):  # history_ids=None
        sequence_output, pooled_output = self.ernie(token_ids,
                                                    position_ids=position_ids, 
                                                    attention_mask=attention_mask)

        sequence_output = F.relu(self.slot_hidden(self.dropout(sequence_output)))  # dropout --> linear --> relu
        pooled_output = F.relu(self.intent_hidden(self.dropout(pooled_output)))

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits


class JointLoss(paddle.nn.Layer):
    def __init__(self, intent_weight=None):
        super(JointLoss, self).__init__()
        self.intent_criterion = paddle.nn.BCEWithLogitsLoss(weight=intent_weight)
        self.slot_criterion = paddle.nn.CrossEntropyLoss()

    def forward(self, intent_logits, slot_logits, intent_labels, slot_labels):
        intent_loss = self.intent_criterion(intent_logits, intent_labels)
        slot_loss = self.slot_criterion(slot_logits, slot_labels)
        loss = intent_loss + slot_loss

        return loss


class SeqEntityScore(object):
    def __init__(self, id2tag):
        self.id2tag = id2tag
        self.real_entities = []
        self.pred_entities = []
        self.correct_entities = []
        
    def reset(self):
        self.real_entities.clear()
        self.pred_entities.clear()
        self.correct_entities.clear()

    def compute(self, real_count, pred_count, correct_count):
        recall = 0 if real_count == 0 else (correct_count / real_count)
        precision = 0 if pred_count == 0 else (correct_count / pred_count)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def get_result(self):
        result = {}
        real_counter = Counter([item[0] for item in self.real_entities])  # item = ['slot', idx_start, idx_end]
        pred_counter = Counter([item[0] for item in self.pred_entities])
        correct_counter = Counter([item[0] for item in self.correct_entities])  # {'slot_1': 3, 'slot_2':5}
        for label, count in real_counter.items():
            real_count = count
            pred_count = pred_counter.get(label, 0)
            correct_count = correct_counter.get(label, 0)
            recall, precision, f1 = self.compute(real_count, pred_count, correct_count)
            result[label] = {"Precision": round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}
        real_total_count = len(self.real_entities)
        pred_total_count = len(self.pred_entities)
        correct_total_count = len(self.correct_entities)
        recall, precision, f1 = self.compute(real_total_count, pred_total_count, correct_total_count)
        result["Total"] = {"Precision": round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}

        return result

    def get_entities_bio(self, seq):
        entities = []
        entity = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = self.id2tag[tag]
                
            if tag.startswith("B-"):
                if entity[2] != -1:
                    entities.append(entity)
                entity = [-1, -1, -1]
                entity[1] = indx
                entity[0] = tag.split('-', maxsplit=1)[1]
                entity[2] = indx
                # entity = ['slot', idx_1, idx_1]
                if indx == len(seq) - 1:  # at the end
                    entities.append(entity)
            elif tag.startswith('I-') and entity[1] != -1:
                _type = tag.split('-', maxsplit=1)[1]  # get slot name
                if _type == entity[0]:
                    entity[2] = indx
                    # entity = ['slot', idx_1, idx_2]
                if indx == len(seq) - 1:  # at the end
                    entities.append(entity)
            else:
                if entity[2] != -1:
                    entities.append(entity)
                entity = [-1, -1, -1]  # reset
        return entities  # [['slot_1', idx_start, idx_end], ['slot_2', idx_start, idx_end]]

    def update(self, real_paths, pred_paths):
        
        if isinstance(real_paths, paddle.Tensor):
            real_paths = real_paths.numpy()
        if isinstance(pred_paths, paddle.Tensor):
            pred_paths = pred_paths.numpy()

        for real_path, pred_path in zip(real_paths, pred_paths):
            real_ents = self.get_entities_bio(real_path)  # [['slot_1', idx_start, idx_end], ['slot_2', idx_start, idx_end]]
            pred_ents = self.get_entities_bio(pred_path)  # [['slot_1', idx_start, idx_end], ['slot_2', idx_start, idx_end]]
            self.real_entities.extend(real_ents)
            self.pred_entities.extend(pred_ents)
            self.correct_entities.extend([pred_ent for pred_ent in pred_ents if pred_ent in real_ents])

    def format_print(self, result, print_detail=False):
        def print_item(entity, metric):
            if entity != "Total":
                print(f"Entity: {entity} - Precision: {metric['Precision']} - Recall: {metric['Recall']} - F1: {metric['F1']}")
            else:
                print(f"Total: Precision: {metric['Precision']} - Recall: {metric['Recall']} - F1: {metric['F1']}")

        print_item("Total", result["Total"])
        if print_detail:
            for key in result.keys():
                if key == "Total":
                    continue
                print_item(key, result[key])
            print("\n")


class MultiLabelClassificationScore(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.all_pred_labels = []
        self.all_real_labels = []
        self.all_correct_labels = []        
    
    def reset(self):
        self.all_pred_labels.clear()
        self.all_real_labels.clear()
        self.all_correct_labels.clear()
     
    def update(self, pred_labels, real_labels):
        if not isinstance(pred_labels, list):
            pred_labels = pred_labels.numpy().tolist()
        if not isinstance(real_labels, list):
            real_labels = real_labels.numpy().tolist()

        for i in range(len(real_labels)):  # iterate along batch size
            for j in range(len(real_labels[0])):  # iterate along number of intent
                if real_labels[i][j] == 1 and  pred_labels[i][j] > 0:
                    self.all_correct_labels.append(self.id2label[j])
                if real_labels[i][j] == 1:
                    self.all_real_labels.append(self.id2label[j])
                if pred_labels[i][j] > 0:
                    self.all_pred_labels.append(self.id2label[j])

    def compute(self, pred_count , real_count, correct_count):
        recall  = 0. if real_count == 0 else (correct_count / real_count)
        precision = 0. if pred_count == 0 else (correct_count / pred_count)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def get_result(self):
        result = {}
        pred_counter = Counter(self.all_pred_labels)
        real_counter = Counter(self.all_real_labels)
        correct_counter = Counter(self.all_correct_labels)
        for label, count in real_counter.items():
            real_count = count
            pred_count = pred_counter[label]
            correct_count = correct_counter[label]
            precision, recall, f1 = self.compute(pred_count, real_count, correct_count)
            result[label] = {"Precision": round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}
        real_total_count = len(self.all_real_labels)
        pred_total_count = len(self.all_pred_labels)
        correct_total_count = len(self.all_correct_labels)
        recall, precision, f1 = self.compute(real_total_count, pred_total_count, correct_total_count)
        result["Total"] = {"Precision": round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}

        return result         

    def format_print(self, result, print_detail=False):
        def print_item(entity, metric):
            if entity != "Total":
                print(f"Entity: {entity} - Precision: {metric['Precision']} - Recall: {metric['Recall']} - F1: {metric['F1']}")
            else:
                print(f"Total: Precision: {metric['Precision']} - Recall: {metric['Recall']} - F1: {metric['F1']}")

        print_item("Total", result["Total"])
        if print_detail:
            for key in result.keys():
                if key == "Total":
                    continue
                print_item(key, result[key])
            print("\n")


def nlu_model_train(logger, project_path, lang=None, num_epoch=100, batch_size=16, max_seq_len=512, learning_rate=3e-5, weight_decay=0.01, warmup_proportion=0.1, max_grad_norm=1.0, seed=666, log_step=50, eval_step=100, use_gpu=True):
    """
    intent_path: ./data/intent_label_mall.txt
    slot_path: ./data/slot_label_mall.txt
    train_path: ./data/train_oc
    dev_path: ./data/dev_oc
    save_path: 
    lang: ["zh", "en"]
    """
    intent_path = os.path.join(project_path, "intent", "intent_label.txt")
    slot_path = os.path.join(project_path, "keyword", "slot_label.txt")
    train_path = os.path.join(project_path, "train")
    dev_path = os.path.join(project_path, "dev")
    save_path = os.path.join(project_path, "ckpt", "best.pdparams")
    # if not os.path.exist(save_path):
    #     os.makedirs(save_path)

    # envir setting
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if use_gpu:
        paddle.set_device("gpu:0")
    else:
        paddle.set_device("cpu")
    
    if not lang:
        model_name = 'ernie-m-base'
    elif lang.lower() == 'en':
        model_name = 'ernie-2.0-base-en'
    else:
        model_name = "ernie-3.0-xbase-zh"

    logger.info('model name: ')
    logger.info(model_name)

    intent2id, id2intent = load_dict(intent_path)
    slot2id, id2slot = load_dict(slot_path)

    logger.info("loading dataset...")
    train_ds = load_dataset(read, data_path=train_path, lang=lang, lazy=False)
    dev_ds = load_dataset(read, data_path=dev_path, lang=lang, lazy=False)
    logger.info("dataset loaded successfully!")

    # convert data into feature form
    tokenizer = ErnieMTokenizer.from_pretrained(model_name)

    trans_func = partial(convert_example_to_feature,
                         tokenizer=tokenizer,
                         slot2id=slot2id,
                         intent2id=intent2id,
                         pad_default_tag="O",
                         max_seq_len=max_seq_len)

    logger.info("converting dataset into features...")
    train_ds = train_ds.map(trans_func, lazy=False)
    dev_ds = dev_ds.map(trans_func, lazy=False)
    logger.info("dataset are converted to features successfully!")


    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        Stack(dtype="float32"),
        Pad(axis=0, pad_val=slot2id["O"], dtype="int64"),
        ):fn(samples)

    train_batch_sampler = paddle.io.BatchSampler(train_ds, batch_size=batch_size, shuffle=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=batch_size, shuffle=False)

    logger.info("creating train and dev data loader...")
    train_loader = paddle.io.DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=batchify_fn, return_list=True)
    dev_loader = paddle.io.DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=batchify_fn, return_list=True)
    logger.info("train and dev data loader created successfully!")

    if not lang:
        ernie = ErnieMModel.from_pretrained(model_name)
        joint_model = JointModel_M(ernie, len(slot2id), len(intent2id), dropout=0.1)
    elif lang.lower() in ['zh', 'en']:
        ernie = ErnieModel.from_pretrained(model_name)
        joint_model = JointModel(ernie, len(slot2id), len(intent2id), dropout=0.1)

    num_training_steps = len(train_loader) * num_epoch
    lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
    decay_params = [p.name for n, p in joint_model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    grad_clip = paddle.nn.ClipGradByGlobalNorm(max_grad_norm)
    optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler, 
                                       parameters=joint_model.parameters(), 
                                       weight_decay=weight_decay, 
                                       apply_decay_param_fun=lambda x: x in decay_params, 
                                       grad_clip=grad_clip)
    
    joint_loss = JointLoss()

    intent_metric = MultiLabelClassificationScore(id2intent)
    slot_metric = SeqEntityScore(id2slot)

    global_step, intent_best_f1, slot_best_f1 = 0, 0., 0.
    
    joint_model.train()
    
    logger.info("Start training!")
    for epoch in range(1, num_epoch+1):
        for batch_data in train_loader:
            input_ids, token_type_ids, intent_labels, tag_ids = batch_data

            if not lang:
                intent_logits, slot_logits = joint_model(input_ids)
            elif lang.lower() in ['zh', 'en']:
                intent_logits, slot_logits = joint_model(input_ids, token_type_ids=token_type_ids)

            loss = joint_loss(intent_logits, slot_logits, intent_labels, tag_ids)

            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

            if global_step > 0 and global_step % log_step == 0:
                logger.info(f"epoch: {epoch} - global_step: {global_step}/{num_training_steps} - loss:{loss.numpy().item():.6f}")
            if global_step > 0 and global_step % eval_step == 0:
                intent_results, slot_results = evaluate(joint_model, dev_loader, intent_metric, slot_metric)
                intent_result, slot_result = intent_results["Total"], slot_results["Total"]
                joint_model.train()
                intent_f1, slot_f1 = intent_result["F1"], slot_result["F1"]
                # if intent_f1 > intent_best_f1 or slot_f1 > slot_best_f1:
                paddle.save(joint_model.state_dict(), save_path)
                if intent_f1 > intent_best_f1:
                    logger.info(f"intent best F1 performence has been updated: {intent_best_f1:.5f} --> {intent_f1:.5f}")
                    intent_best_f1 = intent_f1
                if slot_f1 > slot_best_f1:
                    logger.info(f"slot best F1 performence has been updated: {slot_best_f1:.5f} --> {slot_f1:.5f}")
                    slot_best_f1 = slot_f1
                logger.info(f'intent evalution result: precision: {intent_result["Precision"]:.5f}, recall: {intent_result["Recall"]:.5f},  F1: {intent_result["F1"]:.5f}, current best {intent_best_f1:.5f}')
                logger.info(f'slot evalution result: precision: {slot_result["Precision"]:.5f}, recall: {slot_result["Recall"]:.5f},  F1: {slot_result["F1"]:.5f}, current best {slot_best_f1:.5f}\n')

            global_step += 1
    logger.info("Finish training!")


def evaluate(joint_model, data_loader, intent_metric, slot_metric):

    joint_model.eval()
    intent_metric.reset()
    slot_metric.reset()
    for idx, batch_data in enumerate(data_loader):
        input_ids, token_type_ids, intent_labels, tag_ids = batch_data
        intent_logits, slot_logits = joint_model(input_ids, token_type_ids=token_type_ids)
        # count intent metric
        intent_metric.update(pred_labels=intent_logits, real_labels=intent_labels)
        # count slot metric
        slot_pred_labels = slot_logits.argmax(axis=-1)
        slot_metric.update(pred_paths=slot_pred_labels, real_paths=tag_ids)

    intent_results = intent_metric.get_result()
    slot_results = slot_metric.get_result()

    return intent_results, slot_results


if __name__=="__main__":

    lang = None
    batch_size = 16
    max_seq_len = 512
    num_epoch = 100
    learning_rate = 3e-5
    weight_decay = 0.01
    warmup_proportion = 0.1
    max_grad_norm = 1.0
    log_step = 50
    eval_step = 100
    seed = 666

    intent_path = Config.nlu_model_intent_label_path
    slot_path = Config.nlu_model_slot_label_path
    train_path = Config.nlu_model_train_dir
    dev_path = Config.nlu_model_dev_dir

    if not lang:
        model_name = "ernie-m-base"
        save_path = Config.nlu_model_ckpt_path_multi
    elif lang.startswith("zh"):
        model_name = "ernie-3.0-xbase-zh"
        save_path = Config.nlu_model_ckpt_path_zh
    elif lang.startswith("en"):
        model_name = "ernie-2.0-base-en"
        save_path = Config.nlu_model_ckpt_path_en

    # load and process data
    intent2id, id2intent = load_dict(intent_path)
    slot2id, id2slot = load_dict(slot_path)

    train_ds = load_dataset(read, data_path=train_path, lang=lang, lazy=False)
    dev_ds = load_dataset(read, data_path=dev_path, lang=lang, lazy=False)

    # convert data into feature form
    tokenizer = ErnieTokenizer.from_pretrained(model_name)
    trans_func = partial(convert_example_to_feature,
                         tokenizer=tokenizer,
                         slot2id=slot2id,
                         intent2id=intent2id,
                         pad_default_tag="O",
                         max_seq_len=max_seq_len)

    train_ds = train_ds.map(trans_func, lazy=False)
    dev_ds = dev_ds.map(trans_func, lazy=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        Stack(dtype="float32"),
        Pad(axis=0, pad_val=slot2id["O"], dtype="int64"),
        ):fn(samples)

    train_batch_sampler = paddle.io.BatchSampler(train_ds, batch_size=batch_size, shuffle=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=batch_size, shuffle=False)
    train_loader = paddle.io.DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=batchify_fn, return_list=True)
    dev_loader = paddle.io.DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=batchify_fn, return_list=True)

    # envir setting
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if paddle.get_device().startswith("gpu"):
        paddle.set_device("gpu:0")

    if not lang:
        ernie = ErnieMModel.from_pretrained(model_name)
        joint_model = JointModel_M(ernie, len(slot2id), len(intent2id), dropout=0.1)

    num_training_steps = len(train_loader) * num_epoch
    lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
    decay_params = [p.name for n, p in joint_model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    grad_clip = paddle.nn.ClipGradByGlobalNorm(max_grad_norm)
    optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler, 
                                       parameters=joint_model.parameters(), 
                                       weight_decay=weight_decay, 
                                       apply_decay_param_fun=lambda x: x in decay_params, 
                                       grad_clip=grad_clip)

    joint_loss = JointLoss()

    intent_metric = MultiLabelClassificationScore(id2intent)
    slot_metric = SeqEntityScore(id2slot)

    nlu_model_train(intent_path, slot_path, train_path, dev_path, save_path, lang)




