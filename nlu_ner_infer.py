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
import re
from preprocessing import lemmatizer
from nlu_ner_train import JointModel, JointModel_M


def load_dict(path):
    label2id = {}
    with open(path, "r", encoding="utf-8") as f:
        labels = f.readlines()
        for idx, label in enumerate(labels):
            label = label.strip()
            label2id[label] = idx
    id2label = dict([(idx, label) for label, idx in label2id.items()])
    return label2id, id2label


def split_text(text):
    words = []

    text = text.strip()  # remove new line character \n
    text = re.sub(r'[^\w\s+]', '', text)  # remove punctuation
    splitted_text = ', '.join(jieba.cut(text)).split(', ')  # get segmented text (list)
    # print("cut by jieba: ", splitted_text)
    for t in splitted_text:  # iterate each segmented result
        if t != ' ':  # filter all the white spaces
            has_chinese = re.findall(r'[\u4e00-\u9fff]+', t)  # find if a segmented word is chinese
            if len(t) > 1 and has_chinese:
                for i in range(len(t)):
                    words.append(t[i])
            else:
                # try to do lemmantization for english words
                try:
                    t = lemmatizer.lemmatize(t)
                except:
                    pass
                words.append(t)

    result = ' '.join(words)

    return result


def nlu_predict(
        input_text, 
        joint_model, 
        tokenizer, 
        id2intent, 
        id2slot, 
        converter=None, 
        split_text=split_text, 
        max_seq_len=512
    ):
    
    joint_model.eval()
    
    input_text = split_text(input_text)
    
    if converter:
        input_text = converter.convert(input_text).lower()
    
    splited_input_text = input_text.split()
    print("splited input: ")
    print(splited_input_text)
    
    texts = []
    is_splited_word_piece = []
    for index, word in enumerate(splited_input_text):
        word_piece = tokenizer.tokenize(word)
        # print(word_piece)
        if len(word_piece) > 1:
            is_splited_word_piece.extend([0] + [1]*(len(word_piece)-1))
        else:
            is_splited_word_piece.append(0)
        texts.extend(word_piece)
    print("splited word piece: ")
    print(is_splited_word_piece)
    
    input_ids = [1] + tokenizer.convert_tokens_to_ids(texts) + [2]
    token_type_ids = [0] * len(input_ids)
    seq_len = len(input_ids)
    input_ids = paddle.to_tensor(input_ids).unsqueeze(0)
    token_type_ids = paddle.to_tensor(token_type_ids).unsqueeze(0)

    
    """
    above is new, below is original
    """

    # features = tokenizer(splited_input_text, 
    #                      is_split_into_words=True, 
    #                      max_seq_len=max_seq_len, 
    #                      return_length=True) 
    # input_ids = paddle.to_tensor(features["input_ids"]).unsqueeze(0)
    # token_type_ids = paddle.to_tensor(features["token_type_ids"]).unsqueeze(0)
    # seq_len = features["seq_len"]
    
    intent_logits, slot_logits = joint_model(input_ids, token_type_ids=token_type_ids)
    slot_logits = joint_model(input_ids, token_type_ids=token_type_ids)
    
    # intent_softmax = nn.Softmax()
    # intent_probs = intent_softmax(intent_logits).numpy()[0]
    # # parse intent labels and probs
    # intent_labels_probs = [(id2intent[idx], intent_probs[idx]) for idx, v in enumerate(intent_logits.numpy()[0]) if v > 0]
    
    # parse slot labels
    slot_softmax = nn.Softmax()
    slot_probs = slot_softmax(slot_logits).numpy()[0][1:(seq_len)-1]
    # print(slot_probs)
    slot_pred_labels = slot_logits.argmax(axis=-1).numpy()[0][1:(seq_len)-1]
    # print(slot_pred_labels)
    
    slot_labels = []
    for idx in slot_pred_labels:
        slot_label = id2slot[idx]
        if slot_label != "O":
            slot_label = list(id2slot[idx])
            slot_label[1] = "-"
            slot_label = "".join(slot_label)
        slot_labels.append(slot_label)
    slot_labels = [j for i, j in zip(is_splited_word_piece, slot_labels) if i==0]
    print("slot labels: ")
    print(slot_labels)

    slot_entities = get_entities(slot_labels)
    print("slot entities: ")
    print(slot_entities)
    
    slots = []
    for slot_entity in slot_entities:
        temp_dict = {}
        entity_type, start, end = slot_entity
        temp_dict["type"] = entity_type
        temp_dict["start"] = start
        temp_dict["end"] = end
        # print(f"{entity_name}: ", "".join(splited_input_text[start:end+1]))
        entity_name = splited_input_text[start:end+1]
        is_chinese = False
        for e in entity_name:
            if re.findall(r'[\u4e00-\u9fff]+', e):
                is_chinese = True
                break
        if is_chinese:
            temp_dict["text"] = "".join(splited_input_text[start:end+1])
        else:
            temp_dict["text"] = " ".join(splited_input_text[start:end+1])
        
        slot_prob = slot_probs[start:end+1]
        slot_pred_label = slot_pred_labels[start:end+1]
        confidence_score = 1.0
        for prob, label in zip(slot_prob, slot_pred_label):
            confidence_score *= prob[label]
        temp_dict["con"] = confidence_score             
 
        slots.append(temp_dict)       
 
    # return intent_labels_probs, slots
    return slots


def nlu_predict_m(
        input_text, 
        joint_model, 
        tokenizer, 
        id2intent, 
        id2slot, 
        converter=None, 
        split_text=split_text, 
        max_seq_len=512
    ):
    
    joint_model.eval()

    input_text = split_text(input_text)
    
    if converter:
        input_text = converter.convert(input_text).lower()
    
    splited_input_text = input_text.split()
    print("splited input: ")
    print(splited_input_text)
    
    texts = []
    is_splited_word_piece = []
    for index, word in enumerate(splited_input_text):
        word_piece = tokenizer.tokenize(word)
        # print(word_piece)
        if len(word_piece) > 1:
            is_splited_word_piece.extend([0] + [1]*(len(word_piece)-1))
        else:
            is_splited_word_piece.append(0)
        texts.extend(word_piece)
    print("splited word piece: ")
    print(is_splited_word_piece)
    
    input_ids = [1] + tokenizer.convert_tokens_to_ids(texts) + [2]
    seq_len = len(input_ids)
    input_ids = paddle.to_tensor(input_ids).unsqueeze(0)

    # intent_logits, slot_logits = joint_model(input_ids)
    slot_logits = joint_model(input_ids)
    
    # intent_softmax = nn.Softmax()
    # intent_probs = intent_softmax(intent_logits).numpy()[0]
    # parse intent labels and probs
    # intent_labels_probs = [{"topic": id2intent[idx], "con": float(intent_probs[idx])} for idx, v in enumerate(intent_logits.numpy()[0]) if v > 0]
    
    # parse slot labels
    slot_softmax = nn.Softmax()
    slot_probs = slot_softmax(slot_logits).numpy()[0][1:(seq_len)-1]
    # print(slot_probs)
    slot_pred_labels = slot_logits.argmax(axis=-1).numpy()[0][1:(seq_len)-1]
    # print(slot_pred_labels)
    
    slot_labels = []
    for idx in slot_pred_labels:
        slot_label = id2slot[idx]
        if slot_label != "O":
            slot_label = list(id2slot[idx])
            slot_label[1] = "-"
            slot_label = "".join(slot_label)
        slot_labels.append(slot_label)
    slot_labels = [j for i, j in zip(is_splited_word_piece, slot_labels) if i==0]
    print("slot labels: ")
    print(slot_labels)

    slot_entities = get_entities(slot_labels)
    print("slot entities: ")
    print(slot_entities)
    
    slots = []
    for slot_entity in slot_entities:
        temp_dict = {}
        entity_type, start, end = slot_entity
        temp_dict["type"] = entity_type
        temp_dict["start"] = start
        temp_dict["end"] = end
        # print(f"{entity_name}: ", "".join(splited_input_text[start:end+1]))
        entity_name = splited_input_text[start:end+1]
        is_chinese = False
        for e in entity_name:
            if re.findall(r'[\u4e00-\u9fff]+', e):
                is_chinese = True
                break
        if is_chinese:
            temp_dict["text"] = "".join(splited_input_text[start:end+1])
        else:
            temp_dict["text"] = " ".join(splited_input_text[start:end+1])
        
        slot_prob = slot_probs[start:end+1]
        slot_pred_label = slot_pred_labels[start:end+1]
        confidence_score = 1.0
        for prob, label in zip(slot_prob, slot_pred_label):
            confidence_score *= prob[label]
        temp_dict["con"] = confidence_score             
 
        slots.append(temp_dict)       
 
    # return intent_labels_probs, slots
    return slots


if __name__=="__main__":
    print(split_text("mos cafe"))