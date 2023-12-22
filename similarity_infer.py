import pandas as pd
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
import opencc
from configuration import Config
from segmentation import extract_key_word
import string
from nltk.stem import PorterStemmer
from cantonese_2_chinese import change_cantonese_to_chinese


# load stopwords
stopwords_zh = set([i for i in string.punctuation])
with open(Config.stopwords_zh_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line not in stopwords_zh:
            stopwords_zh.add(line)

stopwords_en = set([i for i in string.punctuation])
with open(Config.stopwords_en_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line not in stopwords_en:
            stopwords_en.add(line)


converter = opencc.OpenCC('t2s.json')
porter = PorterStemmer()


def get_answer_from_faq(lang, question, similarity_model, intent, faq_db):

    if lang == 'zh-hk':
        question_column = 'Question'
        answer_column = 'FAQ_Suggested_Answer'
    elif lang == 'zh-cn':
        question_column = 'Question_Sim'
        answer_column = 'FAQ_Suggested_Answer_Sim'
    elif lang == 'en':
        question_column = 'Question_Eng'
        answer_column = 'FAQ_Suggested_Answer_Eng'
    
    converter = opencc.OpenCC('t2s.json')

    df = faq_db

    # if no intent, return don't understand
    if intent == "nothing" or not intent:
        df_target = df[df.Topic=="nothing"].reset_index(drop=True)
        answer = df_target.loc[0, answer_column]
        most_similar_candidate = None
        similarity = 1.0
        return answer, most_similar_candidate, similarity
    
    # get topic and subtopic
    try:
        topic, subtopic = intent.split("_")[0], intent.split("_")[1]
    except:
        topic, subtopic = intent, None
    
    # get dataframe 
    if subtopic:
        df_target = df.loc[(df["Topic"]==topic) & (df["Subtopic_1"]==subtopic)]
    else:
        df_target = df.loc[df["Topic"]==topic]
    
    # filter dataframe
    df_target = df_target[(~df_target.Question.isnull()) & (~df_target.Question_Sim.isnull()) & (~df_target.Question_Eng.isnull())]
    
    # if no dataframe, return don't understand
    if df_target.empty:
        df_target = df[df.Topic=="nothing"].reset_index(drop=True)
        answer = df_target.loc[0, answer_column]
        most_similar_candidate = None
        similarity = 1.0
        return answer, most_similar_candidate, similarity
   
    # reset index for dataframe
    df_target = df_target.reset_index(drop=True)

    similarities = []

    # do preprocessing for question here!!
    question = question.lower()
    question = converter.convert(question)
    question = change_cantonese_to_chinese(question)

    for i in range(len(df_target)):
        try:
            candidate = df_target.iloc[i][question_column]
            
            # do preprocessing for candidate here!!!
            candidate = candidate.lower()
            candidate = converter.convert(candidate)
            candidate = change_cantonese_to_chinese(candidate)

            question_pair = [[question, candidate]]
            similarity = similarity_model(question_pair)[0]["similarity"]
            similarities.append(similarity)

        except Exception as e:
            print(e)
            continue

    df_target['Similarity'] = similarities

    print(df_target[[question_column, answer_column, "Similarity"]])
    
    try:
        max_similarity = df_target.loc[df_target['Similarity'].idxmax()]
        answer = max_similarity[answer_column]
        most_similar_candidate = max_similarity[question_column]
        similarity = max_similarity['Similarity']
        return answer, most_similar_candidate, similarity

    except Exception as e:
        print(e)
        return None