from operator import mod
import opencc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import jieba
import re
import string
import os
from preprocessing import preprocess, lemmatizer
from configuration import Config


def create_intent_label(
    input_path, 
    col="label", 
    output_path = './data/nlu/intent_label.txt'
    ):

    df = pd.read_excel(input_path, sheet_name=0)
    df = df.replace({np.nan: None})
    intent_labels = df[col].unique()

    with open(output_path, 'w', encoding='utf-8') as out:
        for i in range(len(intent_labels)):
            if i == len(intent_labels) - 1:
                out.write(intent_labels[i].lower())
            else:
                out.write(intent_labels[i].lower()+'\n')


def create_slot_label(
    input_path, 
    col="text_with_entity", 
    lang=None, 
    output_path = './data/nlu/slot_label.txt'
    ):
    df = pd.read_excel(input_path, sheet_name=0)
    df = df.replace({np.nan: None})
    df = df[col]
    rows = df.shape[0]
    entities = set()

    for i in range(rows):
        print(i)
        sent_w_entities = df.loc[i]
        _, _, tags = extract_entity(sent_w_entities, lang=None)
        for tag in tags:
            if tag not in entities:
                entities.add(tag)
    entities = list(entities)

    with open(output_path, 'w', encoding='utf-8') as out:
        out.write('PAD'+'\n')
        out.write('UNK'+'\n')
        for i in range(len(entities)):
            if i == len(entities) - 1:
                out.write(entities[i])
            else:
                out.write(entities[i] + '\n')


def make_dataset(project_path, lang=None, split=False, test_size=0.2, delete_and_update=True):
    
    all_ok = True
    problems = []
    err_message = None
    
    # 1. check if directories of intent, slot, and excel exist
    intent_dir = os.path.join(project_path, "intent")
    slot_dir = os.path.join(project_path, "keyword")
    excel_dir = os.path.join(project_path, "excel")

    try:
        os.path.exists(intent_dir)
    except:
        all_ok = False
        err_message = "intent directory has not been created"
        return all_ok, err_message
    
    try:
        os.path.exists(slot_dir)
    except:
        all_ok = False
        err_message = "keyword directory has not been created"
        return all_ok, err_message
    
    try:
        os.path.exists(excel_dir)
    except:
        all_ok = False
        err_message = "excel directory has not been created"
        return all_ok, err_message
    
    # 2. check if text and excel files are exist
    intent_file = os.path.join(intent_dir, "intent_label.txt")
    slot_file = os.path.join(slot_dir, "slot_label.txt")
    excel_file = os.path.join(excel_dir, "dataset.xlsx")

    try:
        os.path.exists(intent_file)
    except:
        all_ok = False
        err_message = "intent_label.txt under intent directory has not been created"
        return all_ok, err_message
    
    try:
        os.path.exists(slot_file)
    except:
        all_ok = False
        err_message = "slot_label.txt under keyword directory has not been created"
        return all_ok, err_message
    
    try:
        os.path.exists(excel_file)
    except:
        all_ok = False
        err_message = "dataset.xlsx under excel directory has not been created"
        return all_ok, err_message

    # 3. read intent file, slot file, and excel file
    with open(intent_file, 'r', encoding='utf-8') as intent_label_file:
        lines = intent_label_file.readlines()
        intent_labels = [line.strip() for line in lines]

    with open(slot_file, 'r', encoding='utf-8') as slot_label_file:
        lines = slot_label_file.readlines()
        slot_labels = [line.strip() for line in lines]
    
    df = pd.read_excel(excel_file)
    df = df.replace({np.nan: None})

    # 5. Create train and dev directory
    train_dir = os.path.join(project_path, "train")
    dev_dir = os.path.join(project_path, "dev")

    # create directory if doesn't exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir) 
    if not os.path.exists(dev_dir):
        os.makedirs(dev_dir) 
    
    # 6. make training and validating files
    if not lang:
        sheet_name = None
        label_path = r'label'
        seq_in_path = r'seq.in'
        seq_out_path = r'seq.out'
    else:
        lang = lang.lower()
        sheet_name = 'en'
        label_path = r'label_{}'.format(lang)
        seq_in_path = r'seq_{}.in'.format(lang)
        seq_out_path = r'seq_{}.out'.format(lang)

    # define full paths for train and dev data
    train_label_path = os.path.join(train_dir, label_path)
    train_seq_in_path = os.path.join(train_dir, seq_in_path)
    train_seq_out_path = os.path.join(train_dir, seq_out_path)

    dev_label_path = os.path.join(dev_dir, label_path)
    dev_seq_in_path = os.path.join(dev_dir, seq_in_path)
    dev_seq_out_path = os.path.join(dev_dir, seq_out_path)

    # 7. main function
    def make_data(dataframe, mode):

        if mode=='train':
            label_path = train_label_path
            seq_in_path = train_seq_in_path
            seq_out_path = train_seq_out_path
        elif mode=='dev':
            label_path = dev_label_path
            seq_in_path = dev_seq_in_path
            seq_out_path = dev_seq_out_path
        
        # create file if not exist
        if not os.path.exists(label_path):
            with open(label_path, "w") as f:
                pass
        if not os.path.exists(seq_in_path):
            with open(seq_in_path, "a") as f:
                pass
        if not os.path.exists(seq_out_path):
            with open(seq_out_path, "a") as f:
                pass
        
        if delete_and_update:
            # delete file content first
            with open(label_path, 'r+', encoding='utf-8') as f, \
                open(seq_in_path, 'r+', encoding='utf-8') as f_in, \
                open(seq_out_path, 'r+', encoding='utf-8') as f_out:
                f.truncate(0)
                f_in.truncate(0)
                f_out.truncate(0)
        
        with open(label_path, 'a', encoding='utf-8') as f, \
            open(seq_in_path, 'a', encoding='utf-8') as f_in, \
            open(seq_out_path, 'a', encoding='utf-8') as f_out:

            for i in range(dataframe.shape[0]):
                
                ok = True
                label_columns = dataframe.filter(regex=r'^label')
                intents = []
                for label_column in label_columns:
                    intent = dataframe.loc[i, label_column].lower()
                    intents.append(intent)
                if intents:
                    for intent in intents:
                        try:
                            assert intent in intent_labels
                        except:
                            problem = f"Problem on data {mode}{i+1}: {intent} not in intent labels"
                            print(problem)
                            problems.append(problem)
                            ok = False
                
                sent = dataframe.loc[i, 'text_with_entity']
                try:
                    _, text, tag = extract_entity(sent, lang)  # text is list
                except:
                    problem = f'Problem on data {mode}{i+1}: {sent} has syntax problem'
                    print(problem)
                    problems.append(problem)
                    ok = False
                
                for t in tag:
                    try:
                        assert t in slot_labels
                    except:
                        problem = f'Problem on data {mode}{i+1}: {t} is not in slot labels'
                        print(problem)
                        problems.append(problem)
                        ok = False

                if ok:
                    f.write(" ".join(intents) + '\n')  # join elements in list with space
                    f_in.write(' '.join(text).lower() + '\n')
                    f_out.write(' '.join(tag) + '\n')

    if not split:
        df_train = df[df['mode']=='train'].reset_index(drop=True)
        df_dev = df[df['mode']=='test'].reset_index(drop=True)
        make_data(df_train, mode='train')
        make_data(df_dev, mode='dev')
    
    else:
        intents = df['label'].unique()
        for intent in intents:
            df_intent = df[df['label']==intent]
            df_train, df_dev = train_test_split(df_intent, test_size=test_size)
            df_train = df_train.reset_index(drop=True)
            df_dev = df_dev.reset_index(drop=True)
            make_data(df_train, mode='train')
            make_data(df_dev, mode='dev')
    
    if problems:
        problem_path = os.path.join(excel_dir, "dataset_problems.txt")
        with open(problem_path, 'w', encoding='utf-8') as f:
            for problem in problems:
                f.write(problem+'\n')
        all_ok = False
        err_message = f"Problems are found in question set, please see {problem_path} for detail."

    return all_ok, err_message


def annotate_segmented_text(annotated: list):
    text = " ".join([a.split("_")[0] for a in annotated])
    for i in range(len(annotated)):
        if annotated[i].split("_")[0] == "[":
            
            # get information about position and entity type
            
            # get keyword
            start = i + 1
            end = start
            while annotated[end].split("_")[0] != "]":
                end += 1
          
            # get entity type
            err_msg = f"Problems occur in following text:\n{text}\n"
            err_msg += "Reminder: [xxx] should be follow by (yyy)"
            assert annotated[end+1].split("_")[0] == '(', err_msg
            entity_type = annotated[end+2].split("_")[0]
            k = 1
            while annotated[end+2+k].split("_")[0] != ')':
                if annotated[end+2+k] == '__O' or ' _O':
                    entity_type += '_'
                else:
                    entity_type += annotated[end+2+k].split("_")[0]
                k += 1
                # print(entity_type)
            assert annotated[end+2+k].split("_")[0] == ')', err_msg
            
            print(annotated[start:end])
            # print(annotated[end+2:end+2+k])
            print(entity_type)
 
            # assert annotated[end + 1].split("_")[0] == "(" and annotated[end + 3].split("_")[0] == ")", "wrong format"
            # entity_type = annotated[end + 2].split("_")[0]
            
            # annotate entity
            for j in range(start, end):
                if j == start:
                    annotated[j] = annotated[j].split("_")[0] + "_B-" + entity_type
                else:
                    annotated[j] = annotated[j].split("_")[0] + "_I-" + entity_type
            
            # remove [], (entity type)
            annotated_final = []
            for i in range(len(annotated)):
                index_square_brackets = [start-1, end]
                index_brack_left = [end+1]
                index_entitiy_type = []
                for num in range(k+1):
                    index_entitiy_type.append(end+2+num)

                unwanted_index = index_square_brackets + index_brack_left + index_entitiy_type
                if i in unwanted_index:
                    continue
                    # print(annotated[i])
                else:
                    annotated_final.append(annotated[i])
            
            # recursive
            return annotate_segmented_text(annotated_final)

    return annotated


def remove_punctuation_from_annotated_text(annotated_text: list):
    p = string.punctuation
    punctuation_list = [p[i] for i in range(len(p))] + ['？']
    result = []
    for word in annotated_text:
        if word.split("_")[0] not in punctuation_list:
            result.append(word)
    return result


def extract_entity(text, lang=None):
    """
    function to extract entities with their entity types
    我想問[圖書館](facility)喺邊？  --> O O O B-facility I-facility I-facility O O
    """
    text = preprocess(text, lang=None)
    words = []
    text = text.strip()  # remove left space and right space
    # text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    splitted_text = ', '.join(jieba.cut(text)).split(', ')  # get segmented text (list)

    for t in splitted_text:  # iterate each segmented result
        if t != ' ':  # skip all the white space
            has_chinese = re.findall(r'[\u4e00-\u9fff]+', t)  # find if a segmented word is chinese
            if len(t) > 1 and has_chinese:  # e.g. "你好"
                for i in range(len(t)):
                    words.append(t[i])
            else:
                # try to do lemmantization for english words
                try:
                    t = lemmatizer.lemmatize(t)
                except:
                    pass
                words.append(t)
    print(''.join(words))
    annotated = [word + "_O" for word in words]  # initialization
    annotated = annotate_segmented_text(annotated)
    result = remove_punctuation_from_annotated_text(annotated)

    text = [r.split("_")[0] for r in result]
    bio_tag = [r.split("_")[1] for r in result]

    return result, text, bio_tag


if __name__ == '__main__':

    # text = "[Olympian City](mall)这么多期的[礼宾处](facility)电话几号"
    # results = extract_entity(text, 'zh')
    # for r in results:
    #     print(r)
    
    make_dataset(
        project_path="./data/project/immd/", 
        split=True, 
        delete_and_update=False
        )
    
    create_intent_label(
        input_path='data/from_ct/20231213/dataset.xlsx', 
        output_path='./data/project/immd/intent/intent_label_new.txt'
    )

    create_slot_label(
        input_path='data/from_ct/20231213/dataset.xlsx',
        output_path='./data/project/immd/keyword/slot_label_new.txt'
    )
