from pprint import pprint
import logging
from logging.handlers import RotatingFileHandler
import string, time
import collections
import sys, requests, os, json, argparse
import opencc
import paddle
from flask import Flask, jsonify, request, make_response, render_template
from flask_expects_json import expects_json
from jsonschema import ValidationError
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import pandas as pd
from paddlenlp import Taskflow
from paddlenlp.transformers import ErnieMTokenizer, ErnieMModel

from configuration import Config, session
from segmentation import segmentate_sentence, extract_key_word
from preprocessing import preprocess, download_nltk
from nlu_train import JointModel_JA
from nlu_infer import ErnieTokenizer, ErnieModel, JointModel, load_dict, nlu_predict, nlu_predict_ja
from kg_qa import get_answer_from_mall_kg
from kg_qa_estate import get_answer_from_estate_kg
from similarity_infer import get_answer_from_faq, get_answer_from_faq_new
from postprocessing import postprocess
from faq_processing import faq_processing
from kg_processing import kg_processing
from publish import publish_graph, publish_faq, delete_faq
from push2neo4j import push_data

from nlu_data import make_dataset
from nlu_train import nlu_model_train
from similarity_train import similarity_model_train
from similarity_data import validate_similarity_data

logger = logging.getLogger("Rotating Log")
logger.setLevel(logging.INFO)
logging.basicConfig(filename='./log/nlp.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
handler = RotatingFileHandler("./log/nlp.log", maxBytes=10240, backupCount=5)
logger.addHandler(handler)

app = Flask(__name__)
server = Config.server
port = Config.port

# initialize executor
executor = ThreadPoolExecutor(2)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--kms_knowledge_list_url', type=str)
    # parser.add_argument('--develop', dest='develop', action='store_true')
    # parser.set_defaults(develop=False)
    return parser


def get_mall_estate_ids():
    # retrieve malls and estates
    malls = []
    estates = []
    with open(Config.site_id_malls, "r", encoding="utf-8") as f_mall, open(Config.site_id_estates, "r", encoding="utf-8") as f_estate:
        lines_mall = f_mall.readlines()
        for line in lines_mall:
            malls.append(line.strip())
        lines_estate = f_estate.readlines()
        for line in lines_estate:
            estates.append(line.strip())
    
    return malls, estates


def convert_dict(slot_list):

    """
    {
        entity_type1: [e1, e2, ...], 
        entity_type2: [d1, d2, ...]
    }
    """

    if not slot_list:
        return None

    new_dict = {}

    for slot_dict in slot_list:
        entity_type = slot_dict["type"]
        entity_name = slot_dict["text"]

        if not new_dict.get(entity_type):
            new_dict[entity_type] = []
            new_dict[entity_type].append(entity_name)

        else:
            new_dict[entity_type].append(entity_name)

    return new_dict


def get_path(intent):
    if not intent:
        return None
    intent = intent.lower()
    if intent in ["chat", "chat_greeting", "chat_violent"]:
        return 'small_talk'
    elif intent in faq_topics_estate+faq_topics_mall:
        return 'faq'
    else:
        return 'kg'
    

def nlu(input_text, lang, sid, malls, estates):
    # function to do natural language understanding
    if sid in malls:
        site_type = "mall"
    elif sid in estates:
        site_type = "estate"

    if lang in ["zh-hk", "zh-cn"]:
        if site_type == "estate":
            intent_labels, slots = \
                nlu_predict(input_text, nlu_model_zh_estate, nlu_zh_tokenizer, id2intent_estate, id2slot_estate, converter, max_seq_len=max_seq_len)
        elif site_type == "mall":
            intent_labels, slots = \
                nlu_predict(input_text, nlu_model_zh_mall, nlu_zh_tokenizer, id2intent_mall, id2slot_mall, converter, max_seq_len=max_seq_len)
                
    elif lang in ["en"]:
        if site_type == "estate":
            intent_labels, slots = \
                nlu_predict(input_text, nlu_model_en_estate, nlu_en_tokenizer, id2intent_estate, id2slot_estate, converter, max_seq_len=max_seq_len)
        elif site_type == "mall":
            intent_labels, slots = \
                nlu_predict(input_text, nlu_model_en_mall, nlu_en_tokenizer, id2intent_mall, id2slot_mall, converter, max_seq_len=max_seq_len)
    
    elif lang in ["ja"]:
        if site_type == "mall":
            intent_labels, slots = \
                nlu_predict_ja(input_text, nlu_model_ja_mall, nlu_ja_tokenizer, id2intent_mall, id2slot_mall, None, max_seq_len=max_seq_len)
               
    try:
        intent, intent_prob = intent_labels[0]
    except:
        intent, intent_prob = None, None
    
    return intent, intent_prob, slots


def dst_baseline(old_intent, old_slot_list, new_slot_list):
    # intent to be managed 
    old_slots = convert_dict(old_slot_list)
    new_slots = convert_dict(new_slot_list)

    # for new_k, new_v in new_slots.items():
    #     old_slots[new_k] = new_v
    for entity_type, entity_list in new_slots.items():
        old_slots[entity_type] = entity_list
    
    # create new list for slots detected
    # final_slot_list = []
    # for k, v in old_slots.items():
    #     for old_slot in old_slot_list:
    #         if old_slot["text"] == v:
    #             final_slot_list.append(old_slot)
    # 
    #     temp_slots = [i["text"] for i in final_slot_list]  # store values of previous detected entities

    #     for new_slot in new_slot_list:
    #         if new_slot["text"] == v and v not in temp_slots:
    #             final_slot_list.append(new_slot)       
    
    final_slot_list = []
    for entity_type, entity_list in old_slots.items():
        # for each entity type
        for entity in entity_list:
            # for each entity
            for old_slot in old_slot_list:  # in API format
                if old_slot["type"] == entity_type and old_slot["text"] == entity:
                    final_slot_list.append(old_slot)
        
            temp_slots = [i["text"] for i in final_slot_list]  # store values of previous detected entities
    
            for new_slot in new_slot_list:  # in API format
                if new_slot["text"] == entity and entity not in temp_slots:
                    final_slot_list.append(new_slot)   

    return old_intent, final_slot_list


def get_template_faq_df(dir):
    assert os.path.isdir(dir)
    file_name_list = os.listdir(dir)
    assert file_name_list, f"no file found under {dir}."
    merge_df = pd.DataFrame()
    file_path_list = [os.path.join(dir, i) for i in file_name_list]
    df_list = [pd.read_excel(i) for i in file_path_list]
    for df in df_list:
        merge_df = merge_df.append(df, ignore_index=True)
    df = df.drop_duplicates()
    # print(df)
    return df


def nlg(input_text, intent, slots, sid, version, lang, malls, estates):

    if not intent:
        intent = "nothing"
    
    # list_of_files = os.listdir(template_faq_dir)
    # assert list_of_files, f"no file under {template_faq_dir}."
    # template_faq_file_name = sorted(list_of_files, key = lambda x: os.path.getmtime(os.path.join(template_faq_dir, x)))[-1]
    # template_faq_file_path = os.path.join(template_faq_dir, template_faq_file_name)
    
    answer_options = []
    answer_type = "direct"

    # KG
    if get_path(intent) == 'kg':

        template_faq_dir = os.path.join(Config.template_db_base_path, sid, version)
        template_faq_db = get_template_faq_df(template_faq_dir)

        confidence_score = 1.0

        slots = convert_dict(slots) # transform slots into other format
        if sid in malls:
            try:
                topic_stat, answer_text, answer_options, follow_up = \
                    get_answer_from_mall_kg(lang, sid, version, session, intent, slots, template_faq_db)
            except:
                (topic_stat, answer_text), answer_options, follow_up = \
                    get_answer_from_mall_kg(lang, sid, version, session, intent, slots, template_faq_db), [], False
        
        elif sid in estates:
            try:
                topic_stat, answer_text, answer_options, follow_up = \
                    get_answer_from_estate_kg(lang, sid, version, session, intent, slots, template_faq_db)
            except:
                (topic_stat, answer_text), answer_options, follow_up = \
                    get_answer_from_estate_kg(lang, sid, version, session, intent, slots, template_faq_db), [], False
            
        if follow_up:
            answer_type = "follow-up"
        else:
            answer_type = "option" if answer_options else "direct"
        
        return topic_stat, answer_text, confidence_score, answer_options, answer_type
        
    # FAQ
    elif get_path(intent) == 'faq':

        # get template_faq_excel file
        template_faq_dir = os.path.join(Config.template_db_base_path, sid, version)
        template_faq_db = get_template_faq_df(template_faq_dir)

        topic_stat = intent.split("_")

        if lang in ["zh-hk", "zh-cn"]:
            similarity_model = similarity_zh
        elif lang in ["en", "ja"]:
            similarity_model = similarity_zh

        # answer_text, _, confidence_score = get_answer_from_faq(lang, version, input_text, intent, sim_tokenizer, sim_model, template_faq_db)
        
        answer_text, _, confidence_score = get_answer_from_faq_new(lang, input_text, similarity_model, intent, template_faq_db)
        
        return topic_stat, answer_text, confidence_score, answer_options, answer_type

    # Small Talk
    elif get_path(intent) == 'small_talk':
        # get small talk file
        small_talk_dir = os.path.join(Config.small_talks_base_path, sid, version)
        small_talk_db = get_template_faq_df(small_talk_dir)

        topic_stat = intent.split("_")

        if lang in ["zh-hk", "zh-cn"]:
            # sim_tokenizer = sim_tokenizer_zh
            # sim_model = sim_model_zh
            similarity_model = similarity_zh
        elif lang in ["en"]:
            # sim_tokenizer = sim_tokenizer_en
            # sim_model = sim_model_en
            similarity_model = similarity_en
            # similarity_model = similarity_zh
        
        # list_of_files = os.listdir(small_talk_dir)
        # assert list_of_files, f"no small talk file under {small_talk_dir}."
        # small_talk_file_name = sorted(list_of_files, key = lambda x: os.path.getmtime(os.path.join(small_talk_dir, x)))[-1]
        # small_talk_file_path = os.path.join(small_talk_dir, small_talk_file_name)
        
        # answer_text, _, confidence_score = get_answer_from_faq(lang, version, input_text, intent, sim_tokenizer, sim_model, small_talk_db)
        
        answer_text, _, confidence_score = get_answer_from_faq_new(lang, input_text, similarity_model, intent, small_talk_db)

        return topic_stat, answer_text, confidence_score, answer_options, answer_type



@app.route('/api/nlp_pipeline', methods=['GET', 'POST'])
def pipeline():

    ####################################### Validation Start ##############################################
    # get json contents
    try:
        content = request.json
    except:
        results = {
            "success": False,
            "error": {
                "reason": "invalid json format" 
            }
        }
        return json.dumps(results, ensure_ascii=False, indent=3)

    try:
        lang = content["language"]
        question = content["question"]
        version = content["version"]
        site_id = content["site_id"]
        history = content["history"]
    except:
        results = {
            "success": False,
            "error": {
                "reason": "language, question, version, site_id, history must be provided" 
            }
        }
        return json.dumps(results, ensure_ascii=False, indent=3)
    
    # check data type
    if not isinstance(lang, str) or not isinstance(question, str) or not isinstance(version, str) or \
        not isinstance(site_id, str) or not isinstance(history, list):
        results = {
            "success": False,
            "error": {
                "reason": "language, question, version, site_id, history must in valid data type" 
            }
        }
        return json.dumps(results, ensure_ascii=False, indent=3)
    ########################################## Validation End ##############################################


    ########################################## NLP start ###################################################
    try:

        malls, estates = get_mall_estate_ids()
    
        do_nlu = True
        previous_ans_options = None
        old_slot_list = None
        old_topic_list = None
        previous_ans_type = None
        old_intent = None
        old_intent_confidence_score = None
    
        if lang in ["zh-hk", "zh-cn"]:
            stopwords = stopwords_zh
        elif lang in ["en"]:
            stopwords = stopwords_en
        
        if history:
            history = history[0]
            old_question, old_answer = history.get("question"), history.get("answer")
            if old_question and old_answer:
                old_slot_list = old_question.get("entity")
                old_topic_list = old_question.get("topic")
                if not old_topic_list:
                    old_topic_list = ["nothing"]
                old_intent = '_'.join(old_topic_list)
                old_intent_confidence_score = old_question.get("con")
                previous_ans_type = old_answer.get("type")
                previous_ans_options = old_answer.get("options")
        

        # when user click for options
        if old_intent and old_intent in ['foodtype', 'shoptype', 'shopproduct', 'facilitiesinformation_address'] and \
            previous_ans_options and previous_ans_type!="follow-up":
            
            if previous_ans_options:
                if question:
                    if question in previous_ans_options: 
                        do_nlu = False
                        intent = "choose_from_options"
                        intent_prob = old_intent_confidence_score
                        slots = old_slot_list
                        
                        # start adding entity
                        temp_name = None
                        # 1. get option name
                        if question in previous_ans_options:
                            if old_intent in ['foodtype', 'shoptype', 'shopproduct']:
                                temp_name = question.split("】")[-1]
                            elif old_intent in ['facilitiesinformation_address']:
                                temp_name = question
                        
                        else:
                            temp_name = question
                        
                        # change tc to sc
                        if lang=="zh-hk":
                            temp_name = converter.convert(temp_name)
                        
                        # 2. type = shop | restaurant | facility
                        if old_intent in ['foodtype']:
                            temp_type = "restaurant"
                        elif old_intent in ['shoptype', 'shopproduct']:
                            temp_type = "shop"
                        elif old_intent in ['facilitiesinformation_address']:
                            temp_type = "facility"
                        
                        # 3. con = 1.0
                        temp_con = 1.0
                        
                        # 4. start, end
                        temp_start = 0
                        temp_end = len(temp_name)-1
                        
                        # 5. append to slots
                        slots.append({
                            "start": temp_start,
                            "end": temp_end,
                            "text": temp_name,
                            "type": temp_type,
                            "con": temp_con
                        })
    
        # Preprocess the Question
        question = preprocess(question, lang)

        # Try to find exact match of restaurant and shop name
        if previous_ans_type != "follow-up":  # do not do this step for follow-up question
            matched_restaurant_shop_list = []
            restaurant_shop_json = os.path.join(Config.restaurant_shop_dictionary_base_path, f"{site_id}_{version}.json")
            
            if os.path.exists(restaurant_shop_json):
                # open json file
                with open(restaurant_shop_json, "r", encoding="utf-8") as f:
                    # transform from json to dictionary
                    data = json.load(f)
                # get attribute 'data' from dictionary
                data = data.get("data")  # list of dictionary
                
                for i in range(len(data)):
                    restaurant_shop_dict = data[i]
                    restaurant_shop_synonyms = restaurant_shop_dict.get('synonyms').split(", ")
                    if restaurant_shop_synonyms:
                        for restaurant_shop_synonym in restaurant_shop_synonyms:
                            if question == restaurant_shop_synonym:
                                if lang == "zh-hk":
                                    matched_restaurant_shop_list.append(restaurant_shop_dict.get('name'))
                                elif lang == "zh-cn":
                                    matched_restaurant_shop_list.append(restaurant_shop_dict.get('name_sim'))
                                elif lang == "en":
                                    matched_restaurant_shop_list.append(restaurant_shop_dict.get('name_eng'))
            
            if matched_restaurant_shop_list:
                matched_restaurant_shop_list = list(set(matched_restaurant_shop_list))
                do_nlu = False
                intent = "get_all_restaurant_shop_information"
                intent_prob = 1.0
                slots = [{                
                    "start": 0,
                    "end": len(matched_restaurant_shop)-1,
                    "text": matched_restaurant_shop,
                    "type": "restaurant-shop",
                    "con": 1.0
                } for matched_restaurant_shop in matched_restaurant_shop_list]
            print(matched_restaurant_shop_list)
    
        # NLU
        if do_nlu:
            intent, intent_prob, slots = nlu(question, lang, site_id, malls, estates)
            print(intent, slots) 
        
        # DST
        if previous_ans_type == "follow-up":
            old_intent = '_'.join(old_topic_list)
            intent_prob = old_intent_confidence_score
            new_slot_list = slots
            intent, slots = dst_baseline(old_intent, old_slot_list, new_slot_list)
        
        # NLG
        topic_stat, answer_text, confidence_score, answer_options, answer_type = nlg(question, intent, slots, site_id, version, lang, malls, estates)
        try:
            answer_text = answer_text.replace("\n", "")
            # answer_text = answer_text.replace("奧海城", "商場")
            # answer_text = answer_text.replace("奥海城", "商场")
            # answer_options = [i.replace("奧海城", "商場") for i in answer_options]
            # answer_options = [i.replace("奥海城", "商场") for i in answer_options]
            answer_text = answer_text.strip()
        except:
            pass

        # keywords (old version):
        if slots:
            keywords = [converter_s2t.convert(i["text"]) for i in slots]
            keywords = list(set(keywords))
            keywords = [i for i in keywords if i not in stopwords_zh]
        else:
            keywords = []

        # sentiment analysis
        senta_result = senta(question)
        sentiment_label = senta_result[0]["label"]
        sentiment_score = senta_result[0]["score"]
        ########################################## NLP End #####################################################
        
        # api_result
        results = {"success": True, 
                   "data": {
                       "language": lang,
                       "question": {
                           "text": question,
                           "text_seg": segmentate_sentence(question),
                           # "keyword": [i for i in extract_key_word(question, stopwords) if i],
                           "keyword": keywords,
                           "topic": intent.split("_") if intent and intent != "nothing" else None,
                           "topic_stat": intent.split("_") if intent and intent != "nothing" else None,
                           "con": float(intent_prob) if intent_prob else None,
                           "entity": slots,
                           "sentiment": {
                                "label": sentiment_label,
                                "score": sentiment_score
                           }
                        },
                       "answer": {
                           "type": answer_type,
                           "text": answer_text,
                           "text_tts": postprocess(answer_text, lang),
                           "options": answer_options,
                           "con": float(confidence_score)
                           }, 
                       "version": version, 
                       "site_id": site_id
                        } 
                   }
        
        results_json = json.dumps(results, ensure_ascii=False, indent=3)
        
        return results_json
    
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        results = {
            "success": False,
            "error": {
                "reason": str(e) 
            }
        }
        results_json = json.dumps(results, ensure_ascii=False, indent=3)
        return results_json


# Process FAQ Files
@app.route('/api/faq_processing', methods=['GET', 'POST'])
def api_faq_processing():
    # get json contents
    try:
        content = request.json
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        results = {
            "success": False,
            "error": {
                "reason": str(e) 
            }
        }
        print("invalid json format")
        return json.dumps(results, ensure_ascii=False, indent=3)
    
    site_id = content.get("site_id")
    version = content.get("version")
    process = content.get("process")
    
    if process:
        
        file_path = process.get("file_path")
        prev_file_path = process.get("prev_file_path")
        action = process.get("action")
        
        if action not in ["create", "update", "delete"]:
            results = {
                    "success": False,
                    "error": {
                        "reason": "wrong action is given, possible actions are create, update, or delete" 
                    }
                }
            return json.dumps(results, ensure_ascii=False, indent=3)
        
        if action == "update":
            if not file_path or not prev_file_path:
                results = {
                    "success": False,
                    "error": {
                        "reason": "file_path or prev_file path not provided" 
                    }
                }
                return json.dumps(results, ensure_ascii=False, indent=3)
        
        error = faq_processing(file_path, site_id, version, action, prev_file_path)
        
        if not error:
            results = {
                "success": True,
                "data": {}
            }
        
        else:
            results = {
                "success": False,
                "error": {
                    "reason": str(error) 
                }
            }
            print(error)

        return json.dumps(results, ensure_ascii=False, indent=3)


# Process KG files
@app.route('/api/kg_processing', methods=['GET', 'POST'])
def api_kg_processing():
    try:
        content = request.json
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        results = {
            "success": False,
            "error": {
                "reason": str(e) 
            }
        }
        return json.dumps(results, ensure_ascii=False, indent=3)
    
    site_id = content.get("site_id")
    version = content.get("version")
    process = content.get("process")
    history = content.get("history")
    action = process.get("action")
    file_path = process.get("file_path")
    prev_file_path = process.get("prev_file_path")
    callback = content.get("callback")
    callback_id = content.get("callback_id")

    try:
        assert action in ["create", "update", "delete"]
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        results = {
            "success": False,
            "callback_id": callback_id,
            "error": {
                "reason": "wrong action is given, possible actions are create, update, or delete" 
            }
        }
        return json.dumps(results, ensure_ascii=False, indent=3)
    
    try:
        assert version in ["editing", "production"]
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        results = {
            "success": False,
            "callback_id": callback_id,
            "error": {
                "reason": "wrong version is given, possible versions are editing or production"
            }
        }
        return json.dumps(results, ensure_ascii=False, indent=3)

    if action == "update":
        if not file_path or not prev_file_path:
            results = {
            "success": False,
            "callback_id": callback_id,
            "error": {
                "reason": "file_path or prev_file path not provided" 
                }
            }
            return json.dumps(results, ensure_ascii=False, indent=3)
    
    try:
        executor.submit(job_kg_processing, callback, callback_id, site_id, version, action, file_path, prev_file_path, history)
        results = {
            "success": True,
            "callback_id": callback_id
        }
        return json.dumps(results, ensure_ascii=False, indent=3)
    
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        results = {
            "success": False,
            "callback_id": callback_id,
            "error": {
                "reason": str(e) 
                }
            }
        return json.dumps(results, ensure_ascii=False, indent=3)


def job_kg_processing(callback, callback_id, site_id, version, action, file_path, prev_file_path, history):    
    
    print("start processing KG")
    try:
        error = kg_processing(session, file_path, site_id, version, action, prev_file_path, history)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        json_obj = {
            "success": False,
            "callback_id": callback_id,
            "error": {
                "reason": str(error) 
            }
        }
        requests.post(callback, json=json_obj, verify=False)
    
    if not error:
        json_obj = {
            "success": True,
            "callback_id": callback_id
        }
        print("processing KG sucessfully")
    else:
        json_obj = {
            "success": False,
            "callback_id": callback_id,
            "error": {
                "reason": str(error) 
            }
        }
        print("processing KG not sucessfully: ", str(error))
    print(f"start callback to {callback}")
    try:
        requests.post(callback, json=json_obj, verify=False)
        print(f"finish callback to {callback}")
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        print(e)


# Process Publishing
@app.route('/api/publish_data', methods=['GET', 'POST'])
def api_publish_data():
    try:
        content = request.json
        pprint(content)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        results = {
            "success": False,
            "error": {
                "reason": str(e) 
            }
        }
        print(e)
        return json.dumps(results, ensure_ascii=False, indent=3)
    
    callback = content.get("callback")
    callback_id = content.get("callback_id")
    site_id = content.get("site_id")
    
    # list contains json objects {"file_path": "/path/to/file"}
    faq_file_paths_list = content.get("datafile").get("production").get("faq")
    kg_file_paths_list = content.get("datafile").get("production").get("kg")
    
    if not faq_file_paths_list and not kg_file_paths_list:
        results = {
            "success": False,
            "callback_id": callback_id,
            "error": {
                "reason": "Neither faq files or kg files are not provided" 
            }
        }
        print("Neither faq files or kg files are not provided")
        return json.dumps(results, ensure_ascii=False, indent=3)
    
    executor.submit(job_publish_data, 
        callback, callback_id, site_id, kg_file_paths_list, faq_file_paths_list)

    results = {
        "success": True,
        "callback_id": callback_id
    }

    return json.dumps(results, ensure_ascii=False, indent=3)


def job_publish_data(callback, callback_id, site_id, kg_file_paths_list, faq_file_paths_list):    
    print("start pulishing")

    success = True

    if faq_file_paths_list:
        try:
            for faq_json in faq_file_paths_list:
                faq_file_path = faq_json.get("file_path")
                action = faq_json.get("action")
                if action == "delete":
                    delete_faq(faq_file_path)
                elif action == "publish":
                    print("ready to publish faq or small talk")
                    error = publish_faq(session, site_id, faq_file_path)
                    if error:
                        success = False
                        json_obj = {
                            "success": False,
                            "callback_id": callback_id,
                            "error": {
                                "reason": str(error) 
                            }
                        }
                        requests.post(callback, json=json_obj, verify=False)
        
        except Exception as error:
            logging.error("Exception occurred", exc_info=True)
            json_obj = {
                "success": False,
                "callback_id": callback_id,
                "error": {
                    "reason": str(error) 
                }
            }
            requests.post(callback, json=json_obj, verify=False)
    

    if kg_file_paths_list:
        try:
            in_use_file_list = []
            for kg_json in kg_file_paths_list:
                if not kg_json.get("action"):
                    in_use_file_list.append(kg_json.get("file_path"))
                else:
                    if kg_json.get("action") not in ["delete", "publish"]:
                        in_use_file_list.append(kg_json.get("file_path"))

            for kg_json in kg_file_paths_list:
                kg_file_path = kg_json.get("file_path")
                kg_file_time = kg_json.get("timecreated")
                action = kg_json.get("action")
                
                if action == "publish":
                    try:
                        error = publish_graph(session, site_id, kg_file_path=kg_file_path, in_use_file_list=in_use_file_list)
                        if error:
                            print("error from publishing KG: ", error)
                            success = False
                            json_obj = {
                                "success": False,
                                "callback_id": callback_id,
                                "error": {
                                    "reason": str(error) 
                                }
                            }
                            requests.post(callback, json=json_obj, verify=False)
                    except Exception as error:
                        logging.error("Exception occurred", exc_info=True)
                        json_obj = {
                            "success": False,
                            "callback_id": callback_id,
                            "error": {
                                "reason": str(error) 
                            }
                        }
                        requests.post(callback, json=json_obj, verify=False)


                elif action == "delete":
                    try:
                        delete_file_name = kg_file_path.split("_")[-1][:-5]
                        session.run(f"match (n) where n.site_id='{site_id}' and n.mode='production' and n.file_name='{delete_file_name}' detach delete n")
                    except Exception as error:
                        logging.error("Exception occurred", exc_info=True)
                        json_obj = {
                        "success": False,
                        "callback_id": callback_id,
                        "error": {
                            "reason": str(error) 
                            }
                        }
                        requests.post(callback, json=json_obj, verify=False)
        

        except Exception as error:
            logging.error("Exception occurred", exc_info=True)
            json_obj = {
                "success": False,
                "callback_id": callback_id,
                "error": {
                    "reason": str(error) 
                }
            }
            requests.post(callback, json=json_obj, verify=False)
    
    if success:
        json_obj = {
            "success": True,
            "callback_id": callback_id
        }
    
        requests.post(callback, json=json_obj, verify=False)
        print("finish pulishing")


# Update NLP modes
@app.route('/api/update_nlp_models', methods=['GET', 'POST'])
def update_nlp_models():
    try:
        content = request.json
        model_type = content.get("model_type")
        lang = content.get("lang")
        site_type = content.get("site_type")
        callback = content.get("callback")
        callback_id = content.get("callback_id")
    
        # assert model_type in ["nlu", "similiarity"]
        assert model_type in ["nlu", "similarity"], "model_type should either be nlu or similarity"
        # assert lang in ["zh", "en"], "lang should either be zh or en"
        assert site_type in ["mall", "estate"], "site_type should either be mall or estate"
        # assert file name of question set
        # copy file to ./data/question_set

    except Exception as e:
        results = {
                "success": False,
                "callback_id": callback_id,
                "error": {
                    "reason": str(e) 
                    }
            }
        return json.dumps(results, ensure_ascii=False, indent=3)
    
    # if input is valid, pass to executor
    if model_type=="nlu":
        # use nlu_data.py to convert excel to train data and test data
        if site_type == 'mall':
            question_set_path = os.path.join(Config.nlu_model_base_path, 'question_set_mall.xlsx')
        elif site_type == 'estate':
            question_set_path = os.path.join(Config.nlu_model_base_path, 'question_set_estate.xlsx')
    
        nlu_problem_path = os.path.join(Config.nlu_model_base_path, 'dataset_problems.txt')
        data_ok = make_dataset(site_type, question_set_path, nlu_problem_path, lang, split=False, test_size=None, delete_and_update=True)
        
        if not data_ok:
            results = {
                    "success": False,
                    "callback_id": callback_id,
                    "error": {
                        "reason": f"Problems are found in question set, please see {nlu_problem_path} for detail." 
                        }
                }
            print(f"Problems are found in question set, please see {nlu_problem_path} for detail.")
            return json.dumps(results, ensure_ascii=False, indent=3)

        elif data_ok:
            executor.submit(job_update_nlu, site_type, lang, callback, callback_id)
    

    elif model_type=="similarity":
        similarity_question_set_path = os.path.join(Config.similarity_model_base_path, 'training_set_similarity.xlsx')
        similarity_problem_path = os.path.join(Config.similarity_model_base_path, 'dataset_problems.txt')
        data_ok = validate_similarity_data(similarity_question_set_path, similarity_problem_path)

        if not data_ok:
            results = {
                    "success": False,
                    "callback_id": callback_id,
                    "error": {
                        "reason": f"Problems are found in question set, please see {similarity_problem_path} for detail." 
                        }
                }
            print(f"Problems are found in question set, please see {similarity_problem_path} for detail.")
            return json.dumps(results, ensure_ascii=False, indent=3)

        elif data_ok:
            executor.submit(job_update_similarity, lang, callback, callback_id)

    results = {
        "success": True,
        "callback_id": callback_id
    }

    return json.dumps(results, ensure_ascii=False, indent=3)


def job_update_similarity(lang, callback, callback_id):
    train_success = True

    data_dir = os.path.join(Config.similarity_model_base_path, "training_set_similarity.xlsx")
    save_dir = os.path.join(Config.similarity_model_ckpt_path, lang)
    init_ckpt = os.path.join(Config.similarity_model_ckpt_path, lang)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        similarity_model_train(lang, data_dir, save_dir, init_ckpt, seed=1000, batch_size=32, max_seq_length=128, 
                           epochs=50, learning_rate=5e-5, warmup_proportion=0.0, weight_decay=0.0, use_gpu=True)
    except Exception as e:
        train_success = False
        json_obj = {
                "success": False,
                "callback_id": callback_id,
                "error": {
                    "reason": str(e)
                    }
            }
        print(str(e))
        requests.post(callback, json=json_obj, verify=False)

    if train_success:
        json_obj = {
                "success": True,
                "callback_id": callback_id,
            }
        requests.post(callback, json=json_obj, verify=False)


def job_update_nlu(site_type, lang, callback, callback_id):
    
    # use nlu_train.py to fine-tune nlp model
    train_success = True

    today_date = date.today().strftime("%Y/%m/%d")
    today_date = today_date.replace("/", "")

    if site_type == 'mall':
        intent_path = os.path.join(Config.nlu_model_base_path, 'intent_label_mall.txt')
        slot_path = os.path.join(Config.nlu_model_base_path, 'slot_label_mall.txt')
        train_path = os.path.join(Config.nlu_model_base_path, 'train_mall')
        dev_path = os.path.join(Config.nlu_model_base_path, 'dev_mall')
        save_path = os.path.join(Config.nlu_model_ckpt_path, f'mall_{lang}')
    elif site_type == 'estate':
        intent_path = os.path.join(Config.nlu_model_base_path, 'intent_label_estate.txt')
        slot_path = os.path.join(Config.nlu_model_base_path, 'slot_label_estate.txt')
        train_path = os.path.join(Config.nlu_model_base_path, 'train_estate')
        dev_path = os.path.join(Config.nlu_model_base_path, 'dev_estate')
        save_path = os.path.join(Config.nlu_model_ckpt_path, f'estate_{lang}')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    try:
        nlu_model_train(intent_path, slot_path, train_path, dev_path, save_path, lang, 
                    num_epoch=50, batch_size=16, max_seq_len=512, learning_rate=3e-5, weight_decay=0.01, 
                    warmup_proportion=0.1, max_grad_norm=1.0, seed=666, log_step=50, eval_step=100, use_gpu=True)
    except Exception as e:
        train_success = False
        json_obj = {
                "success": False,
                "callback_id": callback_id,
                "error": {
                    "reason": str(e)
                    }
            }
        print(str(e))
        requests.post(callback, json=json_obj, verify=False)

    if train_success:
        json_obj = {
                "success": True,
                "callback_id": callback_id,
            }
        requests.post(callback, json=json_obj, verify=False)


# Health Check
@app.route('/healthy', methods=['GET'])
def check_health():
    results = {
        "status": "ok",
    }
    return json.dumps(results, ensure_ascii=False, indent=3)


# Get latest kg from VMS
def get_latest_kg_from_vms(url, session, action="create"):
    responses = requests.get(url, verify=False).json()
    for response in responses:
        site_id = response.get("site_id")
        print(site_id)
        kg_excel_list_editing = response.get("editing").get("kg")
        kg_excel_list_production = response.get("production").get("kg")
        print(kg_excel_list_editing)
        print(kg_excel_list_production)

        if kg_excel_list_editing:
            version = "editing"
            for i in range(len(kg_excel_list_editing)):
                kg_excel_info = kg_excel_list_editing[i]
                kg_excel_file_path = kg_excel_info.get("file_path")
                kg_excel_time = kg_excel_info.get("timecreated")
                file_name = kg_excel_file_path.split("_")[-1][:-5]
                query = f"match (n) where n.site_id='{site_id}' and n.mode='{version}' and n.file_name='{file_name}' return n"
                results = session.run(query).values()
                print(f'results for {query}: ', results)
                if not results:
                    try:
                        kg_processing(session, kg_excel_file_path, site_id, version, action)
                        logging.info(f'sucessfully build graph for site {site_id} in {version} version!')
                    except Exception as e:
                        logging.error(f'failed to build graph for site {site_id} in {version} version with the following reason: \n{e}')
        
        if kg_excel_list_production:
            version = "production"
            for i in range(len(kg_excel_list_production)):
                kg_excel_info = kg_excel_list_production[i]
                kg_excel_file_path = kg_excel_info.get("file_path")
                kg_excel_time = kg_excel_info.get("timecreated")
                file_name = kg_excel_file_path.split("_")[-1][:-5]
                query = f"match (n) where n.site_id='{site_id}' and n.mode='{version}' and n.file_name='{file_name}' return n"
                results = session.run(query).values()
                print(f'results for {query}: ', results)
                if not results:
                    try:
                        kg_processing(session, kg_excel_file_path, site_id, version, action)
                        logging.info(f'sucessfully build graph for site {site_id} in {version} version!')
                    except Exception as e:
                        logging.error(f'failed to build graph for site {site_id} in {version} version with the following reason: \n{e}')


# Main Program
if __name__ == '__main__':
    
    parser = get_parser()
    args = parser.parse_args()
    max_seq_len = args.max_seq_len
    batch_size = args.batch_size
    kms_knowledge_list_url = args.kms_knowledge_list_url
    
    # paddle.set_device('gpu')
    paddle.set_device('cpu')
    paddle.disable_static()

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

    os.system('neo4j start')
    # time.sleep(30)

    # create converter to convert zh-hk to zh-cn
    converter = opencc.OpenCC('t2s.json')
    converter_s2t = opencc.OpenCC('s2t.json')
    
    # retrieve faq topics for mall
    faq_topics_mall = []
    with open(Config.faq_topics_mall, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            faq_topics_mall.append(line.strip())

    # retrieve faq topics for estate
    faq_topics_estate = []
    with open(Config.faq_topics_estate, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            faq_topics_estate.append(line.strip())

    
    ########################################################################################################
    # LOAD MODEL
    ########################################################################################################
    senta = Taskflow("sentiment_analysis", device_id=-1)
    similarity_en = Taskflow("text_similarity", model="rocketqa-medium-cross-encoder", device_id=-1)
    similarity_zh = Taskflow("text_similarity", model="simbert-base-chinese", device_id=-1)

    # load nlu tokenizer and joint model for chinese and english
    nlu_transformer_zh = "ernie-3.0-xbase-zh"
    nlu_transformer_en = "ernie-2.0-base-en"
    nlu_transformer_ja = "ernie-m-base"

    nlu_zh_tokenizer = ErnieTokenizer.from_pretrained(nlu_transformer_zh)
    nlu_en_tokenizer = ErnieTokenizer.from_pretrained(nlu_transformer_en)
    # nlu_ja_tokenizer = ErnieMTokenizer.from_pretrained(nlu_transformer_ja)

    ### Mall ###
    intent_path_mall = os.path.join(Config.nlu_model_base_path, "intent_label_mall.txt")
    slot_path_mall = os.path.join(Config.nlu_model_base_path, "slot_label_mall.txt")
    intent2id_mall, id2intent_mall = load_dict(intent_path_mall)
    slot2id_mall, id2slot_mall = load_dict(slot_path_mall)

    ernie_zh = ErnieModel.from_pretrained(nlu_transformer_zh)
    nlu_model_zh_mall = JointModel(ernie_zh, len(slot2id_mall), len(intent2id_mall), dropout=0.1)
    nlu_model_zh_mall.load_dict(paddle.load(
        os.path.join(Config.nlu_model_ckpt_path, "mall_zh/best.pdparams")
    ))

    ernie_en = ErnieModel.from_pretrained(nlu_transformer_en)
    nlu_model_en_mall = JointModel(ernie_en, len(slot2id_mall), len(intent2id_mall), dropout=0.1)
    nlu_model_en_mall.load_dict(paddle.load(
        os.path.join(Config.nlu_model_ckpt_path, "mall_en/best.pdparams")
    ))

    """
    ernie_ja = ErnieMModel.from_pretrained(nlu_transformer_ja)
    nlu_model_ja_mall = JointModel_JA(ernie_ja, len(slot2id_mall), len(intent2id_mall), dropout=0.1)
    nlu_model_ja_mall.load_dict(paddle.load(
        os.path.join(Config.nlu_model_ckpt_path, "mall_ja/best.pdparams")
    ))"""

    ### Estate ###
    intent_path_estate = os.path.join(Config.nlu_model_base_path, 'intent_label_estate.txt')
    slot_path_estate = os.path.join(Config.nlu_model_base_path, 'slot_label_estate.txt')
    intent2id_estate, id2intent_estate = load_dict(intent_path_estate)
    slot2id_estate, id2slot_estate = load_dict(slot_path_estate)

    ernie_zh = ErnieModel.from_pretrained(nlu_transformer_zh)
    nlu_model_zh_estate = JointModel(ernie_zh, len(slot2id_estate), len(intent2id_estate), dropout=0.1)
    nlu_model_zh_estate.load_dict(paddle.load(
        os.path.join(Config.nlu_model_ckpt_path, "estate_zh/best.pdparams")
    ))

    ernie_en = ErnieModel.from_pretrained(nlu_transformer_en)
    nlu_model_en_estate = JointModel(ernie_en, len(slot2id_estate), len(intent2id_estate), dropout=0.1)
    nlu_model_en_estate.load_dict(paddle.load(
        os.path.join(Config.nlu_model_ckpt_path, "estate_en/best.pdparams")
    ))
    

    ########################################################################################################
    # Get latest Graphs
    ########################################################################################################
    testing_ok = False
    while not testing_ok:
        try:
            testing_query = f"match (n) where n.site_id='oc' and n.mode='editing' and n.file_name='mall' return n"
            results = session.run(testing_query).values()
            testing_ok = True
            print("neo4j testing ok.")
        except:
            logging.error("Exception occurred", exc_info=True)
            print("neo4j testing is not ok, try connecting neo4j again 10 seconds later.")
            os.system('neo4j start')
            time.sleep(10)

    try:
        get_latest_kg_from_vms(kms_knowledge_list_url, session)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
    
    ########################################################################################################

    app.run(
        host=Config.server,
        port=Config.port,
        debug=False,
        ssl_context=(
            Config.vabot_ssl_cert,
            Config.vabot_ssl_key
            )
        )
