from datetime import datetime
from pprint import pprint
import logging
from logging.handlers import RotatingFileHandler
import string, time
import collections
import sys, requests, os, json, argparse, random
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
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

from nlu_multilabel_train import multilable_classification_train
from nlu_multilabel_infer import multilable_classification_infer
from utils import has_child_directories, list_child_directories
from configuration import Config
from segmentation import segmentate_sentence, extract_key_word
from preprocessing import preprocess, download_nltk
# from nlu_ner_train import JointModel_M
from nlu_train import JointModel_M
from nlu_infer import ErnieTokenizer, ErnieModel, JointModel, load_dict
# from nlu_ner_infer import nlu_predict, nlu_predict_m
from nlu_infer import nlu_predict, nlu_predict_m
from kg_qa import get_answer_from_mall_kg
from kg_qa_estate import get_answer_from_estate_kg
from similarity_infer import get_answer_from_faq
from postprocessing import postprocess
from faq_processing import faq_processing
# from kg_processing import kg_processing
# from publish import publish_graph, publish_faq, delete_faq
from publish import publish_faq, delete_faq
# from push2neo4j import push_data

from nlu_data import make_dataset
from nlu_train import nlu_model_train

logger = logging.getLogger("Rotating Log")
logger.setLevel(logging.INFO)
logging.basicConfig(filename='./log/nlp.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
handler = RotatingFileHandler("./log/nlp.log", maxBytes=1000000, backupCount=5)
logger.addHandler(handler)

app = Flask(__name__)
server = Config.server
port = Config.port

# initialize executor
executor = ThreadPoolExecutor(2)

converter = opencc.OpenCC('t2s.json')
converter_s2t = opencc.OpenCC('s2t.json')

"""
def get_parser():
    parser = argparse.ArgumentParser()
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

    # {
    #     entity_type1: [e1, e2, ...], 
    #     entity_type2: [d1, d2, ...]
    # }

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
        return 'faq'
    intent = intent.lower()
    if intent in ["chat", "chat_greeting", "chat_violent"]:
        return 'small_talk'
    # elif intent in faq_topics:
    #     return 'faq'
    else:
        return 'faq'
"""

def nlu(input_text, project):

    intent_model = proj_intent_model.get(project)
    intent_tokenizer = proj_intent_tokenizer.get(project)
    ner_model = proj_ner_model.get(project)
    ner_tokenizer = proj_ner_tokenizer.get(project)
    id2intent = proj_id2intent.get(project)
    id2slot = proj_id2slot.get(project)

    # part 1: intents (multiple)
    logger.info('start inferring intent...')
    all_intent_results = multilable_classification_infer(
        intent_model, 
        intent_tokenizer,
        id2intent,
        device='cpu', 
        max_seq_length=1024, 
        batch_size=16, 
        excel_dir='excel', 
        excel_file='dataset.xlsx', 
        use_batch=False,
        text=input_text
    )
    # assume process with one text only
    intent_result = all_intent_results[0]  # take the first result from batch

    # part 2: keywords
    logger.info('start inferring keyword...')
    _, slot_result = nlu_predict_m(
        input_text, 
        ner_model, 
        ner_tokenizer, 
        id2intent, 
        id2slot, 
        converter=converter
    )
    # elif lang in ["zh-hk", "zh-cn"]:
    #     intent_labels, slots = nlu_predict(input_text, nlu_model_zh_mall, nlu_zh_tokenizer, id2intent, id2slot, converter)        
    # elif lang in ["en"]:
    #     intent_labels, slots = nlu_predict(input_text, nlu_model_en_mall, nlu_en_tokenizer, id2intent, id2slot, converter)
    
    return intent_result, slot_result
    # return intent, intent_prob, slots

"""
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
    return df
"""

@app.route('/api/classify_email_and_get_keywords', methods=['GET', 'POST'])
def get_topic_and_keywords():

    ####################################### Validation Start ##############################################
    # get json contents
    try:
        content = request.json
        logger.info('Inference Input:')
        logger.info(content)
    except:
        results = {
            "success": False,
            "error": {
                "reason": "invalid json format" 
            }
        }
        return json.dumps(results, ensure_ascii=False, indent=3)

    try:
        project = content["project"]
        logger.info(project)
        subject = content["subject"]
        logger.info(subject)
        content = content["content"]
        logger.info(content)
        logger.info(subject)
        text = subject + content
        logger.info(text)
    except Exception as e:
        results = {
            "success": False,
            "error": {
                "reason": str(e) 
            }
        }
        return json.dumps(results, ensure_ascii=False, indent=3)
    
    # check data type
    if not isinstance(text, str):
        results = {
            "success": False,
            "error": {
                "reason": "language, text, version, site_id, history must in valid data type" 
            }
        }
        return json.dumps(results, ensure_ascii=False, indent=3)
    ########################################## Validation End ##############################################
    
    ########################################## Check training Start ########################################
    """
    training status
    ["in training", "training finished", "training was not successful"]
    """
    model_is_in_training = False
    # get all job's status
    # jobs = {1:'in training', 2:'in training'}
    training_ids = jobs.keys()
    if training_ids:
        for training_id in training_ids:
            status = jobs[training_id].get("status")
            if status == 0:
                model_is_in_training = True
                break
    
    if model_is_in_training:
        logger.info('There is training job right now, cannot do inferring')
        results = {
            "success": False,
            "error": {
                "reason": "NLP model is in training"
            }
        }
        return json.dumps(results, ensure_ascii=False, indent=3)

    ########################################## Check training End ##########################################

    ########################################## NLP start ###################################################
    try:
        # Preprocess the Question
        text = preprocess(text)
    
        # NLU
        intent_result, slot_result = nlu(text, project)
        logger.info("intent result:")
        logger.info(intent_result)
        logger.info("keyword result:")
        logger.info(slot_result)

        ########################################## NLP End #####################################################
        results = {
            "success": True, 
                "data": {
                    "category": intent_result,
                    "keywords": slot_result
                } 
            }

        results_json = json.dumps(results, ensure_ascii=False, indent=3)
        logger.info("API results:")
        logger.info(results_json)
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

"""
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
"""


# Train NLP modes
@app.route('/api/train_email_classifier', methods=['GET', 'POST'])
def train_nlp_models_asyn():
    try:
        content = request.json
        project_path = content.get("project_path")

    except Exception as e:
        results = {
                "success": False,
                "error": {
                    "reason": str(e) 
                    }
            }
        return json.dumps(results, ensure_ascii=False, indent=3)
    
    # use nlu_data.py to convert excel to train data and test data
    data_ok, err_message = make_dataset(project_path, split=True, delete_and_update=False)
    
    if not data_ok:
        results = {
            "success": False,
            "error": {
                "reason": err_message 
                }
        }
        json_results = json.dumps(results, ensure_ascii=False, indent=3)
        logger.info(json_results)
        return json_results

    logger.info("dataset is ok!")
    logger.info("now moving to model training section!")
    
    # generate training id
    training_id = None
    job_ids = jobs.keys()
    if not job_ids:
        training_id = "id-0001"
    if job_ids:
        job_ids = job_ids.sort()
        last_job_id = job_ids[-1]
        training_id = f"id-000{last_job_id+1}"
    msg = f'training id is {training_id}'
    logger.info(msg)

    # pass training job to thread
    executor.submit(job_train_nlu, training_id, project_path)

    # return API result
    results = {
        "success": True,
        "data": {
            "training_id": training_id
        }
    }

    return json.dumps(results, ensure_ascii=False, indent=3)


def job_train_nlu(training_id, project_path):
    jobs[training_id] = {}
    # update job status: 0-in training; 1-finish training 
    jobs[training_id]["status"] = 0
    now = datetime.now()
    logger.info(now)
    jobs[training_id]["training_start"] = now.strftime("%d/%m/%Y %H:%M:%S")
    logger.info(jobs)

    # use nlu_train.py to fine-tune nlp model
    train_success = True

    try:
        # train intent model
        logger.info('start training for intent model')
        result_intent = multilable_classification_train(
            training_id, 
            project_path, 
            batch_size=1, 
            epochs=5,
            logging_steps=5, 
        )
        jobs[training_id]["result_intent"] = result_intent
        logger.info('end training for intent model')

        # train ner model
        logger.info('start training for ner model')
        result_ner = nlu_model_train(
            logger,
            project_path, 
            lang=None, 
            num_epoch=1, 
            batch_size=4, 
            max_seq_len=512, 
            learning_rate=3e-5,
            weight_decay=0.01, 
            warmup_proportion=0.1, 
            max_grad_norm=1.0,
            log_step=50, 
            eval_step=100
        )
        jobs[training_id]["result_ner"] = result_ner
        logger.info('end training for ner model')


    except Exception as e:
        # update job status with error message
        jobs[training_id] = str(e)
        
        train_success = False

        results = {
            "success": False,
            "error": {
                "reason": str(e) 
                }
        }
        json_results = json.dumps(results, ensure_ascii=False, indent=3)
        logger.info("Training Not success!")
        logger.info(json_results)


    if train_success:
        # update job status: 0-in training; 1-finish training 
        jobs[training_id]["status"] = 1
        now = datetime.now()
        jobs[training_id]["training_end"] = now.strftime("%d/%m/%Y %H:%M:%S")
        logger.info("Training success!")


@app.route('/api/get_training_job_status', methods=['GET', 'POST'])
def get_training_job_status():
    """
    training status
    ["in training", "training finished", "training was not successful"]
    """
    try:
        content = request.json
        training_id = content.get("training_id")

    except Exception as e:
        results = {
            "success": False,
            "error": {
                "reason": str(e) 
            }
        }
        return json.dumps(results, ensure_ascii=False, indent=3)
    
    try:
        status = jobs[training_id].get("status")
        training_start = jobs[training_id].get("training_start")
        training_end = jobs[training_id].get("training_end")
        training_result_ner = jobs[training_id].get("result_ner")
        training_result_intent = jobs[training_id].get("result_intent")

        results = {
            "success": True,
            "data": {
                "training_id": training_id,
                "status": status,
                "training_start": training_start,
                "training_end": training_end,
                "result": {
                    "intent": training_result_intent,
                    "keyword": training_result_ner
                },
                "reason": [
                    "insufficient training data",
                    "number of intents out of scope"
                ]
            }
        }
        json_results = json.dumps(results, ensure_ascii=False, indent=3)
        logger.info(json_results)
        return json_results

    except Exception as e:
        results = {
            "success": False,
            "training_id": training_id,
            "error": {
                "reason": str(e) 
            }
        }
        json_results = json.dumps(results, ensure_ascii=False, indent=3)
        logger.info(json_results)
        return json_results

"""
# Train NLP modes
@app.route('/api/train_email_classifier_not_async', methods=['GET', 'POST'])
def train_nlp_models():
    try:
        content = request.json
        project_path = content.get("project_path")

    except Exception as e:
        results = {
                "success": False,
                "error": {
                    "reason": str(e) 
                    }
            }
        return json.dumps(results, ensure_ascii=False, indent=3)
    
    # use nlu_data.py to convert excel to train data and test data
    data_ok, err_message = make_dataset(project_path, split=True, delete_and_update=False)
    
    if not data_ok:
        results = {
                "success": False,
                "error": {
                    "reason": err_message 
                    }
            }
        json_results = json.dumps(results, ensure_ascii=False, indent=3)
        logger.info(json_results)
        return json_results

    logger.info("dataset is ok!")
    logger.info("now moving to model training section!")

    # use nlu_train.py to fine-tune nlp model
    train_success = True

    try:
        nlu_model_train(
            project_path, 
            lang=None, 
            num_epoch=50, 
            batch_size=16, 
            max_seq_len=512, 
            learning_rate=3e-5,
            weight_decay=0.01, 
            warmup_proportion=0.1, 
            max_grad_norm=1.0, 
            seed=666, 
            log_step=50, 
            eval_step=100, 
            use_gpu=True
        )

    except Exception as e:
        train_success = False
        results = {
                "success": False,
                "error": {
                    "reason": str(e)
                    }
            }
        json_results = json.dumps(results, ensure_ascii=False, indent=3)
        logger.info("Training Not success!")
        logger.info(json_results)
        return json_results

    if train_success:
        logger.info("Training success!")
        results = {
                "success": True
            }
        return json.dumps(results, ensure_ascii=False, indent=3)

"""
# Health Check
@app.route('/healthy', methods=['GET'])
def check_health():
    results = {
        "status": "ok",
    }
    return json.dumps(results, ensure_ascii=False, indent=3)


"""
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
"""

# Main Program
if __name__ == '__main__':

    jobs = {}

    nlu_transformer = "ernie-m-base"
    nlu_tokenizer = ErnieMTokenizer.from_pretrained(nlu_transformer)
    ernie_m = ErnieMModel.from_pretrained(nlu_transformer)

    proj_intent_path = {}
    proj_keyword_path = {}
    proj_intent2id = {}
    proj_id2intent = {}
    proj_slot2id = {}
    proj_id2slot = {}
    proj_model = {}
    proj_intent_model = {}
    proj_intent_tokenizer = {}
    proj_ner_model = {}
    proj_ner_tokenizer = {}

    desired_dirs = ['excel', 'intent', 'keyword', 'train', 'dev', 'ckpt_intent']
    parent_dir = "data/project"
    # get all projects
    dirs = list_child_directories(parent_dir)
    # transform to data/project/my_project 
    dirs = [f"{parent_dir}/{i}" for i in dirs]  # ["data/immd", "data/sth"]
    logger.info(dirs)
    # iterate each project
    for dir in dirs:
        # if the project contains the following folders: 'excel', 'intent', 'keyword', 'train', 'dev', 'ckpt'
        if has_child_directories(dir, desired_dirs):
            # get project name
            proj = dir.split("/")[-1]
            
            # get proejct intent path
            intent_path = os.path.join(parent_dir, proj, "intent", "intent_label.txt")
            proj_intent_path[proj] = intent_path
            proj_intent2id[proj], proj_id2intent[proj] = load_dict(intent_path)
            
            # get project slot path
            slot_path = os.path.join(parent_dir, proj, "keyword", "slot_label.txt")
            proj_keyword_path[proj] = slot_path
            proj_slot2id[proj], proj_id2slot[proj] = load_dict(slot_path)
            
            # get project ckpt dir ---> NER
            ckpt_ner_path = os.path.join(parent_dir, proj, "ckpt", "best.pdparams")
            if os.path.exists(ckpt_ner_path):
                temp_ner_model = JointModel_M(ernie_m, len(proj_slot2id[proj]), len(proj_intent2id[proj]), dropout=0.1)
                temp_ner_model.load_dict(paddle.load(ckpt_ner_path))
                proj_ner_model[proj] = temp_ner_model
                proj_ner_tokenizer[proj] = nlu_tokenizer

            # get project ckpt dir ---> intent
            ckpt_intent_dir = os.path.join(parent_dir, proj, "ckpt_intent")
            if os.path.exists(ckpt_intent_dir):
                proj_intent_model[proj] = AutoModelForSequenceClassification.from_pretrained(ckpt_intent_dir)
                proj_intent_tokenizer [proj] = AutoTokenizer.from_pretrained(ckpt_intent_dir)

    logger.info(proj_intent_path)
    logger.info(proj_keyword_path)
    logger.info(proj_intent2id)
    logger.info(proj_slot2id)

    # parser = get_parser()
    # args = parser.parse_args()
    # kms_knowledge_list_url = args.kms_knowledge_list_url
    
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

    # os.system('neo4j start')

    # create converter to convert zh-hk to zh-cn
    converter = opencc.OpenCC('t2s.json')
    converter_s2t = opencc.OpenCC('s2t.json')
    
    # retrieve faq topics
    # faq_topics = []
    # with open(Config.faq_topics, "r", encoding="utf-8") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         faq_topics.append(line.strip())

    
    ########################################################################################################
    # LOAD MODEL
    ########################################################################################################
    # Simiarity Model
    # senta = Taskflow("sentiment_analysis", device_id=-1)
    # similarity_en = Taskflow("text_similarity", model="rocketqa-medium-cross-encoder", device_id=-1)
    # similarity_zh = Taskflow("text_similarity", model="simbert-base-chinese", device_id=-1)

    # NLU Model
    # load nlu tokenizer and joint model for chinese and english
    # nlu_transformer_zh = "ernie-3.0-xbase-zh"
    # nlu_transformer_en = "ernie-2.0-base-en"
    # nlu_transformer = "ernie-m-base"

    # nlu_zh_tokenizer = ErnieTokenizer.from_pretrained(nlu_transformer_zh)
    # nlu_en_tokenizer = ErnieTokenizer.from_pretrained(nlu_transformer_en)
    # nlu_tokenizer = ErnieMTokenizer.from_pretrained(nlu_transformer)

    # intent_path = Config.nlu_model_intent_label_path
    # slot_path = Config.nlu_model_slot_label_path
    # intent2id, id2intent = load_dict(intent_path)
    # slot2id, id2slot = load_dict(slot_path)

    # Initialize model
    # ernie_zh = ErnieModel.from_pretrained(nlu_transformer_zh)
    # nlu_model_zh_mall = JointModel(ernie_zh, len(slot2id_mall), len(intent2id_mall), dropout=0.1)
    # nlu_model_zh_mall.load_dict(paddle.load(Config.nlu_model_ckpt_path_zh))

    # ernie_en = ErnieModel.from_pretrained(nlu_transformer_en)
    # nlu_model_en_mall = JointModel(ernie_en, len(slot2id_mall), len(intent2id_mall), dropout=0.1)
    # nlu_model_en_mall.load_dict(paddle.load(Config.nlu_model_ckpt_path_en))
    
    # ernie = ErnieMModel.from_pretrained(nlu_transformer)
    # if os.path.exists(Config.nlu_model_ckpt_path_multi):
    #     nlu_model = JointModel_M(ernie, len(slot2id), len(intent2id), dropout=0.1)
    #     nlu_model.load_dict(paddle.load(Config.nlu_model_ckpt_path_multi))



    ########################################################################################################
    # Get latest Graphs
    ########################################################################################################
    """
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
    """
    
    ########################################################################################################

    app.run(
        host=Config.server,
        port=Config.port,
        debug=False
        # ssl_context=(
        #     Config.vabot_ssl_cert,
        #     Config.vabot_ssl_key
        #     )
        )
