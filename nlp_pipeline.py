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

from nlu_data import make_dataset, create_intent_label, create_slot_label
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

# Check data
@app.route('/api/get_intent_and_keyword_type', methods=['GET', 'POST'])
def get_intent_and_keyword_type():
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
    
    excel_path = os.path.join(project_path, 'excel', 'dataset.xlsx')
    intent_output_path = os.path.join(project_path, 'intent', 'intent_label.txt')
    slot_output_path = os.path.join(project_path, 'keyword', 'slot_label.txt')

    try:
        create_intent_label(excel_path, output_path=intent_output_path)
        create_slot_label(excel_path, output_path=slot_output_path)

    except Exception as e:
        results = {
                "success": False,
                "error": {
                    "reason": str(e)
                    }
            }
        return json.dumps(results, ensure_ascii=False, indent=3)
    
    # return API result
    results = {
        "success": True,
        "data": f"Success! Updated for paths {intent_output_path} and {slot_output_path}"
    }
    return json.dumps(results, ensure_ascii=False, indent=3)


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
            num_epoch=50, 
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


# Health Check
@app.route('/healthy', methods=['GET'])
def check_health():
    results = {
        "status": "ok",
    }
    return json.dumps(results, ensure_ascii=False, indent=3)


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
    
    ########################################################################################################
    # LOAD MODEL
    ########################################################################################################

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

    app.run(
        host=Config.server,
        port=Config.port,
        debug=False
        # ssl_context=(
        #     Config.vabot_ssl_cert,
        #     Config.vabot_ssl_key
        #     )
        )
