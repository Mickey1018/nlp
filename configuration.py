class Config:
    server = "0.0.0.0"
    port = 8080
    vabot_ssl_cert = './ss_vabot.crt'
    vabot_ssl_key = './ss_vabot.key'
    nlu_data_problem_path = './data/nlu/dataset_problems.txt'
    nlu_model_intent_label_path = './data/nlu/intent_label.txt'
    nlu_model_slot_label_path = './data/nlu/slot_label.txt'
    nlu_model_ckpt_path_zh = './ckpt/nlu/zh/best.pdparams'
    nlu_model_ckpt_path_en = './ckpt/nlu/en/best.pdparams'
    nlu_model_ckpt_path_multi = './ckpt/nlu/multi/best.pdparams'
    nlu_model_train_dir = './data/nlu/train'
    nlu_model_dev_dir = './data/nlu/dev'

    kg_base_path = './data/kg/'
    uri_svacs_kg01_bolt_ssc = "bolt+ssc://svacs-kg01.vabot.org:7687"
    uri_docker_bolt = "bolt://0.0.0.0:7687"
    uri_docker_bolt_ssc = "bolt+ssc://0.0.0.0:7687"  # 0.0.0.0:7687
    uri_docker_https = "https://0.0.0.0:7473"
    
    # neo4j_user_name = "neo4j"
    # neo4j_password = "sino"
    
    kg_labels_path = "./data/kg/labels.txt"
    
    site_id_malls = "./data/site_ids/malls.txt"
    site_id_estates = "./data/site_ids/estates.txt"
    
    faq_topics = "./data/faq_topics/faq_topics.txt"
    template_db_base_path = "./nfs-data/nlp/faqs"
    small_talks_base_path = "./nfs-data/nlp/small_talks"
    
    stopwords_zh_path = "./data/stopwords/stopwords_zh.txt"
    stopwords_en_path = "./data/stopwords/stopwords_en.txt"
    
    restaurant_shop_dictionary_base_path = "./data/restaurant_shop_dictionary/"


# Connect to Graph Database
# driver = GraphDatabase.driver(Config.uri_docker_bolt_ssc, auth=(Config.neo4j_user_name, Config.neo4j_password))
# driver = GraphDatabase.driver(Config.uri_svacs_kg01_bolt_ssc, auth=(Config.neo4j_user_name, Config.neo4j_password))
# session = driver.session()

