import logging
from configuration import Config
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import warnings
import opencc
from preprocessing import node_synonyms_preprocess
warnings.filterwarnings("ignore")


converter = opencc.OpenCC('t2s.json')


'''
def update_node_labels(info_dict):

    new_label = info_dict.get('label')
    old_label = info_dict.get('old_label')
    _id = info_dict.get('id')

    match = 'match (n {id:"' + _id + '"}) '
    remove = 'remove n:{} '.format(old_label)
    _set = 'set n:{}'.format(new_label)
    query = match + remove + _set
    return query'''


def get_action(sheet):
    if not sheet:
        return None
    if len(sheet.split("_"))>1 or sheet in ["Navigation"]:
        return "merge"
    else:
        return "create"


labels = set()
with open(Config.kg_labels_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        if line not in labels:
            labels.add(line)


def create_node(info_dict):

    node_label = info_dict.get('label')

    node_property = '{'
    for k, v in info_dict.items():
        v = str(v).replace('"', '')
        if k not in ['action', 'label']:
            if k in ['synonyms']:
                node_property += '{0}:split("{1}", ", "), '.format(k, str(v))
            else:
                node_property += '{0}:"{1}", '.format(k, str(v))
    node_property = node_property[:-2]

    query = 'create (:{0} {1}'.format(node_label, node_property) + '})'
    return query


def merge_node(info_dict, rel_dict):
    
    rel_type = rel_dict.get('rel_type')
    rel_obj_name = rel_dict.get('rel_obj_name')
    rel_obj_id = rel_dict.get('rel_obj_id')
    method_hk = rel_dict.get('method')
    method_sim = rel_dict.get('method_sim')
    method_eng = rel_dict.get('method_eng')

    id_1 = info_dict.get("id")
    id_2 = rel_obj_id
    mode = info_dict.get("mode")
    site_id = info_dict.get("site_id")

    node_1 = f'match (n1) where n1.id="{id_1}" and ' \
             f'n1.mode="{mode}" and ' \
             f'n1.site_id="{site_id}" '

    node_2 = f'match (n2) where n2.id="{rel_obj_id}" and ' \
             f'n2.mode="{mode}" and ' \
             f'n2.site_id="{site_id}" '

    if method_hk and method_sim and method_eng:
        rel_attr = '{'
        for i in ['method', 'method_sim', 'method_eng']:
            rel_attr += '{0}:"{1}", '.format(i, rel_dict.get(i))
        rel_attr = rel_attr[:-2]
        rel_attr += '}'
        relation = 'merge (n1)-[:{0} {1}]->(n2)'.format(rel_type, rel_attr)
    else:
        relation = 'merge (n1)-[:{0}]->(n2)'.format(rel_type)

    return node_1 + node_2 + relation


def update_node_properties(info_dict):
    _id = info_dict.get("id")
    mode = info_dict.get("mode")
    site_id = info_dict.get("site_id")

    # each node has unique combination of site_id, mode, id
    match = f'match (n) where n.id="{_id}" and ' \
            f'n.mode="{mode}" and ' \
            f'n.site_id="{site_id}" '

    # delete all properties first!!!!
    remove = 'set n = {} '

    node_attr = ''
    for k, v in info_dict.items():
        if k not in ['label', 'action']:
            if k in ['synonyms']:
                node_attr += 'set n.{0}=split("{1}", ", ") '.format(k, str(v))
            else:
                node_attr += 'set n.{0}="{1}" '.format(k, str(v))

    return match + remove + node_attr


def delete_node(info_dict):
    id = info_dict.get("id")
    mode = info_dict.get("mode")
    site_id = info_dict.get("site_id")

    # each node has unique combination of site_id, mode, id
    match = f'match (n) where n.id="{_id}" and ' \
            f'n.mode="{mode}" and ' \
            f'n.site_id="{site_id}" '

    query = match + 'detach delete n'
    return query


def delete_relationship(info_dict, rel_dict):

    rel_type = rel_dict.get('rel_type')
    rel_obj_name = rel_dict.get('rel_obj_name')
    rel_obj_id = rel_dict.get('rel_obj_id')

    id_1 = info_dict.get("id")
    id_2 = rel_obj_id
    mode = info_dict.get("mode")
    site_id = info_dict.get("site_id")

    node_1 = '(n1)'
    relation = f'-[r:{rel_type}]->'
    node_2 = '(n2) '

    where = f'where n1.id="{id_1}") and n1.mode="{mode}" and n1.site_id="{site_id}" and' \
            f'n2.id="{rel_obj_id}" and n2.mode="{mode}" and n2.site_id="{site_id}" '

    query = 'match ' + node_1 + relation + node_2 + where + 'delete r'
    return query


def write_data(session, file_path, sheet_name=None, version=None, site_id=None):

    file_name = file_path.split("_")[-1][:-5]
    
    for label in labels:
        # create constraint
        constraint_name = f"constraint_uniqiness_{label}"
        session.run(f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.key IS UNIQUE")
    
    assert sheet_name, "sheet name should be provided!" 
    
    ###################################################
    # 1. read dataframe
    # 2. replace nan with None
    # 3. filter dataframe in which "in_graph" == "no"
    # 4. reset index
    ###################################################

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        return None
    
    df = df.replace({np.nan: None})
    # df = df[df["action"] != "no action"]

    df = df.reset_index()
    rows = df.shape[0]
    df_columns = df.columns

    assert(r in df_columns for r in ['id', 'name'])

    # iterate each row in dataframe
    for i in range(rows):

        # -----------------start retrieving information--------------------------- #

        info = dict()

        for col_name in ['id', 'name', 'label', 'name_sim', 'name_eng', 'synonyms']:
            if col_name in df.columns:
                if col_name == "synonyms":
                    is_none = pd.isnull(df.loc[i, col_name])
                    if not is_none:
                        info[col_name] = str(df.loc[i, col_name])
                else:
                    info[col_name] = str(df.loc[i, col_name])
            
            if version:
                info['mode'] = str(version)
            
            if site_id:
                info['site_id'] = str(site_id)
        
        info['action'] = str(get_action(sheet=sheet_name))
        
        name_cantonese = info.get("name")
        if name_cantonese:
            name_cantonese_t2s = converter.convert(name_cantonese)
        name_sim = info.get("name_sim")
        name_eng = info.get("name_eng")
        if name_eng:
            name_eng_lower = name_eng.lower()

        if name_cantonese and name_sim and name_eng:
            additional_synonyms = ", ".join([name_cantonese, name_cantonese_t2s, name_sim, name_eng, name_eng_lower])
            if info.get("synonyms"):
                info["synonyms"] = info.get("synonyms") + ", " + additional_synonyms
                info["synonyms"] = list(set(info["synonyms"].split(", ")))
                info["synonyms"] = ", ".join(info["synonyms"])
            else:
                # info["synonyms"] = additional_synonyms
                info["synonyms"] = ", ".join(list(set(additional_synonyms.split(", "))))
        
        # if synonnyms --> do some preprocessing to this!!!!
        synonyms = info.get("synonyms")
        if synonyms:
            preprocessed_synonyms = [node_synonyms_preprocess(i) for i in synonyms.split(", ")]
            preprocessed_synonyms = list(set(preprocessed_synonyms))
            preprocessed_synonyms = ", ".join(preprocessed_synonyms)

            info["synonyms"] = synonyms + ", " + preprocessed_synonyms
            info["synonyms"] = ", ".join(list(set(info["synonyms"].split(", "))))
            # info["synonyms"] = list(set(info["synonyms"].split(", ")))

        info["key"] = info.get("site_id")+'_'+info.get("mode")+'_'+info.get("id")

        info["file_name"] = file_name
                
        # ---retrieve properties information--- #

        property_keys = []
        property_values = []
        k = 1
        property_key_col = "property_key_" + str(k)
        property_value_col = "property_value_" + str(k)
        while property_key_col in df_columns and property_value_col in df_columns:
            property_key = df.loc[i, property_key_col]
            property_value = df.loc[i, property_value_col]
            if property_key and property_value:
                property_keys.append(property_key)
                property_values.append(property_value)
            k += 1
            property_key_col = "property_key_" + str(k)
            property_value_col = "property_value_" + str(k)

        for k, v in zip(property_keys, property_values):
            info[k] = v

        # ---retrieve relationships information--- #

        rel_dict = dict()

        j = 1

        rel_type_col = "rel_type_" + str(j)
        rel_obj_name_col = "rel_obj_name_" + str(j)
        rel_obj_id_col = "rel_obj_id_" + str(j)
        method_hk_col = 'method_' + str(j)  # attribute
        method_sim_col = 'method_sim_' + str(j)  # attribute
        method_eng_col = 'method_eng_' + str(j)  # attribute
        # rel_attr_key_col = "rel_attr_key_" + str(j)
        # rel_attr_value_col = "rel_attr_value_" + str(j)

        while rel_type_col in df_columns and rel_obj_name_col in df_columns and rel_obj_id_col in df_columns:
            rel_type = df.loc[i, rel_type_col]
            rel_obj_name = df.loc[i, rel_obj_name_col]
            rel_obj_id = df.loc[i, rel_obj_id_col]

            if rel_type and rel_obj_name and rel_obj_id:
                rel_dict[str(j)] = dict()  # initialize dictionary
                rel_dict[str(j)]['rel_type'] = rel_type
                rel_dict[str(j)]['rel_obj_name'] = rel_obj_name
                rel_dict[str(j)]['rel_obj_id'] = rel_obj_id

                if method_hk_col in df_columns and method_sim_col in df_columns and method_eng_col in df_columns:
                    method_hk = df.loc[i, method_hk_col]
                    method_sim = df.loc[i, method_sim_col]
                    method_eng = df.loc[i, method_eng_col]

                    if method_hk and method_sim and method_eng:
                        rel_dict[str(j)]['method'] = method_hk
                        rel_dict[str(j)]['method_sim'] = method_sim
                        rel_dict[str(j)]['method_eng'] = method_eng

            j += 1

            rel_type_col = "rel_type_" + str(j)
            rel_obj_name_col = "rel_obj_name_" + str(j)
            rel_obj_id_col = "rel_obj_id_" + str(j)
            method_hk_col = 'method_' + str(j)
            method_sim_col = 'method_sim_' + str(j)
            method_eng_col = 'method_eng_' + str(j)
            # rel_attr_key_col = "rel_attr_key_" + str(j)
            # rel_attr_value_col = "rel_attr_value_" + str(j)

        # -----------------end of retrieving information--------------------------- #

        # -----------------start adding information--------------------------- #

        action = info["action"]

        # add information to graph through Cypher query

        if action == "create":
            query_create_node = create_node(info_dict=info)
            session.run(query_create_node)
            for x in range(len(rel_dict)):
                query_merge_node = merge_node(info_dict=info, rel_dict=rel_dict[str(x + 1)])
                session.run(query_merge_node)

        elif action == "merge":
            for x in range(len(rel_dict)):
                query_merge_node = merge_node(info_dict=info, rel_dict=rel_dict[str(x+1)])
                session.run(query_merge_node)

        elif action == "un-merge":
            for x in range(len(rel_dict)):
                query_delete_rel = delete_relationship(info_dict=info, rel_dict=rel_dict[str(x+1)])
                session.run(query_delete_rel)

        elif action == "update":
            query_update_node_attribute = update_node_properties(info_dict=info)
            session.run(query_update_node_attribute)

        elif action == "delete":
            query_delete_nodes = delete_node(info_dict=info)
            session.run(query_delete_nodes)

