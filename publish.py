import os, re
import shutil
from push2neo4j import push_data, copy_previous_file
import pandas as pd
from kg_processing import validate_graph, validate_id_column


def is_small_talk_file(source_file_path):
    df = pd.read_excel(source_file_path)
    print(df)
    topics = df.Topic.unique()
    print(topics)
    print(type(topics))
    if topics.tolist() == ['nothing', 'chat']:
        return True
    else:
        return False


def get_nlp_small_talk_path(file_path, site_id, version):
    """
    ./nfs-data/nlp/small_talks//editing/excelfile.xlsx
    """
    base_dir = './nfs-data/'
    excel_file_name = file_path.split('/')[-1]
    save_dir = os.path.join(base_dir, 'nlp', 'small_talks', site_id, version)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return os.path.join(save_dir, excel_file_name)


def get_nlp_path(file_path, site_id, version, kg_file_path=None, faq_file_path=None):
    base_dir = './nfs-data/'
    excel_file_name = file_path.split('/')[-1]
    if faq_file_path:
        save_dir = os.path.join(base_dir, 'nlp', 'faqs', site_id, version)
    elif kg_file_path:
        save_dir = os.path.join(base_dir, 'nlp', 'graphs', site_id, version)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return os.path.join(save_dir, excel_file_name)


def rebuild_source_path(file_path):
    pattern = re.compile(r"(.*)(/nfs-data/.*)$")
    output_path = pattern.sub(".\\2", file_path)
    return output_path


def get_newest_file(path):
    files = os.listdir(path)
    if files:
        paths = [os.path.join(path, basename) for basename in files]
    if paths:
        return max(paths, key=os.path.getctime)
    else:
        return None


def publish_graph(session, site_id, kg_file_path, in_use_file_list):

    version = "production"
    file_name = kg_file_path.split("_")[-1][:-5]

    try:
        # do validation
        kg_file_path = rebuild_source_path(kg_file_path)
        if not os.path.exists(kg_file_path):
            return f"cannot find {kg_file_path}"
        
        has_id_column = validate_id_column(kg_file_path)
        if not has_id_column:
            return f"column 'id' should be exist in excel file!!"

        validate, sheet, duplicated_ids = validate_graph(kg_file_path)
        if not validate:
            duplicated_ids = ", ".join(duplicated_ids)
            return f"In {sheet}, id(s) {duplicated_ids} are duplicated"
        print("valid!")
        
        # 1. delete all nodes that are not in use
        if in_use_file_list:
            in_use_files = ", ".join(in_use_file_list)
            session.run(f"match (n) where n.site_id='{site_id}' and n.mode='{version}' and NOT n.file_name in split('{in_use_files}', ', ') detach delete n")
        else:
            session.run(f"match (n) where n.site_id='{site_id}' and n.mode='{version}' detach delete n")
        print("graph deleted")
        
        # copy kg_file_path
        # output_path = get_nlp_path(kg_file_path, site_id, version, kg_file_path=kg_file_path, faq_file_path=None)
        # pattern = re.compile(r"(.*)(\.xlsx)$")
        # output_path = pattern.sub("\\1_copy\\2", output_path)
        # print(output_path)
        # copied_path = copy_previous_file(kg_file_path, site_id, output_path)
        
        # 2. build graph from copied kg_file_path
        push_data(session, kg_file_path, site_id, version)
        print(f"finish push data from {kg_file_path}")

        # delete copied kg_file_path
        # os.remove(copied_path)

        # 5. copy kg_file_path from vms to nlp
        # save_path = get_nlp_path(kg_file_path, site_id, version, kg_file_path=kg_file_path, faq_file_path=None)
        # shutil.copyfile(kg_file_path, save_path)
        # print(f"copied from {kg_file_path} to {save_path}")
        # # for loop --> delete all files accept the latest one
        # loop_dir = "/".join(save_path.split("/")[:-1])
        # latest_excel_file = save_path.split("/")[-1]
        # for f in os.listdir(loop_dir):
        #     if f != latest_excel_file:
        #         if os.path.isdir(f):
        #             shutil.rmtree(os.path.join(loop_dir, f))
        #             print(f"{f} delted!")
        #         else:
        #             os.remove(os.path.join(loop_dir, f))
        #             print(f'{f} deleted!')
    
    except Exception as e:
        print(e)
        try:
            # delete nodes for publish
            session.run(f"match (n) where n.site_id='{site_id}' and n.mode='{version}' and n.file_name='{file_name}' detach delete n")

            """
            # find the latest file
            kg_path = get_nlp_path(kg_file_path, site_id, version, kg_file_path=kg_file_path, faq_file_path=None)
            if not os.path.exists(kg_path):
                os.mkdir(kg_path)
            kg_dir = "/".join(kg_path.split("/")[:-1])
            latest_kg_file = get_newest_file(kg_dir)  # latest.xlsx
            
            if latest_kg_file:
                # copy latest file
                pattern = re.compile(r"(.*)(\.xlsx)$")
                output_path = pattern.sub("\\1_copy\\2", latest_kg_file)
                copied_path = copy_previous_file(kg_file_path, site_id, output_path)
                print(copied_path)
                # delete whole graph first
                session.run(f"match (n) where n.site_id='{site_id}' and n.mode='{version}' detach delete n")
                # build graph from copied kg_file_path
                print(f"start pushing data from {copied_path}")
                push_data(session, copied_path, site_id, version)
                print(f"finish pushing data from {copied_path}")
                # delete copied kg_file_path
                os.remove(copied_path)"""

        except Exception as e:
            return e
        
        return e
    

def publish_faq(session, site_id, faq_file_path):
    print("publishing faq or small talk")
    version = "production"
    try:
        faq_file_path = rebuild_source_path(faq_file_path)
        print("faq path given from vms: ", faq_file_path)
        is_small_talk = is_small_talk_file(faq_file_path)
        print("is small talk? ", is_small_talk)
        
        # 1. copy faq_file_path from vms to nlp
        if is_small_talk:
            save_path = get_nlp_small_talk_path(faq_file_path, site_id, version)
        else:
            save_path = get_nlp_path(faq_file_path, site_id, version, kg_file_path=None, faq_file_path=faq_file_path)
        print(save_path)
        shutil.copyfile(faq_file_path, save_path)
        
        # # for loop --> delete all files accept the latest one
        # loop_dir = "/".join(save_path.split("/")[:-1])
        # latest_excel_file = save_path.split("/")[-1]
        # for f in os.listdir(loop_dir):
        #     if f != latest_excel_file:
        #         os.remove(os.path.join(loop_dir, f))
        #         print(f'{f} deleted!')
    
    except Exception as e:
        return e


def delete_faq(faq_file_path):
    try:
        faq_file_path = rebuild_source_path(faq_file_path)
        is_small_talk = is_small_talk_file(faq_file_path)
        
        # 1. copy faq_file_path from vms to nlp
        if is_small_talk:
            delete_path = get_nlp_small_talk_path(faq_file_path, site_id, version)
        else:
            delete_path = get_nlp_path(faq_file_path, site_id, version, kg_file_path=None, faq_file_path=faq_file_path)
        
        # 2. delete file in production directory
        os.remove(delete_path)
    
    except Exception as e:
            return e


if __name__ == "__main__":
    faq_file_path = "/opt/nfs-data/faqs/production/abc.xlsx"
    save_path = get_nlp_path(faq_file_path, "oc", "production", kg_file_path=None, faq_file_path=faq_file_path)
    loop_dir = "/".join(save_path.split("/")[:-1])
    latest_excel_file = 'abc.xlsx'
    for f in os.listdir(loop_dir):
        if f != latest_excel_file:
            os.remove(os.path.join(loop_dir, f))
            print(f'{f} deleted!')