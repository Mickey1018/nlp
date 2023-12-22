# For eiditing version only

import pandas as pd
import opencc
import os
import shutil
import re

converter = opencc.OpenCC('t2s.json')
# df = df[~df.Question.isnull()].reset_index()


def is_small_talk_file(source_file_path):
    df = pd.read_excel(source_file_path, sheet_name=0)
    topics = df.Topic.unique()
    # if chat in topics.tolist() == ['nothing', 'chat']:
    if chat in topics.tolist():
        return True
    else:
        return False

# write to excel file
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


# write to excel file
def get_nlp_faq_path(file_path, site_id, version):
    """
    ./nfs-data/nlp/faqs/oc/editing/excelfile.xlsx
    """
    base_dir = './nfs-data/'
    excel_file_name = file_path.split('/')[-1]
    save_dir = os.path.join(base_dir, 'nlp', 'faqs', site_id, version)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return os.path.join(save_dir, excel_file_name)


def rebuild_source_path(file_path):
    pattern = re.compile(r"(.*)(/nfs-data/.*)$")
    output_path = pattern.sub(".\\2", file_path)
    return output_path


def faq_processing(file_path, site_id, version, action, prev_file_path=None):

    file_path = rebuild_source_path(file_path)
    is_small_talk = is_small_talk_file(file_path)

    if action == "create":
        try:
            if is_small_talk:
                save_path = get_nlp_small_talk_path(file_path, site_id, version)
            else:
                save_path = get_nlp_faq_path(file_path, site_id, version)
            # copy file from vms to nlp
            shutil.copyfile(file_path, save_path)
            print(f"copied file to {save_path}!")
            return None
        
        except Exception as e:
            return e
    

    elif action == "update":
        try:
            # copy file from vms to nlp
            if is_small_talk:
                save_path = get_nlp_small_talk_path(file_path, site_id, version)
            else:
                save_path = get_nlp_faq_path(file_path, site_id, version)
            shutil.copyfile(file_path, save_path)
            print(f"copied file to {save_path}!")
            
            # remove previous file
            if is_small_talk:
                prev_path = get_nlp_small_talk_path(prev_file_path, site_id, version)
            else:
                prev_path = get_nlp_faq_path(prev_file_path, site_id, version)
            if os.path.exists(prev_path):
                os.remove(prev_path)
                print(f"removed file {prev_path}!")

            return None

        except Exception as e:
            return e
    

    elif action == "delete":
        try:
            # delete file in nlp directory
            if is_small_talk:
                delete_path = get_nlp_small_talk_path(file_path, site_id, version)
            else:
                delete_path = get_nlp_faq_path(file_path, site_id, version)
            os.remove(delete_path)
            print(f"removed file {delete_path}!")
            return None

        except Exception as e:
            return e


if __name__ == "__main__":

    # file_path = "/opt/nfs-data/faqs/new_filename.xlsx"
    # site_id = "oc"
    # version = 'editing'
    # result = get_nlp_faq_path(file_path, site_id, version)
    # print(result)
    # path = rebuild_source_path("/opt/nfs-data/faqs/abc.xlsx")
    # print(path)
    print(is_small_talk_file("./nfs-data/nlp/small_talks/editing/small_talk.xlsx"))
    print(is_small_talk_file("./nfs-data/nlp/faqs/oc/editing/faq_template_oc.xlsx"))
