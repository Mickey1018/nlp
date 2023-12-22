from push2neo4j import push_data, copy_previous_file
from push2neo4j import determine_mall_or_estate
from push2neo4j import add_site_to_malls
from push2neo4j import add_site_to_estates
from push2neo4j import remove_site_from_malls
from push2neo4j import remove_site_from_estates
import os, re
from re import search
import shutil
from neo4j import GraphDatabase
import pandas as pd
import collections
import math


def validate_id_column(excel_file):
    print("validating if id in column")
    valid = True
    sheets = pd.read_excel(excel_file, sheet_name=None)
    for sheet in sheets:
        if sheet != "Navigation" and not search('_', sheet):
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet)
                'id' in df.columns.tolist()
            except:
                valid = False
                break
    print("valid: ", valid)
    return valid


def validate_graph(excel_file):
    print("validing duplicated ids")
    validate = True
    sheets = pd.read_excel(excel_file, sheet_name=None)
    for sheet in sheets:
        if sheet != "Navigation" and not search('_', sheet):
            df = pd.read_excel(excel_file, sheet_name=sheet)
            try:
                df = df[df.id.notnull()]
            except:
                continue
            print("successfully filter dataframe")
            ids = df["id"].to_list()
            print("ids: ", ids)
            duplicated_ids = [item for item, count in collections.Counter(ids).items() if count > 1]
            if duplicated_ids:
                validate = False
                return validate, sheet, duplicated_ids
    return validate, None, None


def get_nlp_path(file_path, site_id, version):
    base_dir = './nfs-data/'
    excel_file_name = file_path.split('/')[-1]
    save_dir = os.path.join(base_dir, 'nlp', 'graphs', site_id, version)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return os.path.join(save_dir, excel_file_name)


def rebuild_source_path(file_path):
    print("in rebuild source path")
    print(file_path)
    pattern = re.compile(r"(.*)(/nfs-data/.*)$")
    output_path = pattern.sub(".\\2", file_path)
    print(output_path)
    return output_path


def kg_processing(session, file_path, site_id, version, action, prev_file_path=None, history=None):
    
    file_name = file_path.split("_")[-1][:-5]
    if prev_file_path:
        prev_file_name = prev_file_path.split("_")[-1][:-5]
        prev_file_path = rebuild_source_path(prev_file_path)
    file_path = rebuild_source_path(file_path)
    site_type = determine_mall_or_estate(file_path)
    
    if not site_type:
        return "cannot read excel file"
    
    has_id_column = validate_id_column(file_path)
    if not has_id_column:
        return f"column 'id' should be exist in excel file!!"
    
    validate, sheet, duplicated_ids = validate_graph(file_path)
    if not validate:
        duplicated_ids = ", ".join(duplicated_ids)
        return f"In {sheet}, id(s) {duplicated_ids} are duplicated"
    
    ##################################################Create#############################################################
    if action == "create":
        try:
            # push data
            push_data(session, file_path, site_id, version)
            
            # copy file from vms to nlp
            save_path = get_nlp_path(file_path, site_id, version)
            shutil.copyfile(file_path, save_path)
            print(f"copied file to {save_path}!")
            
            # add site_id to text file
            if site_type == "mall":
                add_site_to_malls(site_id)
            elif site_type == "estate":
                add_site_to_estates(site_id)

            return None
        
        except Exception as e:
            print("Error: ", e)
            try:
                
                # delete the nodes created
                session.run(f"match (n) where n.site_id='{site_id}' and n.mode='{version}' and n.file_name='{file_name}' detach delete n")

                # query graph to see there are any nodes, if so, do not remove site_id from text file
                # otherwise, remove site_id from text file
                check_exist_query = f"match (n) where n.site_id='{site_id}' return n"
                check_exist_result = session.run(check_exist_query).values()
                if not check_exist_result:
                    if site_type == "mall":
                        remove_site_from_malls(site_id)
                    elif site_type == "estate":
                        remove_site_from_estates(site_id)
                
                # use previous latest graph
                # if history:
                #     # get latest file
                #     history_path_list = []
                #     for record in history:
                #         temp_full_path = record.get("file_path")  # "/opt/nfs-data/graphs/filename.xlsx"
                #         temp_path = rebuild_source_path(temp_full_path)  # "./nfs-data/graphs/filename.xlsx"
                #         history_path_list.append(temp_path)  # ["./nfs-data/graphs/filename.xlsx", ...]
                #     
                #     latest_history_path = None
                #     if history_path_list:
                #         latest_history_path = max(history_path_list, key=os.path.getctime)
                #     
                #     if latest_history_path:
                #         pattern = re.compile(r"(.*)(\.xlsx)$")
                #         output_path = pattern.sub("\\1_copy\\2", latest_history_path)
                #         copy_latest_history_path = copy_previous_file(latest_history_path, site_id, output_path)
                #     
                #         print(f"start pushing data from {copy_latest_history_path}")
                #         push_data(session, copy_latest_history_path, site_id, version)
                #         print(f"finish pushing data from {copy_latest_history_path}")
                # 
                #         os.remove(copy_latest_history_path)
                #         print(f"remove {copy_latest_history_path}")
                # 
            except Exception as e:
                return e
            
            return e

    ##################################################Update#############################################################
    elif action == "update":

        # update graph with updated excel file        
        try:
            # 1. update site_id
            if site_type == "mall":
                add_site_to_malls(site_id)
            elif site_type == "estate":
                add_site_to_estates(site_id)
            
            # copy excel file first
            # output_path = get_nlp_path(file_path, site_id, version)
            # pattern = re.compile(r"(.*)(\.xlsx)$")
            # output_path = pattern.sub("\\1_copy\\2", output_path)
            # copy_file_path = copy_previous_file(file_path, site_id, output_path)

            # 2. delete the graph with previous file name 
            session.run(f"match (n) where n.site_id='{site_id}' and n.mode='{version}' and n.file_name='{prev_file_name}' detach delete n")

            # 3. create graph with new file name
            push_data(session, file_path, site_id, version)

            # # remove the copied excel file
            # os.remove(copy_file_path)
            
            # 4. copy from vms to nlp
            save_path = get_nlp_path(file_path, site_id, version)
            shutil.copyfile(file_path, save_path)
            print(f"copied file to {save_path}!")

            # 5. remove previous excel file
            prev_path = get_nlp_path(prev_file_path, site_id, version)
            if os.path.exists(prev_path):
                os.remove(prev_path)
                print(f"removed file {prev_path}!")

            return None
        
        except Exception as e:
            # if update is not successful, reuse the previous excel to build graph
            try:
                # copy previous excel file
                # output_path = get_nlp_path(prev_file_path, site_id, version)
                # pattern = re.compile(r"(.*)(\.xlsx)$")
                # output_path = pattern.sub("\\1_copy\\2", output_path)
                # copy_prev_file_path = copy_previous_file(prev_file_path, site_id, output_path)
    
                # 1. delete the new graph 
                session.run(f"match (n) where n.site_id='{site_id}' and n.mode='{version}' and n.file_name='{file_name}' detach delete n")

                # 2. also delete the previous graph
                session.run(f"match (n) where n.site_id='{site_id}' and n.mode='{version}' and n.file_name='{prev_file_name}' detach delete n")
                
                # 3. rebuild graph from the previous excel file
                push_data(session, prev_file_path, site_id, version)
    
                # remove copied file
                # os.remove(copy_prev_file_path)
            
            except Exception as e:
                return e
            
            return e
        
    ##################################################Delete#############################################################
    elif action == "delete":
        try:
            
            # delete graph with matched site_id, version, file_name
            delete_query = f"match (n) where n.site_id='{site_id}' and n.mode='{version}' and n.file_name='{file_name}' detach delete n"
            session.run(delete_query)
            print(f"Graph with site id {site_id} in {version} version with file name {file_name} is deleted!")

            # query graph to see there are any nodes, if so, do not remove site_id from text file
            # otherwise, remove site_id from text file
            check_exist_query = f"match (n) where n.site_id='{site_id}' return n"
            check_exist_result = session.run(check_exist_query).values()
            if not check_exist_result:
                if site_type == "mall":
                    remove_site_from_malls(site_id)
                elif site_type == "estate":
                    remove_site_from_estates(site_id)

            # delete file in nfs-data/nlp
            path_to_delete = get_nlp_path(file_path, site_id, version)
            if os.path.exists(path_to_delete):
                os.remove(path_to_delete)
                print("path {path_to_delete} remove!")
            
            # if history:
            #     # get latest file
            #     history_path_list = []
            #     for record in history:
            #         temp_full_path = record.get("file_path")  # "/opt/nfs-data/graphs/filename.xlsx"
            #         temp_path = rebuild_source_path(temp_full_path)  # "./nfs-data/graphs/filename.xlsx"
            #         history_path_list.append(temp_path)  # ["./nfs-data/graphs/filename.xlsx", ...]
            #     
            #     latest_history_path = None
            #     if history_path_list:
            #         latest_history_path = max(history_path_list, key=os.path.getctime)
 
            #     # create graph
            #     if latest_history_path:
            #         pattern = re.compile(r"(.*)(\.xlsx)$")
            #         output_path = pattern.sub("\\1_copy\\2", latest_history_path)
            #         copy_latest_history_path = copy_previous_file(latest_history_path, site_id, output_path)
            #         
            #         print(f"start pushing data from {copy_latest_history_path}")
            #         push_data(session, copy_latest_history_path, site_id, version)
            #         print(f"finish pushing data from {copy_latest_history_path}")
 
            #         os.remove(copy_latest_history_path)
            #         print(f"remove {copy_latest_history_path}")

            return None

        except Exception as e:
            return e


if __name__ == "__main__":
    print(validate_graph("./data/kg/sino_mall.xlsx"))
    '''
    try:
        # Connect to Graph Database
        driver = GraphDatabase.driver("bolt://0.0.0.0:7688", auth=('neo4j', 'sino'))
        session = driver.session()
        site_id = "oc"
        version = "editing"
        file_path = "/opt/nfs-data/graphs/sino_mall_test.xlsx"
        prev_file_path = "/opt/nfs-data/graphs/sino_mall_test_prev.xlsx"
        action = "update"
        kg_processing(session, file_path, site_id, version, action, prev_file_path)

    except Exception as e:
        print("bug: ", e)
    '''
