from neo4j_driver import write_data
from neo4j import GraphDatabase
import argparse
import pandas as pd
# from configuration import Config, session
from configuration import Config
import os, json
from preprocessing import node_synonyms_preprocess
import opencc

converter = opencc.OpenCC('t2s.json')

'''
driver = GraphDatabase.driver('bolt+ssc://zotac-05.vabot.org:7687', auth=(Config.neo4j_user_name, Config.neo4j_password))
session = driver.session()
'''

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site_id', type=str)
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--version', type=str)
    return parser


# retrieve malls and estates
def get_mall_estates():
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


def add_site_to_malls(site_id):
    malls, estates = get_mall_estates()
    if site_id not in malls:
        with open(Config.site_id_malls, "a", encoding="utf-8") as f:
            f.write(str(site_id)+'\n')

def add_site_to_estates(site_id):
    malls, estates = get_mall_estates()
    if site_id not in estates:
        with open(Config.site_id_estates, "a", encoding="utf-8") as f:
            f.write(str(site_id)+'\n')

def remove_site_from_malls(site_id):
    malls, estates = get_mall_estates()
    if site_id in malls:
        with open(Config.site_id_malls, "r", encoding="utf-8") as f:
            lines = f.readlines()
            with open(Config.site_id_malls, "w", encoding="utf-8") as f_w:
                for line in lines:
                    line = line.strip()
                    if line != site_id:
                        f_w.write(line+"\n")

def remove_site_from_estates(site_id):
    malls, estates = get_mall_estates()
    if site_id in estates:
        with open(Config.site_id_estates, "r", encoding="utf-8") as f:
            lines = f.readlines()
            with open(Config.site_id_estates, "w", encoding="utf-8") as f_w:
                for line in lines:
                    line = line.strip()
                    if line != site_id:
                        f_w.write(line+"\n")


def determine_mall_or_estate(file_path):
    print("in determine mall or estate function")
    print("open", file_path)
    if not os.path.exists(file_path):
        return False
    sheets = pd.read_excel(file_path, sheet_name=None)
    if 'Mall' in sheets:
        return "mall"
    elif 'Estate' in sheets:
        return "estate"


def get_sheet_names(file_path):
    sheet_names = []
    # open excel file from file path
    sheets = pd.read_excel(file_path, sheet_name=None)
    # find if Mall or Estate in sheet
    if 'Mall' in sheets:
        sheet_names = ['Mall', 'FoodType', 'Restaurant', 'Restaurant_FoodType', 'Restaurant_Floor', 'ShopType',
                       'Shop', 'Shop_ShopType', 'Shop_Floor', 'Location', 'Facility', 'Facility_FacilityType', 
                       'Facility_Floor', 'Navigation', 'Service']  # junban important
    
    elif 'Estate' in sheets:
        sheet_names = ['Estate', 'ClubhouseFacility', 'ClubhouseFacility_FacilityType', 'NearbyFacilityType', 'NearbyFacility',
                       'NearbyFac_NearbyFacType', 'Location', 'Navigation']
    return sheet_names


def write_restaurant_shop_name_to_json(file_path, output_json_name):

    sheets = pd.read_excel(file_path, sheet_name=None)
    if "Restaurant" not in sheets or "Shop" not in sheets:
        return None

    df_restaurant = pd.read_excel(file_path, sheet_name="Restaurant")
    df_restaurant = df_restaurant[["name", "name_sim", "name_eng", "synonyms"]]
    df_shop = pd.read_excel(file_path, sheet_name="Shop")
    df_shop = df_shop[["name", "name_sim", "name_eng", "synonyms"]]

    df = df_restaurant.append(df_shop, ignore_index=True)
    info_list = []

    for i in range(len(df)):
        name_cantonese = str(df.loc[i, "name"])
        if name_cantonese:
            name_cantonese_t2s = converter.convert(name_cantonese)
        name_sim = str(df.loc[i, "name_sim"])
        name_eng = str(df.loc[i, "name_eng"]).lower()

        if name_cantonese and name_sim and name_eng:
            additional_synonyms = ", ".join([name_cantonese, name_cantonese_t2s, name_sim, name_eng])
            synonyms = df.loc[i, "synonyms"]
            if not pd.isnull(synonyms):
                synonyms = str(df.loc[i, "synonyms"])
                synonyms = synonyms + ", " + additional_synonyms
            else:
                synonyms = additional_synonyms
        
        # if synonnyms --> do some preprocessing to this!!!!
        if synonyms:  # A, B, C, D
            preprocessed_synonyms = [node_synonyms_preprocess(i) for i in synonyms.split(", ")]
            preprocessed_synonyms = ", ".join(preprocessed_synonyms)
            synonyms = synonyms + ", " + preprocessed_synonyms
            
            info_list.append({
                'name': name_cantonese,
                'name_sim': name_sim,
                'name_eng': name_eng,
                'synonyms': synonyms
            })

    info_dict = {"data": info_list}
    
    with open(output_json_name, "w", encoding="utf-8") as output_json:
        json.dump(info_dict, output_json, ensure_ascii=False)



def push_data(session, file_path, site_id, version):
    
    if not os.path.exists(Config.restaurant_shop_dictionary_base_path):
        os.mkdir(Config.restaurant_shop_dictionary_base_path)

    output_json_name = os.path.join(Config.restaurant_shop_dictionary_base_path, f"{site_id}_{version}.json")
    if os.path.exists(output_json_name):
        open(output_json_name, "w").close()

    sheet_names = get_sheet_names(file_path)
    assert bool(sheet_names)==True, "Wrong excel file is used"
    
    write_restaurant_shop_name_to_json(file_path, output_json_name)
    print(f"save json file {output_json_name} successfully!")

    for sheet_name in sheet_names:
        write_data(session=session, file_path=file_path, sheet_name=sheet_name, version=version, site_id=site_id)
        print(f"finish pushing data from sheet {sheet_name}!")
    print("finish pushing data to neo4j!")



def copy_previous_file(input_path, site_id, output_path):
    print("copying previous file")
    # get required sheet names
    sheet_names = get_sheet_names(input_path)
    print(sheet_names)

    # get all sheet names from excel
    sheetnames_to_df = pd.read_excel(input_path, sheet_name=None)
    print(sheetnames_to_df)

    sheet_to_df = dict()
    
    for sheet_name, df in sheetnames_to_df.items():
        
        if sheet_name in sheet_names:

            # filter out 'delete' and 'un-merge'
            df = df[~df['action'].isin(['un-merge', 'delete'])].reindex()
            
            # if previous actions are 'create' or 'merge', keep the values
            # if previous action is 'update', change it to 'create'
            # if previous action is 'no action', change it to either 'create' or 'merge' depending on sheet name
            for i in range(len(df)):
                
                if df.loc[i, 'action'] == "update":
                    df.loc[i, 'action'] = "create"

                elif df.loc[i, 'action'] == "no action":
                    
                    if sheet_name not in ["Facility_FacilityType",  # start of mall
                                          "Facility_Floor",
                                          "Restaurant_FoodType",
                                          "Restaurant_Floor",
                                          "Shop_ShopType",
                                          "Shop_Floor",
                                          "Mall_Navigation",
                                          "Location_Navigation",  # end of mall
                                          "ClubhouseFacility_FacilityType", # start of estate
                                          "NearbyFac_NearbyFacType"  
                                          "Navigation"]:  # end of estate!
                        df.loc[i, 'action'] = "create"
                    
                    else:
                        df.loc[i, 'action'] = "merge"
             
            # store mapping for sheet name and dataframe 
            sheet_to_df[sheet_name] = df
    print(sheet_to_df)

    # write excel
    print(sheet_to_df)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet, dataframe in sheet_to_df.items():
            print(sheet, dataframe)
            dataframe.to_excel(writer, sheet_name=sheet, index=False)
    
    return output_path



if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    site_id = args.site_id
    file_path = args.file_path
    version = args.version

    session.run(f"match (n) where n.site_id='{site_id}' and n.mode='{version}' detach delete n")
    push_data(session, file_path, site_id, version)
    