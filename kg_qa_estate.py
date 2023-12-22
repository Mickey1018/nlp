import random
import re
import pandas as pd
from neo4j import GraphDatabase
import warnings

warnings.filterwarnings("ignore")

def get_first_entity(slot_list):
    if not slot_list:
        return None
    else:
        return slot_list[0]

# define mapping of intent and required slots
Intent_RequiredSlots = {"facilitiesinformation_address": ["facility", "phase", "floor"],
                        "facilitiesinformation_contact": ["facility", "phase"]}  # junban daiji!
general_terms = ['厕所', '洗手间', '卫生间', 'toilets', 'washrooms', 'restrooms']


# function to check required slots are filled
def check_required_slots(intent, slots):
    required_slots = Intent_RequiredSlots[intent]
    existing_slots = {}  # {slot1: value1, slot2: value2, ...}
    too_general_terms = []  # ['厕所']
    missing_slots = []  # [slot1, slot2]
    for req_slot in required_slots:
        slot_value = slots.get(req_slot)
        if slot_value:
            is_too_general = False
            for term in general_terms:
                if slot_value in term:
                    is_too_general = True
                    too_general_terms.append(term)
                    break
            if not is_too_general:
                existing_slots[req_slot]=slot_value
        else:
            missing_slots.append(req_slot)
    return existing_slots, too_general_terms, missing_slots


# function to generate text from KG
def generate_text(sid, topic, subtopic_1, subtopic_2, template_db, lang, **kwargs):
    
    # df = pd.read_excel(template_db, sheet_name=0)
    df = template_db

    if topic and subtopic_1 and subtopic_2:
        df_target = df.loc[(df['Topic']==topic) & (df['Subtopic_1']==subtopic_1) & (df['Subtopic_2']==subtopic_2)]
    elif topic and subtopic_1 and not subtopic_2:
        df_target = df.loc[(df['Topic']==topic) & (df['Subtopic_1']==subtopic_1) & (df['Subtopic_2'].isnull())]
    elif topic and not subtopic_1 and not subtopic_2:
        df = df[df["Topic"]==topic]
        df_target = df[df["Subtopic_1"].isna()]
    
    assert not df_target.empty, "no dataframe is found"

    if lang == 'zh-hk':
        template = df_target.iloc[0]['KG_Answer_Template']
    elif lang == 'zh-cn':
        template = df_target.iloc[0]['KG_Answer_Template_Sim']
    elif lang == 'en':
        template = df_target.iloc[0]['KG_Answer_Template_Eng']

    def replace_template(template):
        matched = re.search(r'\{.*?\}', template)
        if matched:
            start, end = matched.span()
            key_word = template[start+1:end-1]
            for k, v in kwargs.items():
                if k == key_word:
                    if start == 0:
                        template = v + template[end:]
                    else:
                        template = template[:start] + v + template[end:]
            return replace_template(template)
        return template

    return replace_template(template)


def get_answer_from_estate_kg(lang, site_id, version, session, intent, slots, template_db='template.xlsx'):

    follow_up = False
    options = []
    num_of_choice = 3

    if lang == 'zh-hk':
        name = 'name'
        address = 'address'
        method = 'method'
        location = 'location'
        contact = 'contact'
        opening_hours = 'opening_hours'
        charges = 'charges'
        seat = 'seat'
        size = 'size'
        provision = 'provision'
        menu = 'menu'
        paymethod = 'paymethod'
    elif lang == 'zh-cn':
        name = 'name_sim'
        address = 'address_sim'
        method = 'method_sim'
        location = 'location_sim'
        contact = 'contact'
        opening_hours = 'opening_hours_sim'
        charges = 'charges_sim'
        seat = 'seat_sim'
        size = 'size_sim'
        provision = 'provision_sim'
        menu = 'menu_sim'
        paymethod = 'paymethod_sim'
    elif lang == 'en':
        name = 'name_eng'
        address = 'address_eng'
        method = 'method_eng'
        location = 'location_eng'
        contact = 'contact'
        opening_hours = 'opening_hours_eng'
        charges = 'charges_eng'
        seat = 'seat_eng'
        size = 'size_eng'
        provision = 'provision_eng'
        menu = 'menu_eng'
        paymethod = 'paymethod_eng'

    # cypher query for each topic
    ##################################################################################################
    # Clubhouse Facility
    ##################################################################################################
    if intent == "clubhousefacility_type":

        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n:ClubhouseFacility)-[:PartOf]->(m) where "\
                f"n.site_id='{site_id}' and n.mode='{version}' and " \
                f"m.name='會所設施' return n.{name}"
        results = session.run(query).values()
        results = list(set([r[0] for r in results]))
        
        answer = ''
        
        clubhouse_facilities = ', '.join([r for r in results]) if results else ''
        
        if clubhouse_facilities:
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, clubhouse_facilities=clubhouse_facilities)

        else:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
        
        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer


    elif intent == "clubhousefacility_openinghours":

        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        if slots:
            clubhousefacility = get_first_entity(slots.get("clubhousefacility"))
        else:
            clubhousefacility = None

        print(clubhousefacility)

        if not clubhousefacility:
            subtopic_2 = "answer-not-found"
            topic_stat = [topic, subtopic_1, subtopic_2]
            answer = generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            return topic_stat, answer

        query = f"match (n:ClubhouseFacility)-[:PartOf]->(m) where m.name='會所設施' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' and " \
                f"toLower('{clubhousefacility}') in n.synonyms " \
                f"return n.{name}, n.{opening_hours}"
        print(query)
        results = session.run(query).values()
        print(results)
        answer = ''
        if results:
            for result in results:
                clubhousefacility, opening_hours = result[0], result[1]
                if clubhousefacility and opening_hours:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, clubhousefacility=clubhousefacility, opening_hours=opening_hours)
            
        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
        
        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer


    elif intent == "clubhousefacility_charges":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        if slots:
            clubhousefacility = get_first_entity(slots.get("clubhousefacility"))
        else:
            clubhousefacility = None

        if not clubhousefacility:
            subtopic_2 = "answer-not-found"
            topic_stat = [topic, subtopic_1, subtopic_2]
            answer = generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            return topic_stat, answer

        query = f"match (n:ClubhouseFacility)-[:PartOf]->(m) where m.name='會所設施' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' and " \
                f"toLower('{clubhousefacility}') in n.synonyms " \
                f"return n.{name}, n.{charges}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                clubhousefacility, charges = result[0], result[1]
                if clubhousefacility and charges:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, clubhousefacility=clubhousefacility, charges=charges)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
        
        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer


    elif intent == "clubhousefacility_location" or intent=="clubhousefacility_existence":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        if slots:
            clubhousefacility = get_first_entity(slots.get("clubhousefacility"))
        else:
            clubhousefacility = None

        if not clubhousefacility:
            subtopic_2 = "answer-not-found"
            topic_stat = [topic, subtopic_1, subtopic_2]
            answer = generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            return topic_stat, answer

        query = f"match (n:ClubhouseFacility)-[:PartOf]->(m) where m.name='會所設施' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' and " \
                f"toLower('{clubhousefacility}') in n.synonyms " \
                f"return n.{name}, n.{location}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                clubhousefacility, location = result[0], result[1]
                if clubhousefacility and location:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, clubhousefacility=clubhousefacility, location=location)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, clubhousefacility=clubhousefacility)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer


    elif intent == "clubhousefacility_seat":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        if slots:
            clubhousefacility = get_first_entity(slots.get("clubhousefacility"))
        else:
            clubhousefacility = None

        if not clubhousefacility:
            subtopic_2 = "answer-not-found"
            topic_stat = [topic, subtopic_1, subtopic_2]
            answer = generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            return topic_stat, answer

        query = f"match (n:ClubhouseFacility)-[:PartOf]->(m) where m.name='會所設施' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' and " \
                f"toLower('{clubhousefacility}') in n.synonyms " \
                f"return n.{name}, n.{seat}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                clubhousefacility, seat = result[0], result[1]
                if clubhousefacility and seat:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, clubhousefacility=clubhousefacility, seat=seat)

        if not answer:
            subtopic_2 = "not_found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, clubhousefacility=clubhousefacility)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer


    elif intent == "clubhousefacility_size":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        if slots:
            clubhousefacility = get_first_entity(slots.get("clubhousefacility"))
        else:
            clubhousefacility = None

        if not clubhousefacility:
            subtopic_2 = "answer-not-found"
            topic_stat = [topic, subtopic_1, subtopic_2]
            answer = generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            return topic_stat, answer

        query = f"match (n:ClubhouseFacility)-[:PartOf]->(m) where m.name='會所設施' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' and " \
                f"toLower('{clubhousefacility}') in n.synonyms " \
                f"return n.{name}, n.{size}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                clubhousefacility, size = result[0], result[1]
                if clubhousefacility and size:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, clubhousefacility=clubhousefacility, size=size)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer


    elif intent == "clubhousefacility_provision":   
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        if slots:
            clubhousefacility = get_first_entity(slots.get("clubhousefacility"))
        else:
            clubhousefacility = None

        if not clubhousefacility:
            subtopic_2 = "answer-not-found"
            topic_stat = [topic, subtopic_1, subtopic_2]
            answer = generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            return topic_stat, answer

        query = f"match (n:ClubhouseFacility)-[:PartOf]->(m) where m.name='會所設施' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' and " \
                f"toLower('{clubhousefacility}') in n.synonyms " \
                f"return n.{name}, n.{provision}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                clubhousefacility, provision = result[0], result[1]
                if clubhousefacility and provision:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, clubhousefacility=clubhousefacility, provision=provision)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    

    ##################################################################################################
    # Clubhouse Restaurant
    ##################################################################################################
    
    elif intent == "clubhouserestaurant_menu":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n:ClubhouseFacility) where n.name='會所餐廳' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{menu}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                menu = result[0]
                if menu:
                    answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, menu=menu)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    

    elif intent == "clubhouserestaurant_payment":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n:ClubhouseFacility) where n.name='會所餐廳' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{paymethod}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                paymethod = result[0]
                if paymethod:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, paymethod=paymethod)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    

    elif intent == "clubhouserestaurant_address":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n:ClubhouseFacility) where n.name='會所餐廳' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{location}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                location = result[0]
                if location:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, location=location)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    

    elif intent == "clubhouserestaurant_contact":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n:ClubhouseFacility) where n.name='會所餐廳' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{contact}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                contact = result[0]
                if contact:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, contact=contact)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    

    elif intent == "clubhouserestaurant_openinghours":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n:ClubhouseFacility) where n.name='會所餐廳' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{opening_hours}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                openinghours = result[0]
                if openinghours:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, openinghours=openinghours)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer


    ##################################################################################################
    # Clubhouse information
    ##################################################################################################
    elif intent == "clubhouseinformation_email":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n:Clubhouse) where n.name='會所' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.email"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                email = result[0]
                if email:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, email=email)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    

    elif intent == "clubhouseinformation_address":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n:Clubhouse) where n.name='會所' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{address}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                address = result[0]
                if address:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, address=address)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    

    elif intent == "clubhouseinformation_contact":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n:Clubhouse) where n.name='會所' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{contact}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                contact = result[0]
                if contact:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, contact=contact)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    

    elif intent == "clubhouseinformation_openinghours":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n:Clubhouse) where n.name='會所' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{opening_hours}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                openinghours = result[0]
                if openinghours:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, openinghours=openinghours)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    
    ##################################################################################################
    # Address information
    ##################################################################################################
    elif intent == "estateinformation_address":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n) where n.name='寶馬山花園' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{name}, n.{address}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                estate, address = result[0], result[1]
                if estate and address:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, estate=estate, address=address)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    

    elif intent == "estateinformation_email":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n) where n.name='寶馬山花園' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{name}, n.email"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                estate, email = result[0], result[1]
                if estate and email:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, estate=estate, email=email)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    

    elif intent == "cs_contact":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n) where n.name='客戶服務中心' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{name}, n.{contact}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                name, contact = result[0], result[1]
                if name and contact:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, name=name, contact=contact)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    

    elif intent == "cs_address":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n) where n.name='客戶服務中心' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{name}, n.{address}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                name, address = result[0], result[1]
                if name and address:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, name=name, address=address)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    

    elif intent == "mo_contact":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n) where n.name='管理處' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{name}, n.{contact}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                name, contact = result[0], result[1]
                if name and contact:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, name=name, contact=contact)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    

    elif intent == "mo_address":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n) where n.name='管理處' and "\
                f"n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{name}, n.{address}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                name, address = result[0], result[1]
                if name and address:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, name=name, address=address)

        if not answer:
            subtopic_2 = "answer-not-found"
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        topic_stat = [topic, subtopic_1, subtopic_2]
        return topic_stat, answer
    

    elif intent == "nearby":
        topic, subtopic_1, subtopic_2 = intent, None, None
        
        if slots:
            facilitytype = get_first_entity(slots.get("facilitytype"))
            shoptype = get_first_entity(slots.get("shoptype"))
        else:
            subtopic_1 = "facility"
            subtopic_2 = "answer-not-found"
            answer = generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer

        if not facilitytype and not shoptype:
            subtopic_1 = "facility"
            subtopic_2 = "answer-not-found"
            answer = generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer

        if facilitytype:
            subtopic_1 = 'facility'

            # 1. find (facility)-[:PartOf]->(facility type)
            query = f"match (f)-[:PartOf]->(ft) where " \
                    f"toLower('{facilitytype}') in ft.synonyms and "\
                    f"f.site_id='{site_id}' and f.mode='{version}' " \
                    f"return f.{name}, f.{address}, f.{contact}, f.{opening_hours}"
            results = session.run(query).values()
            answer = ''
            if results:
                for result in results:
                    facilityname, address, contact, openinghours = result[0], result[1], result[2], result[3]
                    
                    if facilityname and address and contact and openinghours:
                        subtopic_2 = "addess+contact+openinghours"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                facilityname=facilityname, address=address, 
                                                contact=contact, openinghours=openinghours)
                    
                    elif facilityname and address and not contact and not openinghours:
                        subtopic_2 = "addess"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                facilityname=facilityname, address=address)
                    
                    elif facilityname and not address and contact and not openinghours:
                        subtopic_2 = "contact"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                facilityname=facilityname, contact=contact)
                    
                    elif facilityname and not address and not contact and openinghours:
                        subtopic_2 = "openinghours"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                facilityname=facilityname, openinghours=openinghours)
                    
                    elif facilityname and address and contact and not openinghours:
                        subtopic_2 = "addess+contact"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                facilityname=facilityname, address=address, contact=contact)
                    
                    elif facilityname and address and contact and not openinghours:
                        subtopic_2 = "address+openinghours"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                facilityname=facilityname, address=address, openinghours=openinghours)
                    
                    elif facilityname and not address and contact and openinghours:
                        subtopic_2 = "contact+openinghours"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                facilityname=facilityname, contact=contact, openinghours=openinghours)

            if not answer:
                subtopic_2 = "answer-not-found"
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer

        if shoptype:
            subtopic_1 = 'shop'

            # 1. find (shop)-[:PartOf]->(shop type)
            query = f"match (s)-[:PartOf]->(st) where " \
                    f"toLower('{shoptype}') in st.synonyms and "\
                    f"s.site_id='{site_id}' and s.mode='{version}' " \
                    f"return s.{name}, s.{address}, s.{contact}, s.{opening_hours}"
            results = session.run(query).values()
            answer = ''
            if results:
                for result in results:
                    shopname, address, contact, openinghours = result[0], result[1], result[2], result[3]
                    if shopname and address and contact and openinghours:
                        subtopic_2 = "addess+contact+openinghours"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                shopname=shopname, address=address, 
                                                contact=contact, openinghours=openinghours)
                    
                    elif facilityname and address and not contact and not openinghours:
                        subtopic_2 = "addess"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                shopname=shopname, address=address)
                    
                    elif facilityname and not address and contact and not openinghours:
                        subtopic_2 = "contact"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                shopname=shopname, contact=contact)
                    
                    elif facilityname and not address and not contact and openinghours:
                        subtopic_2 = "openinghours"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                shopname=shopname, openinghours=openinghours)
                    
                    elif facilityname and address and contact and not openinghours:
                        subtopic_2 = "addess+contact"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                shopname=shopname, address=address, contact=contact)
                    
                    elif facilityname and address and contact and not openinghours:
                        subtopic_2 = "address+openinghours"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                shopname=shopname, address=address, openinghours=openinghours)
                    
                    elif facilityname and not address and contact and openinghours:
                        subtopic_2 = "contact+openinghours"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                shopname=shopname, contact=contact, openinghours=openinghours)

            if not answer:
                subtopic_2 = "answer-not-found"
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer

    
    elif intent == "navigation":
        topic, subtopic_1, subtopic_2 = 'navigation', None, None

        location_from, location_to = get_first_entity(slots.get("locationfrom")), get_first_entity(slots.get("locationto"))
        
        if not location_to:
            subtopic_1 = 'answer-not-found'
            answer = generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, location_to=end)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer
        
        if location_from:
            # find if direct path exist
            query = f"match p=(s)-[:GetAccessTo*1..2]->(e) where " \
                    f"toLower('{location_from}') in s.synonyms and " \
                    f"toLower('{location_to}') in e.synonyms and " \
                    f"e.site_id='{site_id}' and e.mode='{version}' " \
                    f"return p, s.{name}, e.{name} ORDER BY length(p)"
            
            results = session.run(query).values()
            answer = ''
            
            if results:
                result = results[0]  # get result with the shortest path
                result_path, start, end = result[0], result[1], result[2]
                paths = list(result_path.__dict__.get('_relationships'))  # retrieve paths from path object
                for path in paths:
                    relationship_info = path.__dict__.get('_properties')
                    method = relationship_info.get(method)
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                  location_from=start, method=method, location_to=end)
            
            if not answer:
                subtopic_1 = "answer-not-found"
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, location_to=end)
                
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer


        elif not location_from:
            # try to get starting location and direct path
            query = f"match p=(s)-[:GetAccessTo*1..2]->(e) where " \
                    f"toLower('{location_to}') in e.synonyms " \
                    f"and e.site_id='{site_id}' and e.mode='{version}' " \
                    f"return p, s.{name}, e.{name} ORDER BY length(p)"
            
            results = session.run(query).values()
            answer = ''

            if results:
                result = results[0]  # get result with the shortest path
                result_path, start, end = result[0], result[1], result[2]
                paths = list(result_path.__dict__.get('_relationships'))  # retrieve paths from path object
                for path in paths:
                    relationship_info = path.__dict__.get('_properties')
                    method = relationship_info.get(method)
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                  location_from=start, method=method, location_to=end)
            
            if not answer:
                subtopic_1 = 'answer-not-found'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, location_to=end)

            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer
