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
    required_slots = Intent_RequiredSlots.get(intent)
    existing_slots = {}  # {slot1: value1, slot2: value2, ...}
    too_general_terms = []  # ['厕所']
    missing_slots = []  # [slot1, slot2]
    if required_slots:
        for req_slot in required_slots:
            slot_value = get_first_entity(slots.get(req_slot))
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
    elif lang == 'ja':
        template = df_target.iloc[0]['KG_Answer_Template_Ja']

    def replace_template(template):
        matched = re.search(r'\{.*?\}', template)
        if matched:
            start, end = matched.span()
            key_word = template[start+1:end-1]
            for k, v in kwargs.items():
                if k == key_word and v:
                    if k in ["shop_number"]:
                        pass
                        # postprocess v
                        if lang in ["zh-hk", "zh-cn"]:
                            pattern = re.compile(r"\s*-\s*")
                            v = pattern.sub("至", v)
                            
                        elif lang in ["en"]:
                            pattern = re.compile(r"\s*-\s*")
                            v = pattern.sub(" to ", v)
                            
                    if start == 0:
                        template = v + template[end:]
                    else:
                        template = template[:start] + v + template[end:]
            return replace_template(template)
        return template

    return replace_template(template)


def get_answer_from_mall_kg(lang, site_id, version, session, intent, slots, template_db):

    follow_up = False
    options = []
    num_of_choice = 3

    # opening hours & method (navigation)
    if lang == 'zh-hk':
        name = 'name'
        address = 'address'
        method = 'method'
        opening_hours = 'opening_hours'
    elif lang == 'zh-cn':
        name = 'name_sim'
        address = 'address_sim'
        method = 'method_sim' 
        opening_hours = 'opening_hours_sim'
    elif lang == 'en':
        name = 'name_eng'
        address = 'address_eng'
        method = 'method_eng'
        opening_hours = 'opening_hours_eng'
    elif lang == 'ja':
        name = 'name_ja'
        address = 'address_ja'
        method = 'method_ja'
        opening_hours = 'opening_hours_ja'
    

    if intent == "service":
        topic, subtopic_1, subtopic_2 = intent, None, None
        if not slots:
            subtopic_1 = "answer-not-found"
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        action, shopproduct = get_first_entity(slots.get("action")), get_first_entity(slots.get("shopproduct"))
        
        if not action and not shopproduct:
            subtopic_1 = "answer-not-found"
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        if action and not shopproduct:
            query = f'match (service)-[:ServedBy]->(provider) where ' \
                    f'service.mode="{version}" and service.site_id="{site_id}" and ' \
                    f'toLower("{action}") in service.synonyms ' \
                    f'return service.{name}, provider.{name}, labels(provider)'
        elif shopproduct and not action:
            query = f'match (service)-[:ServedBy]->(provider) where ' \
                    f'service.mode="{version}" and service.site_id="{site_id}" and ' \
                    f'toLower("{shopproduct}") in service.synonyms ' \
                    f'return service.{name}, provider.{name}, labels(provider)'
        
        elif action and shopproduct:
            query_1 = f'match (service)-[:ServedBy]->(provider) where ' \
                      f'service.mode="{version}" and service.site_id="{site_id}" and ' \
                      f'toLower("{shopproduct}") in service.synonyms ' \
                      f'return service.{name}, provider.{name}, labels(provider)'
            query_2 = f'match (service)-[:ServedBy]->(provider) where ' \
                      f'service.mode="{version}" and service.site_id="{site_id}" and ' \
                      f'toLower("{action}") in service.synonyms ' \
                      f'return service.{name}, provider.{name}, labels(provider)'
            results_1 = session.run(query_1).values()
            results_2 = session.run(query_2).values()
            if results_1 and not results_2:
                query = query_1
            elif not results_1 and results_2:
                query = query_2
            elif results_1 and results_2:
                query = query_2

        results = session.run(query).values()
        answer = ''
        if results:
            provider_2_service = dict()  # initialization
            for result in results:
                service, provider, provider_label = result[0], result[1], result[2]
                if service and provider:
                    # update service(s) provided with provider
                    if not provider_2_service.get(provider):
                        provider_2_service[provider] = dict()
                        provider_2_service[provider]["services"] = [service]
                        provider_2_service[provider]["label"] = provider_label
                    else:
                        provider_2_service[provider]["services"].append(service)

            if not provider_2_service:
                subtopic_1 = "answer-not-found"
                return generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            
            for provider, provider_info in provider_2_service.items():
                services_list = provider_info["services"]
                if len(services_list) > 1:
                    services = ", ".join([i for i in services_list])
                elif len(services_list)==1:
                    services = services_list[0]
                
                # If provider is facility
                if "Facility" in provider_info["label"]:
                    
                    if "ATM" in provider_info["label"]:
                        # change intent
                        intent = "facilitiesinformation_address"
                        # add slot
                        slots["facility"] = ["atm"]
                        # return self function
                        return get_answer_from_mall_kg(lang, site_id, version, session, intent, slots, template_db)
                        
                    subtopic_1 = "provided-by-mall-facility"
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                  provider=provider, services=services)
                
                elif "Shop" in provider_info["label"]:
                    subtopic_1 = "provided-by-shop"
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, provider=provider)
                    address_query = f'match (s)-[:At]->(floor)-[:At]->(phase) where s.{name}="{provider}" and s.mode="{version}" and s.site_id="{site_id}" ' \
                                    f'return s.{address}, floor.{name}, phase.{name}, s.shopnumber'
                    address_results = session.run(address_query).values()
                    if address_results:
                        address_result = address_results[0]
                        shop_name, floor, phase, shop_number = address_result[0], address_result[1], address_result[2], address_result[3]
                        if shop_name and floor and phase and shop_number:
                            answer += " " + generate_text(site_id, "shopinformation", "address", None, template_db, lang,
                                                          shop_name=shop_name, floor=floor, phase=phase, shop_number=shop_number)

                elif "ShopType" in provider_info["label"]:
                    subtopic_1 = "provided-by-shoptype"
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, provider=provider)
                
                elif "Location" in provider_info["label"]:
                    subtopic_1 = "provided-by-location"
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, provider=provider)
                    # try to find path to location
                    slots["locationto"] = [provider]
                    try:
                        topic_stat, answer_text, answer_options, follow_up = \
                            get_answer_from_mall_kg(lang, site_id, version, session, "navigation", slots, template_db)
                    except:
                        (topic_stat, answer_text), answer_options, follow_up = \
                            get_answer_from_mall_kg(lang, site_id, version, session, "navigation", slots, template_db), [], False
                    if 'answer-not-found' not in topic_stat:
                        answer += " " + answer_text

        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer
        else:
            subtopic_1 = "answer-not-found"
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)


        # output two services if same key word? e.g. mobile phone --> 手提電話充電服務 本地電話使用服
        # redirect to other intent


    elif intent == "facilitiesinformation_contact":

        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        facility, phase = get_first_entity(slots.get("facility")), get_first_entity(slots.get("phase"))

        if not facility:
            subtopic_2 = 'answer-not-found'
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        # _, too_general_terms, _ = check_required_slots(intent, slots)
        # if too_general_terms:
        #     term = too_general_terms[0].lower()
 
        #     query = f'match (n)-[:PartOf]->(m)-[:PartOf]->(k) where ' \
        #             f'toLower(m.synonyms) contains "{term}" and ' \
        #             f'k.name="設施" and ' \
        #             f'm.mode="{version}" and m.site_id="{site_id}" '\
        #             f'return n.{name} order by n.id'
        #     results = session.run(query).values()
        #     if results:
        #         options = [result[0] for result in results]  # ['男洗手間', '女洗手間', '傷殘人士洗手間']
        #         follow_up = True
        #         subtopic_2 = 'followup-too-general'
        #         ans = generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
        #                             options=', '.join(options))
        #         return ans, options, follow_up

        # deal with facility that is in serveral phase
        if not phase:
            query = f'match (facility)<-[:PartOf]-(n)-[:At]->(f:Floor)-[:At]->(phase:Phase) where ' \
                        f'toLower("{facility}") in facility.synonyms and ' \
                        f'facility.mode="{version}" and facility.site_id="{site_id}" ' \
                        f'return distinct phase order by phase.id'
            results = session.run(query).values()
            if results:
                options = [result[0].__dict__.get("_properties").get(name) for result in results]
                follow_up = True
                subtopic_2 = "followup-phase"
                ans = generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                    options=', '.join(options))
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, ans, options, follow_up

        if facility and phase:
            query = f"match (n:Facility)-[:At]->(f:Floor)-[:At]->(p:Phase) where n.site_id='{site_id}' and n.mode='{version}' and " \
                    f"toLower('{facility}') in n.synonyms and toLower('{phase}') in p.synonyms " \
                    f"return n.{name}, n.contact"
            results = session.run(query).values()
            answer = ''
            if results:
                for result in results:
                    facility, contact = result[0], result[1]
                    if facility and contact:
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, facility=facility, contact=contact)
            else:
                subtopic_2 = 'answer-not-found'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer
        
        # deal with facility that is in single category, e.g. 客戶服務中心
        if facility:
            query = f"match (n:Facility) where n.site_id='{site_id}' and n.mode='{version}' and " \
                    f"toLower('{facility}') in n.synonyms " \
                    f"return n.{name}, n.contact"
            results = session.run(query).values()
            answer = ''
            if results:
                for result in results:
                    facility, contact = result[0], result[1]
                    if facility and contact:
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, facility=facility, contact=contact)
            else:
                subtopic_2 = 'answer-not-found'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer


    elif intent in ["facilitiesinformation_openinghours", "carpark_openinghours"]:

        intent = "facilitiesinformation_openinghours"

        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        facility, phase = get_first_entity(slots.get("facility")), get_first_entity(slots.get("phase"))

        if phase:  # if phase is provided
            query = f"match (n:Facility)-[:At]->(f:Floor)-[:At]->(p:Phase) where n.site_id='{site_id}' and n.mode='{version}' and " \
                    f"toLower('{facility}') in n.synonyms and toLower('{phase}') in p.synonyms " \
                    f"return n.{name}, n.{opening_hours}"
        else:  # if phase is not provided
            query = f"match (n:Facility) where n.site_id='{site_id}' and n.mode='{version}' and " \
                    f"toLower('{facility}') in n.synonyms return n.{name}, n.{opening_hours}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                facility, opening_hours = result[0], result[1]
                if facility and opening_hours:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, facility=facility, opening_hours=opening_hours)
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up
        else:
            subtopic_2 = 'answer-not-found'
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up


    elif intent in ["facilitiesinformation_address", "carpark_address"]:
        
        if intent == "carpark_address":
            intent = "facilitiesinformation_address"

        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        if not slots:
            subtopic_2 = 'answer-not-found'
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)

        facility, phase, floor = get_first_entity(slots.get("facility")), get_first_entity(slots.get("phase")), get_first_entity(slots.get("floor"))

        if not facility:
            subtopic_2 = 'answer-not-found'
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
        
        # query to deal with oc stem lab
        query = f'match (s)<-[:NextTo]-(n)-[:At]->(f:Floor)-[:At]->(phase:Phase) '\
                f'match (n)-[:PartOf]->(m) where m.name="設施" and ' \
                f'toLower("{facility}") in n.synonyms and ' \
                f'n.mode="{version}" and n.site_id="{site_id}" ' \
                f'return n.{name}, phase.{name}, s.{name}, s.shop_number, f.{name}'
        # print(query)
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                facility, phase, shop, shop_number, floor = result[0], result[1], result[2], result[3], result[4]
                if facility and phase and shop and shop_number and floor:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                  facility=facility, phase=phase, shop=shop, shop_number=shop_number, floor=floor)
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up


        _, too_general_terms, _ = check_required_slots(intent, slots)
        
        
        # Check if facility is ATM
        ##################### ATM ##############################
        facility_is_atm_query = f'match (n:ATM)-[:PartOf]->(f) where f.name="設施" and ' \
                                f'toLower("{facility}") in n.synonyms and ' \
                                f'n.mode="{version}" and n.site_id="{site_id}" ' \
                                f'return n.name'

        facility_is_atm = session.run(facility_is_atm_query).values()
        
        if facility_is_atm:
            get_atm_query = f'match (atm:ATM)-[:PartOf]->(ATM:ATM) where ' \
                            f'atm.mode="{version}" and atm.site_id="{site_id}" ' \
                            f'return distinct atm order by atm.id'
            get_atm_results = session.run(get_atm_query).values()
            answer = ''
            if get_atm_results:
                options = [result[0].__dict__.get("_properties").get(name) for result in get_atm_results]
                if options:
                    atms = ', '.join(options)
                    subtopic_2 = "followup-too-general"
                    answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, options=atms)
            if answer:
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
        ##################### ATM ##############################

        if too_general_terms:
            term = too_general_terms[0].lower()
            query = f'match (shop)<-[:NextTo]-(fac)-[:PartOf]->(cat_small)-[:PartOf]->(cat_big)-[:PartOf]->(k) where ' \
                    f'toLower("{term}") in cat_big.synonyms and ' \
                    f'k.name="設施" and ' \
                    f'labels(shop) in [["Shop"], ["Restaurant"]] and ' \
                    f'cat_big.mode="{version}" and cat_big.site_id="{site_id}" '\
                    f'return distinct cat_small order by cat_small.id'
            results = session.run(query).values()
            answer = ''
            if results:
                options = [result[0].__dict__.get("_properties").get(name) for result in results]  # ['男洗手間', '女洗手間', '傷殘人士洗手間']
                subtopic_2 = 'followup-too-general'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, options=', '.join(options))
            if answer:
                follow_up = True
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
            else:
                subtopic_2 = 'answer-not-found'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
            
        if not phase:
            query = f'match (facility)<-[:PartOf]-(n)-[:At]->(f:Floor)-[:At]->(phase:Phase) ' \
                    f'match (n)-[:NextTo]->(shop) where ' \
                    f'labels(shop) in [["Shop"], ["Restaurant"]] and ' \
                    f'toLower("{facility}") in facility.synonyms and ' \
                    f'facility.mode="{version}" and facility.site_id="{site_id}" ' \
                    f'return distinct phase order by phase.id'
            results = session.run(query).values()
            answer = ''
            if results:
                options = [result[0].__dict__.get("_properties").get(name) for result in results]
                subtopic_2 = 'followup-phase'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, options=', '.join(options))
            if answer:
                follow_up = True
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
            else:
                subtopic_2 = 'answer-not-found'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
        
        if not floor:
            query = f'match (facility)<-[:PartOf]-(n)-[:At]->(f:Floor)-[:At]->(phase:Phase) ' \
                    f'match (n)-[:NextTo]->(shop) where ' \
                    f'labels(shop) in [["Shop"], ["Restaurant"]] and ' \
                    f'toLower("{facility}") in facility.synonyms and ' \
                    f'toLower("{phase}") in phase.synonyms and ' \
                    f'facility.mode="{version}" and facility.site_id="{site_id}" ' \
                    f'return distinct f order by f.id'
            results = session.run(query).values()
            answer = ''
            if results:
                if len(results) > 1:
                    options = [result[0].__dict__.get("_properties").get(name) for result in results]
                    subtopic_2 = 'followup-floor'
                    answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, options=', '.join(options))
                else:
                    floor = results[0][0].__dict__.get("_properties").get(name)
            if answer and len(results) > 1:
                follow_up = True
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
            elif not answer and not floor:
                subtopic_2 = 'answer-not-found'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up

        if facility and phase and floor:
            query = f'match (facility)<-[:PartOf]-(n)-[:At]->(f:Floor)-[:At]->(phase:Phase) where ' \
                    f'toLower("{facility}") in facility.synonyms and ' \
                    f'toLower("{phase}") in phase.synonyms and ' \
                    f'toLower("{floor}") in f.synonyms and ' \
                    f'n.mode="{version}" and n.site_id="{site_id}" ' \
                    f'match (s)<-[:NextTo]-(n) '\
                    f'return n.{name}, phase.{name}, s.{name}, s.shop_number, f.{name}'
            results = session.run(query).values()
            answer = ''
            if results:
                for i, result in enumerate(results):
                    facility, phase, shop, shop_number, floor = result[0], result[1], result[2], result[3], result[4]
                    if facility and phase and shop and shop_number and floor:
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                      facility=facility, phase=phase, shop=shop, shop_number=shop_number, floor=floor)
            if answer:
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
            else:
                subtopic_2 = 'answer-not-found'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up


    elif intent == "mallinformation_openinghours":

        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n:Mall) where n.site_id='{site_id}' and n.mode='{version}' return n.{name}, n.{opening_hours}"

        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                mall, opening_hours = result[0], result[1]
                if mall and opening_hours:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, mall=mall, opening_hours=opening_hours)
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer
        else:
            subtopic_2 = 'answer-not-found'
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer


    elif intent == "mallinformation_contact":

        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n:Mall) where n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{name}, n.contact"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                mall, contact = result[0], result[1]
                if mall and contact:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, mall=mall, contact=contact)
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer
        else:
            subtopic_2 = 'answer-not-found'
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer


    elif intent == "mallinformation_email":

        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        query = f"match (n:Mall) where n.site_id='{site_id}' and n.mode='{version}' " \
                f"return n.{name}, n.email"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                mall, email = result[0], result[1]
                if mall and email:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, mall=mall, email=email)
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer
        else:
            subtopic_2 = 'answer-not-found'
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer


    elif intent == "mallinformation_address":

        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None

        phase = get_first_entity(slots.get("phase"))

        if phase:
            query = f"match (n:Phase) where n.site_id='{site_id}' and n.mode='{version}' " \
                    f"and toLower('{phase}') in n.synonyms return n.{name}, n.{address}"
        else:
            query = f"match (n:Phase) where n.site_id='{site_id}' and n.mode='{version}' " \
                    f"return n.{name}, n.{address} order by n.id"
        
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                mall, address = result[0], result[1]
                if mall and address:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, mall=mall, address=address)
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer
        else:
            subtopic_2 = 'answer-not-found'
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer


    elif intent == "foodtype":

        topic, subtopic_1, subtopic_2 = 'foodtype', None, None
        
        if slots:
            phase, floor, foodtype = get_first_entity(slots.get("phase")), get_first_entity(slots.get("floor")), get_first_entity(slots.get("foodtype"))
        else:
            phase, floor, foodtype = None, None, None
        
        if not foodtype:
            subtopic_1 = 'too-general'
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang), options, follow_up

        if phase:
            query = f"match (f)<-[:PartOf]-(n)-[:At]->(floor)-[:At]->(phase) where " \
                    f"n.site_id='{site_id}' and n.mode='{version}' and " \
                    f"toLower('{foodtype}') in f.synonyms and " \
                    f"toLower('{phase}') in phase.synonyms " \
                    f"return n.{name}, phase.{name}" if not floor else \
                        f"match (f)<-[:PartOf]-(n)-[:At]->(floor)-[:At]->(phase) where " \
                        f"n.site_id='{site_id}' and n.mode='{version}' and " \
                        f"toLower('{foodtype}') in f.synonyms and " \
                        f"toLower('{phase}') in phase.synonyms and " \
                        f"toLower('{floor}') in floor.synonyms " \
                        f"return n.{name}, phase.{name}"  # return more informationn!
            results = session.run(query).values()
            print(results)
            answer = ''
            if results:
                phase = results[0][1]
                results_name = list(set([r[0] for r in results]))
                # randomly pick restaurants from results
                results = random.sample(results_name, min(num_of_choice, len(results_name)))
                options.extend(['【'+phase+'】'+r for r in results])
                restaurants = ', '.join([r for r in results]) if options else ''
                if restaurants:
                    answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, phase=phase, restaurants=restaurants)
            
            if answer and len(options)==1:
                query = f"match (f)<-[:PartOf]-(n)-[:At]->(floor)-[:At]->(phase) where " \
                        f"n.site_id='{site_id}' and n.mode='{version}' and " \
                        f"toLower('{foodtype}') in f.synonyms and " \
                        f"toLower('{phase}') in phase.synonyms " \
                        f"return n.{name}, floor.{name}, phase.{name}, n.shop_number, n.{opening_hours}, n.contact"
                results = session.run(query).values()
                results = results[0]
                restaurant, floor, phase, shop_number, opening_hours, contact = \
                    results[0], results[1], results[2], results[3], results[4], results[5]
                
                if restaurant and floor and phase and shop_number and opening_hours and contact:
                    topic = "restaurantinformation"
                    subtopic_1 = "address-openinghours-contact"
                    answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                            restaurant=restaurant, phase=phase, floor=floor,
                                            shop_number=shop_number, opening_hours=opening_hours, contact=contact)
                
                elif restaurant and floor and phase and shop_number and opening_hours and not contact:
                    topic = "restaurantinformation"
                    subtopic_1 = "address-openinghours"
                    answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                            restaurant=restaurant, phase=phase, floor=floor,
                                            shop_number=shop_number, opening_hours=opening_hours)
                
                elif restaurant and floor and phase and shop_number and not opening_hours and contact:
                    topic = "restaurantinformation"
                    subtopic_1 = "address-contact"
                    answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                            restaurant=restaurant, phase=phase, floor=floor,
                                            shop_number=shop_number, contact=contact)
                
                elif restaurant and floor and phase and shop_number and not opening_hours and not contact:
                    topic = "restaurantinformation"
                    subtopic_1 = "address"
                    answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                            restaurant=restaurant, phase=phase, floor=floor,
                                            shop_number=shop_number)

                options = []

            if answer:
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
            else:
                subtopic_1 = 'answer-not-found'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
        else:
            answer = ''
            phases = session.run(f'match (n:Phase) with distinct n.{name} as name, n.id as id ' \
                                 f'order by id return name').values()
            phases = [i[0] for i in phases]
            phases_set = []
            for phase in phases:
                if phase not in phases_set:
                    phases_set.append(phase)
            phases = phases_set
            for phase in phases:
                query = f"match (f)<-[:PartOf]-(n)-[:At]->(floor)-[:At]->(phase) where " \
                        f"n.site_id='{site_id}' and n.mode='{version}' and " \
                        f"toLower('{foodtype}') in f.synonyms and " \
                        f"toLower('{phase}') in phase.synonyms " \
                        f"return n.{name}" if not floor else \
                            f"match (f)<-[:PartOf]-(n)-[:At]->(floor)-[:At]->(phase) where " \
                            f"n.site_id='{site_id}' and n.mode='{version}' and " \
                            f"toLower('{foodtype}') in f.synonyms and " \
                            f"toLower('{phase}') in phase.synonyms and " \
                            f"toLower('{floor}') in floor.synonyms " \
                            f"return n.{name}"

                results = session.run(query).values()
                temp_answer = ''
                if results:
                    # all_results.append(results)
                    results_name = list(set([r[0] for r in results]))
                    # randomly pick restaurants from results
                    results_name = random.sample(results_name, min(len(results_name), num_of_choice))
                    options.extend(['【'+phase+'】'+r for r in results_name])
                    restaurants = ', '.join([r for r in results_name]) if options else ''
                    if restaurants:
                        temp_answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, phase=phase, restaurants=restaurants)
                answer += temp_answer
            
            if answer and len(options)==1:
                query = f"match (f)<-[:PartOf]-(n)-[:At]->(floor)-[:At]->(phase) where " \
                        f"n.site_id='{site_id}' and n.mode='{version}' and " \
                        f"toLower('{foodtype}') in f.synonyms " \
                        f"return n.{name}, floor.{name}, phase.{name}, n.shop_number, n.{opening_hours}, n.contact"
                print(query)
                results = session.run(query).values()
                results = results[0]
                print(results)
                restaurant, floor, phase, shop_number, opening_hours, contact = \
                    results[0], results[1], results[2], results[3], results[4], results[5]
                
                if restaurant and floor and phase and shop_number and opening_hours and contact:
                    topic = "restaurantinformation"
                    subtopic_1 = "address-openinghours-contact"
                    answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                            restaurant=restaurant, phase=phase, floor=floor,
                                            shop_number=shop_number, opening_hours=opening_hours, contact=contact)
                
                elif restaurant and floor and phase and shop_number and opening_hours and not contact:
                    topic = "restaurantinformation"
                    subtopic_1 = "address-openinghours"
                    answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                            restaurant=restaurant, phase=phase, floor=floor,
                                            shop_number=shop_number, opening_hours=opening_hours)
                
                elif restaurant and floor and phase and shop_number and not opening_hours and contact:
                    topic = "restaurantinformation"
                    subtopic_1 = "address-contact"
                    answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                            restaurant=restaurant, phase=phase, floor=floor,
                                            shop_number=shop_number, contact=contact)
                
                elif restaurant and floor and phase and shop_number and not opening_hours and not contact:
                    topic = "restaurantinformation"
                    subtopic_1 = "address"
                    answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                            restaurant=restaurant, phase=phase, floor=floor, shop_number=shop_number)

                options = []

            if answer:
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
            
            # what if restaurant was detected instead of foodtype?
            topic, subtopic_1, subtopic_2 = "restaurantinformation", "address", None
            query = f"match (r:Restaurant)-[:At]->(floor)-[:At]->(phase) where " \
                    f"r.site_id='{site_id}' and r.mode='{version}' " \
                    f"and toLower('{foodtype}') in r.synonyms " \
                    f"return r.{name}, floor.{name}, phase.{name}, r.shop_number"
            results = session.run(query).values()
            if results:
                for result in results:
                    restaurant, floor, phase, shop_number = result[0], result[1], result[2], result[3]
                    if restaurant and floor and phase and shop_number:
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                      restaurant=restaurant, floor=floor, phase=phase, shop_number=shop_number)
            if answer:
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
            else:
                topic, subtopic_1, subtopic_2 = "foodtype", None, None
                options = []
                follow_up = False
            
            if not answer:
                topic = 'foodtype'
                subtopic_1 = 'answer-not-found'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up


    elif intent == "restaurantinformation_address":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None
        phase = get_first_entity(slots.get("phase")) 
        restaurant = get_first_entity(slots.get("restaurant"))
        foodtype = get_first_entity(slots.get("foodtype"))

        if not foodtype and not restaurant:
            subtopic_2 = 'answer-not-found'
            answer = generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up

        if phase:
            query = f"match (r)-[:At]->(floor)-[:At]->(phase) where " \
                    f"r.site_id='{site_id}' and r.mode='{version}' and " \
                    f"labels(r) in [['Restaurant']] and " \
                    f"toLower('{restaurant}') in r.synonyms and " \
                    f"toLower('{phase}') in phase.synonyms " \
                    f"return r.{name}, floor.{name}, phase.{name}, r.shop_number"
        else:
            query = f"match (r)-[:At]->(floor)-[:At]->(phase) where " \
                    f"r.site_id='{site_id}' and r.mode='{version}' and " \
                    f"labels(r) in [['Restaurant']] and " \
                    f"toLower('{restaurant}') in r.synonyms " \
                    f"return r.{name}, floor.{name}, phase.{name}, r.shop_number"
        results = session.run(query).values()
        answer = ''
        if results:
            print(results)
            for result in results:
                restaurant, floor, phase, shop_number = result[0], result[1], result[2], result[3]
                if restaurant and floor and phase and shop_number:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                  restaurant=restaurant, floor=floor, phase=phase, shop_number=shop_number)
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up
        
        
        # try to find foodtype instead of restaurant
        intent = "foodtype"
        if foodtype:
            slots["foodtype"] = [foodtype]
        elif restaurant:
            slots["foodtype"] = [restaurant]

        topic_stat, answer, options, follow_up = get_answer_from_mall_kg(lang, site_id, version, session, intent, slots, template_db)
        
        if 'answer-not-found' not in topic_stat:
            return topic_stat, answer, options, follow_up
        else:
            answer = ''
            options = []
            follow_up = False
        

        # change to navigation intent, try to find location
        intent = "navigation"
        slots["locationto"] = [restaurant]
        topic_stat, answer = get_answer_from_mall_kg(lang, site_id, version, session, intent, slots, template_db)
        
        if 'answer-not-found' not in topic_stat:
            return topic_stat, answer, options, follow_up
        else:
            answer = ''

        if not answer:
            subtopic_2 = 'answer-not-found'
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up

    
    elif intent == "restaurantinformation_openinghours":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None
        phase, restaurant = get_first_entity(slots.get("phase")), get_first_entity(slots.get("restaurant"))
        if phase:
            query = f"match (r)-[:At]->(floor)-[:At]->(phase) where r.site_id='{site_id}' and r.mode='{version}' " \
                    f"and toLower('{restaurant}') in r.synonyms and toLower('{phase}') in phase.synonyms " \
                    f"return r.{name}, r.{opening_hours}"
        else:
            query = f"match (r) where r.site_id='{site_id}' and r.mode='{version}' " \
                    f"and toLower('{restaurant}') in r.synonyms return r.{name}, r.{opening_hours}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                restaurant, opening_hours = result[0], result[1]
                if restaurant and opening_hours:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                  restaurant=restaurant, opening_hours=opening_hours)
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up
        else:
            subtopic_2 = 'answer-not-found'
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up


    elif intent == "restaurantinformation_contact":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None
        phase, restaurant = get_first_entity(slots.get("phase")), get_first_entity(slots.get("restaurant"))
        if phase:
            query = f"match (r)-[:At]->(floor)-[:At]->(phase) where r.site_id='{site_id}' and r.mode='{version}' " \
                    f"and toLower('{restaurant}') in r.synonyms and toLower('{phase}') in phase.synonyms " \
                    f"return r.{name}, r.contact"
        else:
            query = f"match (r) where r.site_id='{site_id}' and r.mode='{version}' " \
                    f"and toLower('{restaurant}') in r.synonyms return r.{name}, r.contact"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                restaurant, contact = result[0], result[1]
                if restaurant and contact:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, restaurant=restaurant, contact=contact)
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up
        else:
            subtopic_2 = 'answer-not-found'
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up


    elif intent in ["restaurantinformation_payment", "restaurantinformation_menu"]:
        intent = "restaurantinformation_payment"
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None
        phase, restaurant, query = get_first_entity(slots.get("phase")), get_first_entity(slots.get("restaurant")), None
        shop = get_first_entity(slots.get("shop"))
        if shop and not restaurant:
            restaurant = shop
        if phase:
            query = f"match (r)-[:At]->(floor)-[:At]->(phase) where r.site_id='{site_id}' and r.mode='{version}' " \
                    f"and toLower('{restaurant}') in r.synonyms and toLower('{phase}') in phase.synonyms " \
                    f"return r.{name}, r.contact"
        else:
            query = f"match (r) where r.site_id='{site_id}' and r.mode='{version}' " \
                    f"and toLower('{restaurant}') in r.synonyms return r.{name}, r.contact"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                restaurant, contact = result[0], result[1]
                if restaurant and contact:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, restaurant=restaurant, contact=contact)
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up
        else:
            subtopic_2 = 'answer-not-found'
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up


    elif intent == "shoptype":
        if slots:
            phase, shoptype, shopproduct = get_first_entity(slots.get("phase")), get_first_entity(slots.get("shoptype")), get_first_entity(slots.get('shopproduct'))
        else:
            phase, shoptype, shopproduct = None, None, None

        # topic = "shopproduct" if shopproduct else "shoptype"
        topic = "shoptype"
        subtopic_1, subtopic_2 = None, None

        if not shoptype and not shopproduct:
            subtopic_1 = "answer-not-found"
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang), [], False
        
        if shopproduct or shoptype:
            if shopproduct:
                p = shopproduct
            if shoptype:
                p = shoptype

            if phase:
                query = f"match (s)<-[:PartOf]-(n)-[:At]->(floor)-[:At]->(phase) where " \
                        f"n.site_id='{site_id}' and n.mode='{version}' and " \
                        f"toLower('{p}') in s.synonyms and " \
                        f"toLower('{phase}') in phase.synonyms " \
                        f"return n.{name}, s.{name}, phase.{name}"
                results = session.run(query).values()
                answer = ''
                if results:
                    phase = results[0][-1]
                    results = list(set([r[0] for r in results]))
                    # randomly pick restaurants from results
                    results = random.sample(results, min(len(results), num_of_choice))
                    options.extend(['【'+phase+'】'+r for r in results])
                    shops = ', '.join([r for r in results]) if options else ''
                    if shops:
                        answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, phase=phase, shops=shops)
                
                if answer and len(options)==1:
                    query = f"match (s)<-[:PartOf]-(n)-[:At]->(floor)-[:At]->(phase) where " \
                            f"n.site_id='{site_id}' and n.mode='{version}' and " \
                            f"toLower('{p}') in s.synonyms and " \
                            f"toLower('{phase}') in phase.synonyms " \
                            f"return n.{name}, floor.{name}, phase.{name}, n.shop_number, n.{opening_hours}, n.contact"
                    results = session.run(query).values()
                    results = results[0]
                    shop, floor, phase, shop_number, opening_hours, contact = \
                        results[0], results[1], results[2], results[3], results[4], results[5]
                    
                    if shop and floor and phase and shop_number and opening_hours and contact:
                        topic = "shopinformation"
                        subtopic_1 = "address-openinghours-contact"
                        answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                shop=shop, phase=phase, floor=floor,
                                                shop_number=shop_number, opening_hours=opening_hours, contact=contact)
                
                    elif shop and floor and phase and shop_number and opening_hours and not contact:
                        topic = "shopinformation"
                        subtopic_1 = "address-openinghours"
                        answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                shop=shop, phase=phase, floor=floor,
                                                shop_number=shop_number, opening_hours=opening_hours)
                    
                    elif shop and floor and phase and shop_number and not opening_hours and contact:
                        topic = "shopinformation"
                        subtopic_1 = "address-contact"
                        answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                shop=shop, phase=phase, floor=floor,
                                                shop_number=shop_number, contact=contact)
    
                    options = []

                if answer:
                    topic_stat = [topic, subtopic_1, subtopic_2]
                    return topic_stat, answer, options, follow_up
                else:
                    subtopic_1 = 'answer-not-found'
                    answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, shopproduct=shopproduct)
                    topic_stat = [topic, subtopic_1, subtopic_2]
                    return topic_stat, answer, options, follow_up
            else:
                answer = ''
                phases = session.run(f'match (n:Phase) with distinct n.{name} as name, n.id as id ' \
                                 f'order by id return name').values()
                phases = [i[0] for i in phases]
                phases_set = []
                for phase in phases:
                    if phase not in phases_set:
                        phases_set.append(phase)
                phases = phases_set
                for phase in phases:
                    query = f"match (s)<-[:PartOf]-(n)-[:At]->(floor)-[:At]->(phase) where " \
                            f"n.site_id='{site_id}' and n.mode='{version}' and " \
                            f"toLower('{p}') in s.synonyms and " \
                            f"toLower('{phase}') in phase.synonyms " \
                            f"return n.{name}, s.{name}"
                    results = session.run(query).values()
                    temp_answer = ''
                    if results:
                        results = list(set([r[0] for r in results]))
                        print(results)
                        # randomly pick restaurants from results
                        results = random.sample(results, min(len(results), num_of_choice))
                        options.extend(['【'+phase+'】'+r for r in results])
                        shops = ', '.join([r for r in results]) if options else ''
                        if shops:
                            temp_answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, phase=phase, shops=shops)
                    answer += temp_answer
                
                if answer and len(options)==1:
                    query = f"match (s)<-[:PartOf]-(n)-[:At]->(floor)-[:At]->(phase) where " \
                            f"n.site_id='{site_id}' and n.mode='{version}' and " \
                            f"toLower('{p}') in s.synonyms " \
                            f"return n.{name}, floor.{name}, phase.{name}, n.shop_number, n.{opening_hours}, n.contact"
                    results = session.run(query).values()
                    results = results[0]

                    shop, floor, phase, shop_number, opening_hours, contact = \
                        results[0], results[1], results[2], results[3], results[4], results[5]
                    
                    if shop and floor and phase and shop_number and opening_hours and contact:
                        topic = "shopinformation"
                        subtopic_1 = "address-openinghours-contact"
                        answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                shop=shop, phase=phase, floor=floor,
                                                shop_number=shop_number, opening_hours=opening_hours, contact=contact)
                
                    elif shop and floor and phase and shop_number and opening_hours and not contact:
                        topic = "shopinformation"
                        subtopic_1 = "address-openinghours"
                        answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                shop=shop, phase=phase, floor=floor,
                                                shop_number=shop_number, opening_hours=opening_hours)
                    
                    elif shop and floor and phase and shop_number and not opening_hours and contact:
                        topic = "shopinformation"
                        subtopic_1 = "address-contact"
                        answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                shop=shop, phase=phase, floor=floor,
                                                shop_number=shop_number, contact=contact)
    
                    options = []

                if answer:
                    topic_stat = [topic, subtopic_1, subtopic_2]
                    print(topic_stat, answer, options, follow_up)
                    return topic_stat, answer, options, follow_up
                
                # what if shop was detected instead of shoptype?
                topic, subtopic_1, subtopic_2 = "shopinformation", "address", None
                query = f"match (s)-[:At]->(floor)-[:At]->(phase) where " \
                        f"s.site_id='{site_id}' and s.mode='{version}' " \
                        f"and toLower('{shoptype}') in s.synonyms " \
                        f"return s.{name}, floor.{name}, phase.{name}, s.shop_number"
                results = session.run(query).values()
                if results:
                    for result in results:
                        shop_name, floor, phase, shop_number = result[0], result[1], result[2], result[3]
                        if shop_name and floor and phase and shop_number:
                            answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                          shop_name=shop_name, floor=floor, phase=phase, shop_number=shop_number)
                
                if answer:
                    topic_stat = [topic, subtopic_1, subtopic_2]
                    return topic_stat, answer, options, follow_up
                else:
                    topic, subtopic_1, subtopic_2 = "shoptype", None, None
                    options = []
                    follow_up = False
                
                if not answer:
                    topic = 'shoptype'
                    subtopic_1 = 'answer-not-found'
                    answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, shopproduct=shopproduct)
                    topic_stat = [topic, subtopic_1, subtopic_2]
                    return topic_stat, answer, options, follow_up


    elif intent == "shopinformation_address":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None
        
        phase = get_first_entity(slots.get("phase"))
        shop = get_first_entity(slots.get("shop"))
        shoptype = get_first_entity(slots.get("shoptype"))
        
        if not shop and not shoptype:
            subtopic_2 = 'answer-not-found'
            answer = generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, shop=shop)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up
        
        if not shop and shoptype:
            shop = shoptype

        query = f"match (s)-[:At]->(floor)-[:At]->(phase) where s.site_id='{site_id}' and s.mode='{version}' " \
                f"and toLower('{shop}') in s.synonyms and " \
                f"toLower('{phase}') in phase.synonyms " \
                f"return s.{name}, floor.{name}, phase.{name}, s.shop_number" if phase else \
                    f"match (s)-[:At]->(floor)-[:At]->(phase) where s.site_id='{site_id}' and s.mode='{version}' " \
                    f"and toLower('{shop}') in s.synonyms " \
                    f"return s.{name}, floor.{name}, phase.{name}, s.shop_number"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                shop_name, floor, phase, shop_number = result[0], result[1], result[2], result[3]
                if shop_name and floor and phase and shop_number:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                  shop_name=shop_name, floor=floor, phase=phase, shop_number=shop_number)
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up
        
        # change to shoptype intent
        intent = "shoptype"
        slots["shoptype"] = [shop]
        topic_stat, answer, options, follow_up = get_answer_from_mall_kg(lang, site_id, version, session, intent, slots, template_db)
        
        if 'answer-not-found' not in topic_stat:
            return topic_stat, answer, options, follow_up
        else:
            answer = ''
            options = []
            follow_up = False
        
        # change to navigation intent
        intent = "navigation"
        slots["locationto"] = [shop]
        topic_stat, answer = get_answer_from_mall_kg(lang, site_id, version, session, intent, slots, template_db)
        
        if 'answer-not-found' not in topic_stat:
            return topic_stat, answer, options, follow_up
        else:
            answer = ''
        
        if not answer:
            subtopic_2 = 'answer-not-found'
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, shop=shop)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up


    elif intent == "shopinformation_openinghours":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None
        phase, shop = get_first_entity(slots.get("phase")), get_first_entity(slots.get("shop"))
        if phase:
            query = f"match (s)-[:At]->(floor)-[:At]->(phase) where s.site_id='{site_id}' and s.mode='{version}' " \
                    f"and toLower('{shop}') in s.synonyms and toLower('{phase}') in phase.synonyms " \
                    f"return s.{name}, s.{opening_hours}"
        else:
            query = f"match (s) where s.site_id='{site_id}' and s.mode='{version}' " \
                    f"and toLower('{shop}') in s.synonyms return s.{name}, s.{opening_hours}"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                shop, opening_hours = result[0], result[1]
                if shop and opening_hours:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, shop=shop, opening_hours=opening_hours)
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up
        else:
            subtopic_2 = 'answer-not-found'
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, shop=shop)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up


    elif intent == "shopinformation_contact":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None
        phase, shop = get_first_entity(slots.get("phase")), get_first_entity(slots.get("shop"))
        if phase:
            query = f"match (s)-[:At]->(floor)-[:At]->(phase) where s.site_id='{site_id}' and s.mode='{version}' " \
                    f"and toLower('{shop}') in s.synonyms and toLower('{phase}') in phase.synonyms " \
                    f"return s.{name}, s.contact"
        else:
            query = f"match (s) where s.site_id='{site_id}' and s.mode='{version}' " \
                    f"and toLower('{shop}') in s.synonyms return s.{name}, s.contact"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                shop, contact = result[0], result[1]
                if shop and contact:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, shop=shop, contact=contact)
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up
        else:
            subtopic_2 = 'answer-not-found'
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, shop=shop)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up


    elif intent == "shopinformation_payment":
        topic, subtopic_1, subtopic_2 = intent.split('_')[0], intent.split('_')[1], None
        phase, shop, query = get_first_entity(slots.get("phase")), get_first_entity(slots.get("shop")), None
        restaurant = get_first_entity(slots.get("restaurant"))
        if restaurant and not shop:
            shop = restaurant
        if phase:
            query = f"match (s)-[:At]->(floor)-[:At]->(phase) where s.site_id='{site_id}' and s.mode='{version}' " \
                    f"and toLower('{shop}') in s.synonyms and toLower('{phase}') in phase.synonyms " \
                    f"return s.{name}, s.contact"
        else:
            query = f"match (s) where s.site_id='{site_id}' and s.mode='{version}' " \
                    f"and toLower('{shop}') in s.synonyms return s.{name}, s.contact"
        results = session.run(query).values()
        answer = ''
        if results:
            for result in results:
                shop, contact = result[0], result[1]
                if shop and contact:
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, shop=shop, contact=contact)
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up
        else:
            subtopic_2 = 'answer-not-found'
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, shop=shop)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up


    elif intent == "navigation":
        topic, subtopic_1, subtopic_2 = 'navigation', None, None
        location_from = get_first_entity(slots.get("locationfrom"))
        location_to = get_first_entity(slots.get("locationto"))
        restaurant = get_first_entity(slots.get("restaurant"))
        shop = get_first_entity(slots.get("shop"))
        
        if restaurant:
            query = f"match (r)-[:At]->(floor)-[:At]->(phase) where " \
                    f"r.site_id='{site_id}' and r.mode='{version}' and " \
                    f"labels(r) in [['Restaurant'], ['Shop']] and " \
                    f"toLower('{restaurant}') in r.synonyms " \
                    f"return r.{name}, floor.{name}, phase.{name}, r.shop_number"
            results = session.run(query).values()
            answer = ''
            if results:
                print(results)
                for result in results:
                    restaurant, floor, phase, shop_number = result[0], result[1], result[2], result[3]
                    if restaurant and floor and phase and shop_number:
                        topic, subtopic_1 = "restaurantinformation", "address"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                      restaurant=restaurant, floor=floor, phase=phase, shop_number=shop_number)
                if answer:
                    topic_stat = [topic, subtopic_1, subtopic_2]
                    return topic_stat, answer, options, follow_up
            
        elif not restaurant and shop:
            query = f"match (r)-[:At]->(floor)-[:At]->(phase) where " \
                    f"r.site_id='{site_id}' and r.mode='{version}' and " \
                    f"labels(r) in [['Restaurant'], ['Shop']] and " \
                    f"toLower('{shop}') in r.synonyms " \
                    f"return r.{name}, floor.{name}, phase.{name}, r.shop_number"
            results = session.run(query).values()
            answer = ''
            if results:
                print(results)
                for result in results:
                    shop, floor, phase, shop_number = result[0], result[1], result[2], result[3]
                    if shop and floor and phase and shop_number:
                        topic, subtopic_1 = "shopinformation", "address"
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                      shop=shop, floor=floor, phase=phase, shop_number=shop_number)
                if answer:
                    topic_stat = [topic, subtopic_1, subtopic_2]
                    return topic_stat, answer, options, follow_up
        
        if location_from and location_to:
            # 1. find if direct path exist
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
                    # subtopic_1 = 'not-start-from-shop'
                    if start and method and end:
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                      location_from=start, method=method, location_to=end)
            if answer:
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer
            
            # 2. if no direct path, find path starting from shop or restaurant
            query = f"match p=(s)-[:GetAccessTo*1..2]->(e) match (s)-[:At]->(floor)-[:At]->(phase) where " \
                    f"labels(s) in [['Shop'], ['Restaurant']] and " \
                    f"toLower('{location_from}') in phase.synonyms " \
                    f"and toLower('{location_to}') in e.synonyms " \
                    f"and e.site_id='{site_id}' and e.mode='{version}' " \
                    f"return p, phase.{name}, floor.{name}, s.{name}, s.shop_number, e.{name} " \
                    f"ORDER BY length(p)"
            results = session.run(query).values()
            answer = ''
            if results:
                result = results[0]  # get result with the shortest path
                result_path, phase, floor, shop_name, shop_number, location_to = \
                    result[0], result[1], result[2], result[3], result[4], result[5]
                paths = list(result_path.__dict__.get('_relationships'))  # retrieve paths from path object
                for path in paths:
                    relationship_info = path.__dict__.get('_properties')
                    method = relationship_info.get(method)
                    if phase and floor and shop_name and shop_number:
                        subtopic_1 = 'start-from-shop'
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                      phase=phase, floor=floor, shop_number=shop_number,
                                                      shop=shop_name, method=method, location_to=location_to)
            if answer:
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer
            
            else:
                subtopic_1 = 'answer-not-found'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, location_to=location_to)
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer

        elif not location_from and location_to:
            # 1. try to get starting location and direct path
            query = f"match p=(s)-[:GetAccessTo*1..2]->(e) where " \
                    f"not labels(s) in [['Shop'], ['Restaurant']] and " \
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
                    # subtopic_1 = 'not-start-from-shop'
                    if start and method and end:
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                      location_from=start, method=method, location_to=end)
            if answer:
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer
            
            # 2. try to find path start from shop or restaurant:
            query = f"match p=(s)-[:GetAccessTo*1..2]->(e) match (s)-[:At]->(floor)-[:At]->(phase) where " \
                    f"labels(s) in [['Shop'], ['Restaurant']] and " \
                    f"toLower('{location_to}') in e.synonyms " \
                    f"and e.site_id='{site_id}' and e.mode='{version}' " \
                    f"return p, phase.{name}, floor.{name}, s.{name}, s.shop_number, e.{name} ORDER BY length(p)"
            results = session.run(query).values()
            answer = ''
            if results:
                result = results[0]  # get result with the shortest path
                result_path, phase, floor, shop_name, shop_number, location_to = \
                    result[0], result[1], result[2], result[3], result[4], result[5]
                paths = list(result_path.__dict__.get('_relationships'))  # retrieve paths from path object
                for path in paths:
                    relationship_info = path.__dict__.get('_properties')
                    method = relationship_info.get(method)
                    if phase and floor and shop_name and shop_number and location_to:
                        subtopic_1 = 'start-from-shop'
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                      phase=phase, floor=floor, shop_number=shop_number,
                                                      shop=shop_name, method=method, location_to=location_to)
            if answer:
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer
            
            # 3. if locationto is shop or restaurant, change intent
            query = f"match (n) where toLower('{location_to}') in n.synonyms and " \
                    f"n.site_id='{site_id}' and n.mode='{version}' and " \
                    f"labels(n) in [['Shop'], ['Restaurant']] " \
                    f"return labels(n)"
            results = session.run(query).values()
            answer = ''
            if results:
                label = results[0]  # label = [['label]]
                if ["Shop"] in label:
                    topic, subtopic_1, subtopic_2 = "shopinformation", "address", None
                    query = f"match (s)-[:At]->(floor)-[:At]->(phase) where " \
                            f"s.site_id='{site_id}' and s.mode='{version}' " \
                            f"and toLower('{location_to}') in s.synonyms " \
                            f"return s.{name}, floor.{name}, phase.{name}, s.shop_number"
                    results = session.run(query).values()
                    if results:
                        for result in results:
                            shop_name, floor, phase, shop_number = result[0], result[1], result[2], result[3]
                            if shop_name and floor and phase and shop_number:
                                answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                              shop_name=shop_name, floor=floor, phase=phase, shop_number=shop_number)
                    if answer:
                        topic_stat = [topic, subtopic_1, subtopic_2]
                        return topic_stat, answer

                elif ["Restaurant"] in label:
                    topic, subtopic_1, subtopic_2 = "restaurantinformation", "address", None
                    query = f"match (r)-[:At]->(floor)-[:At]->(phase) where " \
                            f"r.site_id='{site_id}' and r.mode='{version}' " \
                            f"and toLower('{location_to}') in r.synonyms " \
                            f"return r.{name}, floor.{name}, phase.{name}, r.shop_number"
                    results = session.run(query).values()
                    if results:
                        for result in results:
                            restaurant, floor, phase, shop_number = result[0], result[1], result[2], result[3]
                            if restaurant and floor and phase and shop_number:
                                answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                              restaurant=restaurant, floor=floor, phase=phase, shop_number=shop_number)
                    if answer:
                        topic_stat = [topic, subtopic_1, subtopic_2]
                        return topic_stat, answer

            if not answer:
                topic = "navigation"
                subtopic_1 = "answer-not-found"
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, location_to=location_to) 
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer
    
        anwer = ''
        if not answer:
            topic = "navigation"
            subtopic_1 = 'answer-not-found'
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
    

    elif intent == "choose_from_options":
        print(slots)
        topic, subtopic_1, subtopic_2 = None, None, None

        foodtype = None
        shoptype = None
        shopproduct = None
        facility = None

        foodtype_list = slots.get("foodtype")
        if foodtype_list:
            foodtype = foodtype_list[-1]

        restaurant_list = slots.get("restaurant")
        if restaurant_list:
            restaurant = restaurant_list[-1]

        shoptype_list = slots.get("shoptype")
        if shoptype_list:
            shoptype = shoptype_list[-1]

        shopproduct_list = slots.get('shopproduct')
        if shopproduct_list:
            shopproduct = shopproduct_list[-1]
        
        shop_list = slots.get("shop")
        if shop_list:
            shop = shop_list[-1]
        
        facility_list = slots.get("facility")
        if facility_list:
            facility = facility_list[-1]
        
        answer = ''

        if foodtype:
            print(foodtype)
            query = f"match (f)<-[:PartOf]-(n)-[:At]->(floor)-[:At]->(phase) where " \
                    f"n.site_id='{site_id}' and n.mode='{version}' and " \
                    f"toLower('{foodtype}') in f.synonyms and " \
                    f"labels(f) in [['RestaurantType'], ['Restaurant']] and " \
                    f"toLower('{restaurant}') in n.synonyms " \
                    f"return n.{name}, floor.{name}, phase.{name}, n.shop_number, n.{opening_hours}, n.contact"
            print(query)
            results = session.run(query).values()
            print(results)
            results = results[0]
            restaurant, floor, phase, shop_number, opening_hours, contact = \
                results[0], results[1], results[2], results[3], results[4], results[5]
            
            if restaurant and floor and phase and shop_number and opening_hours and contact:
                topic = "restaurantinformation"
                subtopic_1 = "address-openinghours-contact"
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                        restaurant=restaurant, phase=phase, floor=floor,
                                        shop_number=shop_number, opening_hours=opening_hours, contact=contact)
            
            elif restaurant and floor and phase and shop_number and not opening_hours and not contact:
                topic = "restaurantinformation"
                subtopic_1 = "address"
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                        restaurant=restaurant, phase=phase, floor=floor, shop_number=shop_number)
            
            elif restaurant and floor and phase and shop_number and opening_hours and not contact:
                topic = "restaurantinformation"
                subtopic_1 = "address-openinghours"
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                        restaurant=restaurant, phase=phase, floor=floor,
                                        shop_number=shop_number, opening_hours=opening_hours)
            
            elif restaurant and floor and phase and shop_number and not opening_hours and contact:
                topic = "restaurantinformation"
                subtopic_1 = "address-contact"
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                        restaurant=restaurant, phase=phase, floor=floor,
                                        shop_number=shop_number, contact=contact)
            
            if answer:
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
            else:
                topic, subtopic_1, subtopic_2 = 'restaurantinformation', 'address', 'answer-not-found'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
        

        elif shoptype or shopproduct:
            print(shoptype, shopproduct)
            if shopproduct:
                p = shopproduct
            elif shoptype:
                p = shoptype

            query = f"match (s)<-[:PartOf]-(n)-[:At]->(floor)-[:At]->(phase) where " \
                    f"n.site_id='{site_id}' and n.mode='{version}' and " \
                    f"toLower('{p}') in s.synonyms and " \
                    f"toLower('{shop}') in n.synonyms " \
                    f"return n.{name}, floor.{name}, phase.{name}, n.shop_number, n.{opening_hours}, n.contact"
            
            results = session.run(query).values()
            results = results[0]
            shop, floor, phase, shop_number, opening_hours, contact = \
                results[0], results[1], results[2], results[3], results[4], results[5]
            
            if shop and floor and phase and shop_number and opening_hours and contact:
                topic = "shopinformation"
                subtopic_1 = "address-openinghours-contact"
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                        shop=shop, phase=phase, floor=floor,
                                        shop_number=shop_number, opening_hours=opening_hours, contact=contact)
            
            elif shop and floor and phase and shop_number and not opening_hours and not contact:
                topic = "shopinformation"
                subtopic_1 = "address"
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                        shop=shop, phase=phase, floor=floor, shop_number=shop_number)
        
            elif shop and floor and phase and shop_number and opening_hours and not contact:
                topic = "shopinformation"
                subtopic_1 = "address-openinghours"
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                        shop=shop, phase=phase, floor=floor,
                                        shop_number=shop_number, opening_hours=opening_hours)
            
            elif shop and floor and phase and shop_number and not opening_hours and contact:
                topic = "shopinformation"
                subtopic_1 = "address-contact"
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                        shop=shop, phase=phase, floor=floor,
                                        shop_number=shop_number, contact=contact)
            
            if answer:
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
            else:
                topic, subtopic_1, subtopic_2 = 'shopinformation', 'address', 'answer-not-found'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
    
        
        elif facility:  # facility is exactly the name of ATM
            print(facility)
            query = f'match (s)<-[:NextTo]-(n)-[:At]->(f:Floor)-[:At]->(phase:Phase) ' \
                    f'where toLower("{facility}") in n.synonyms and ' \
                    f'n.site_id="{site_id}" and n.mode="{version}" ' \
                    f'return  n.{name}, phase.{name}, s.{name}, s.shop_number, f.{name}'
            print(query)
            results = session.run(query).values()
            print(results)
            topic, subtopic_1 = "facilitiesinformation", "address"
            answer = ''
            if results:
                for result in results:
                    facility, phase, shop, shop_number, floor = result[0], result[1], result[2], result[3], result[4]
                    if facility and phase and shop and shop_number and floor: 
                        answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang,
                                                      facility=facility, phase=phase, shop=shop, shop_number=shop_number, floor=floor)
            if answer:
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up
            else:
                subtopic_2 = 'answer-not-found'
                answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
                topic_stat = [topic, subtopic_1, subtopic_2]
                return topic_stat, answer, options, follow_up


    elif intent == "get_all_restaurant_shop_information":
        
        topic, subtopic_1, subtopic_2 = None, None, None

        restaurant_shop_list = slots.get("restaurant-shop")
        
        answer = ''
        
        for restaurant_shop in restaurant_shop_list:

            restaurant_shop = restaurant_shop.lower()

            query = f"match (n)-[:At]->(floor)-[:At]->(phase) where " \
                    f"n.site_id='{site_id}' and n.mode='{version}' and " \
                    f"toLower(n.{name})='{restaurant_shop}' " \
                    f"return n.{name}, floor.{name}, phase.{name}, n.shop_number, n.{opening_hours}, n.contact"
            print(query)
            
            results = session.run(query).values()
            
            if results:
                results = results[0]
                restaurant, floor, phase, shop_number, open_hours, contact = \
                    results[0], results[1], results[2], results[3], results[4], results[5]
                
                if restaurant and floor and phase and shop_number and open_hours and contact:
                    topic = "restaurantinformation"
                    subtopic_1 = "address-openinghours-contact"
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                  restaurant=restaurant, phase=phase, floor=floor,
                                                  shop_number=shop_number, opening_hours=open_hours, contact=contact)
                
                elif restaurant and floor and phase and shop_number and not open_hours and not contact:
                    topic = "restaurantinformation"
                    subtopic_1 = "address"
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                  restaurant=restaurant, phase=phase, floor=floor, shop_number=shop_number)
                
                elif restaurant and floor and phase and shop_number and open_hours and not contact:
                    topic = "restaurantinformation"
                    subtopic_1 = "address-openinghours"
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                  restaurant=restaurant, phase=phase, floor=floor,
                                                  shop_number=shop_number, opening_hours=open_hours)
                
                elif restaurant and floor and phase and shop_number and not open_hours and contact:
                    topic = "restaurantinformation"
                    subtopic_1 = "address-contact"
                    answer += " " + generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang, 
                                                  restaurant=restaurant, phase=phase, floor=floor,
                                                  shop_number=shop_number, contact=contact)
        
        if answer:
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up
        
        else:
            topic = "restaurantinformation"
            subtopic_1 = 'address'
            subtopic_2 = 'answer-not-found'
            answer += generate_text(site_id, topic, subtopic_1, subtopic_2, template_db, lang)
            topic_stat = [topic, subtopic_1, subtopic_2]
            return topic_stat, answer, options, follow_up