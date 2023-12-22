from paddlenlp import Taskflow
 
 
schema_hkic = [
    'location',
    'date',
    'occupation',
    'nationality',
    'requirement',
    'duration',
    'name',
    'service',
    'document_replacement',
    'government_office',
    'appointment_date',
    'travel'
]
 
schema_visit = [
    'location',
    'date',
    'occupation',
    'nationality',
    'requirement',
    'duration'
    'name'
]
 
ie_hkic = Taskflow('information_extraction',
              schema=schema_hkic,
              model='uie-m-base',
              schema_lang="en",
              device="cpu")
 
ie_visit = Taskflow('information_extraction',
              schema=schema_visit,
              model='uie-m-base',
              schema_lang="en",
              device="cpu")
 
hkic_texts = [
    "你好，請問在海外人仕想回來更換智能身份証，程序是怎樣？但更換後亦沒有時間在港等候領取新身份証，可以怎樣呢？ ",
    "你好，本人現居海外，持有有效香港身份證，因疫情最近才決定回港換領新的智能身份證，訂了機票後，才發現上綱預約最早辦理日期為明年二月六日。本人會於今年十一月廿八至一月九日逗留香港，想請問有沒有途徑可在期間辦理申請？",
    "你好， 本人會於2023年11月22日從荷蘭回港一個月更換智能身份證，但在你們的網上預約系統發現預約期全部滿，我一定要在此一個月回港期辦理更換手續，我應該怎麼辦？",
    "本人和太太由於久居海外，但已定於今年 12月9日 回港 直至 1 月 21 日。本來打算在網上預約換證，但發現所有換證地點的日子卻顯示已滿，如此我們如何換證？  而且我們回港期間，更會去其他國家幾天旅行，這樣的話會不會影響我們出入境？ 目前我會所持有的身份證簽發日期分別為 15-11-07 和  24-12-07 本人出生日期為  18-04-1980  我太太是   10-09-1979 ",
    "你好，請問持有舊款成人身份證需要在到港第2天立即去入境處換領嗎？沒有到港第2天去換領身份證會罰款嗎？但我在網上預約日期最快在2024年2月，我是在2023年12月到港的， 謝謝",
    "您好！請問一下，身份證需要換領了，最遲什麼時候來換領？怎麼預約？現在能預約嗎？謝謝 ",
    "我打算在11月27到12月11之間回港想請問如果在網站上預約時間已經到2024年我是否可以在辦事處等候如無預約？或是必須預約 ? ",
    "入境處負責同事你好: 本人一直在國內,因錯過全港市民換領身份證計劃, 最近打算回港更換新智能身份證,但有以下問題: 1/各中心延長服務時間基本上今年內都已約滿,是不是我不可以直接去現場申請換證? 2/如打算去申請換證,我是否用護照來出入香港,因為換證有處理時間,但期間我需要出入中港兩地. 請回覆.謝謝. "
]
 
visit_texts = [
    "Hi Sir/Madam My family is visiting Hong Kong 2nd week of November for a holiday. I would like to bring my helper along, she is Myanmmar  citizen. May I know which visa to apply for her? Is there one for domestic helper where I will be her guarantor? ",
    "Visit visa\n Hello\n Sir\n What are the documents required for Hong Kong visa? ",
    "Respected sir/maam, I would like to know about the visa requirements for the United state green card with a nepali passport holder. What should i have to do to visit hongkong for the visit visa?",
    "My boyfriend holds a North Macedonian passport and we are intending to visit Hong Kong in November. We are intending to be in Hong Kong for 11 days, 4 days in Macau and 3 days again in Hong Kong. We just wanted to double check that that is allowed within the 14 days of visa-free period that Macedonians have to be in Hong Kong, as a total of 11+3 days, or whether the 14 days will just reset if he leaves Hong Kong and goes to Macau for a few days, and then re-enters Hong Kong?",
    "Dear sir/Madam\n My inquiry is about ILR UK holder UK resident need Hong Kong visa to travel from UK or can visit without any visa permission as Nationality is not Brittash \nLooking forward for your kind reply",
    "I am a holder of a Lithuanian passport and I’m Not clear it I require a visa to HK for purpose of visiting as holiday for 10 days. Information online is very confusing as it states visa free period for visit not exceeding 90 days and then directly under it states always requires a visa.\nCan you please advise whether I need a visa and share a link with me to apply for it should I require. ",
    "my name is Andrei and I intend to travel to HK. I would like to know what conditions must be met by a citizen of Kenya (residence in Qatar) in order to be able to travel for tourist purposes to the territory of Hong Kong"
]
 
 
def extract_keywrods(texts, model, threshold=0.8):
    for i, text in enumerate(texts):
        results = model(text)
        print("email index: ", i+1)
        print("email content: ", text)
        for result in results:
            if result:
                keyword_types = list(result.keys())
                for keyword_type in keyword_types:
                    print("keyword_type: ", keyword_type)
                    for keywords in list(result.values()):
                        for keyword_info in keywords:
                            if keyword_info.get("probability") > threshold:
                                print("     keyword: ", keyword_info.get("text"))
                                print("     confidence: ", keyword_info.get("probability"))
        print()
 
 
if __name__ == "__main__":
    extract_keywrods(hkic_texts, ie_hkic)
    # extract_keywrods(visit_texts, ie_visit)