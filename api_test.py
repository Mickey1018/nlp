import requests
import json, os
from configuration import Config

def test_pipeline(endpoint, project=None, text=None, training_id=None):
    url = "http://localhost:{0}/api/{1}".format(Config.port, endpoint)

    if endpoint == "classify_email_and_get_keywords":

        json_data = {
            "subject": "",
            "content": text,
            "project": project
        }
        print(json_data)

    elif endpoint == "train_email_classifier":
        json_data = {
            "project_path": os.path.join('data', 'project', project)
        }
    
    elif endpoint == "get_training_job_status":
        json_data = {
            "training_id": training_id
        }
    
    elif endpoint == "get_intent_and_keyword_type":
        json_data = {
            "project_path": os.path.join('data', 'project', project)
        }

    response = requests.post(url, json=json_data, verify=False)
    print(response.status_code)  # Print the HTTP status code
    print(response.text)  # Print the response body


if __name__ == "__main__":
    # test_pipeline(
    #     endpoint="train_email_classifier",
    #     project='immd'
    # )

    test_pipeline(
        endpoint="classify_email_and_get_keywords",
        project='immd',
        text="敬啟者：您好！關於線下提交永居申請事宜，本人有如下問題想諮詢貴處：1.提交永居的話，是否需要提交給居留權組？地點是否在灣仔辦公室？在哪一層樓的哪個窗口交資料？2.是否需要提前預約？如需，如何預約？3.有哪些文件是在交資料時必須提交的？哪些是可以候補的？4.交完資料後，是否會有回執，證明本人的資料是交過的？5.如果線下遞交，是否一定要申請人本人親自前往？煩請閣下就以上問題給予指導，不勝感激！"
    )

    # test_pipeline(
    #     endpoint="get_training_job_status",
    #     training_id="id-0001"
    # )

    # test_pipeline(
    #     endpoint='get_intent_and_keyword_type',
    #     project='immd'
    # )


