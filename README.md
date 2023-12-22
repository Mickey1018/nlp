# NLP pipeline & APIs

## Download docker image
```
$docker load < sino-nlpv2.0-rc.4-patch1.tar.gz
```

### Installation with Docker Compose (recommand)
Wait for 30 minutes after running the following command for downloading the required models.
The port number is 8090 by default. However you can edit the docker compose file to configurate the port number for the service.
```
$docker-compose up -d -f docker-nlp_v20rc4-patch1.yml
```

You may choose 8090, 7687, 7473 as your port number as configured in docker-compose.yml. However you can change port number as you want, as long as the port you choosed is consistant with the one configured in docker-compose.yml


## API document
For API document, please refer to the following link for details.

http://gitlab.vabot.org/cli/art342cp_sino/-/wikis/Internal-API-NLP

## Quick Start
Send the following json data with Post request to 
```https://<your_server_ip>:<your_port>/api/nlp_pipeline```
```json
{
    "language": "zh-hk",
    "site_id": "oc",
    "question": "奧海城OC STEM Lab創意工作室喺邊？",
    "version": "editing",
    "history": []
}
```

The expected results are as follow:
```json
{
    "success": true,
    "data": {
        "language": "zh-hk",
        "question": {
            "text": "奧海城OC STEM Lab創意工作室喺邊？",
            "text_seg": "",
            "keyword": [],
            "topic_detected": [
                "ShopInformation",
                "Address"
            ],
            "confidence_score": 0.9999890327453613,
            "entity_detected": [
                {
                    "type": "Mall",
                    "start_position": 0,
                    "end_position": 2,
                    "text": "奥海城",
                    "confidence_score": 0.9996101632078545
                },
                {
                    "type": "Shop",
                    "start_position": 3,
                    "end_position": 10,
                    "text": "ocstemlab创意工作室",
                    "confidence_score": 0.9997906831130816
                }
            ]
        },
        "answer": {
            "type": "direct",
            "text": "OC STEM Lab 創意工作室位於奧海城2期1/F, 在135-136號舖渣打銀行附近。",
            "options": [],
            "confidence_score": 1.0
        },
        "version": "editing",
        "site_id": "oc"
    }
}
```

