# NLP API for Email Classification

## 1. Classify Email and get Keywords

<br>

### 1.1 Endpoint
https://server:8080/api/classify_email_and_get_keywords

<br>

### 1.2 Input Parameters
```json
{
    "subject": ""
    "content": "你好，本人現居海外，持有有效香港身份證，因疫情最近才決定回港換領新的智能身份證，訂了機票後，才發現上綱預約最早辦理日期為明年二月六日。本人會於今年十一月廿八至一月九日逗留香港，想請問有沒有途徑可在期間辦理申請？"
}
```

_Input parameter description:_
| Parameter | Type | Description |
| ------ | ------ | ------ |
| subject | string | Subject of Email |
| content | string | Content of Email |

<br>

### 1.3 Return Value
```json
{
    "success": true,
    "data": {
        "catagory": [
            {
                "topic": "Renew_HKID",
                "con": 0.93
            }
        ],
        "keywords": [
            { 
                "start": 45,
                "end": 50,
                "text":  "明年二月六日",
                "type": "辦理日期",
                "con": 0.88
            }
        ]
    }
}
```
_Return parameter description: _
| Parameter | Type | Description |
| ------ | ------ | ------ |
| category > topic | string | Topic detected in email |
| category > con | float | Confident level of topic |
| keywords > start | int | Start position of the keyword |
| keywords > end | int | End position of the keyword |
| keywords > text | string | Keyword detected |
| keywords > type | string | Type of the keyword |
| keywords > con | float | Confidence level of the keyword |
| version | string | ["editing", "published"] |

_Return message when an error occurs:_
```json
{
    "success": false,
    "error": {
        "reason": "Some errors occurred" 
    }
}
```
_Return parameter description: _
| Parameter | Type | Description |
| ------ | ------ | ------ |
| error > reason | string | Reason of failure |

<br>

## 2. Train Email Classifier

<br>

### 2.1 Endpoint
https://server:8080/api/train_email_classifier

<br>

### 2.2 Input Parameters
```json
{
    "dataset_path": "/path/to/train/dataset.xlsx"
}
```
_Input parameter description:_
| Parameter | Type | Description |
| ------ | ------ | ------ |
| dataset_path | string | path to training dataset, it should be a file instead of directory |

<br>

### 2.3 Return Value
_Return message when successful:_
```json
{
    "success": true
}
```
_Return message when an error occurs:_
```json
{
    "success": false,
    "error": {
        "reason": "Invalid format in training dataset" 
    }
}
```
_Return parameter description:_
| Parameter | Type | Description |
| ------ | ------ | ------ |
| error > reason | string | Reason of failure |

<br>