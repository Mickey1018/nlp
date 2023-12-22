import re
from symbol import except_clause
# This is a python script to postprocess text for tts
# 1. "/F" --> "層", "floor"
# 2. deep link --> don't send long link
# 3. no space between Chinese when pass to tts
# 4. No. 70中間不要出現space
# 5. 還發現一個問題 哈哈哈 就是10a.m - 4p.m 中間不能有space 哦

def postprocess(text, lang):

    if not text:
        return None
    
    # remove new line character
    text = str(text)
    text = text.replace("\n", "")
    
    # remove url
    url_pattern = "(:\s*)?((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*(\s*)?"
    text = re.sub(url_pattern, "", text) if lang != 'en' else re.sub(url_pattern, " ", text)

    # "/F" --> "層", "floor"
    if lang == "zh-hk":
        p = re.compile(r'([a-zA-Z]+)\/F')
        text = p.sub("\\1樓層", text)
        p = re.compile(r'([0-9]+)\/F')
        text = p.sub("\\1樓", text)
    elif lang == "zh-cn":
        p = re.compile(r'([a-zA-Z0-9]*)\/F')
        text = p.sub("\\1层", text)
    elif lang == "en":
        p = re.compile(r'([a-zA-Z0-9]*)\/F')
        text = p.sub("\\1 floor", text)
    
    # change " to '
    p = re.compile(r'(["]+)')
    text = p.sub("'", text)

    # No. 70中間不要出現space
    pattern_1 = re.compile(r"([Nn]o\.)\s+(\w+)\s?")
    text = pattern_1.sub("\\1\\2", text)

    # 10a.m - 4p.m 中間不能有space
    pattern_2 = re.compile(r"(\d*:?\d*[APap]\.?[Mm]\.?)\s+(-)\s+(\d*:?\d*[APap]\.?[Mm]\.?)")
    text = pattern_2.sub("\\1\\2\\3", text)

    return text

 
if __name__ == "__main__":
    text = "For information on member discounts, please refer to the following website: https://www.olympiancity.com.hk/tc/Promotion"
    # text = 'UG/F"'
    # text = "no. 263C"
    # text = "10a.m - 4p.m"
    result = postprocess(text, "zh-hk")
    print('original: ', text)
    print('postprocessed: ', result)