import jieba
from configuration import Config

def segmentate_sentence(sentence):
    seg_list = list(jieba.cut(sentence))
    seg_list = [i.strip() for i in seg_list]
    return " ".join(seg_list)


def extract_key_word(sentence, stopwords):
    seg_list = list(jieba.cut(sentence))
    seg_list = [i.strip() for i in seg_list]
    for word in seg_list:
        if word.lower() in stopwords:
            seg_list.remove(word)
    return seg_list

if __name__ == "__main__":
    str = "中银柜员机喺邊"
    print(segmentate_sentence(str))