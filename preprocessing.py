import re, string
import nltk
from time import time
import opencc

def download_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
converter = opencc.OpenCC('t2s.json')

cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "i'd": "i would",
  "i'd've": "i would have",
  "i'll": "i will",
  "i'll've": "i will have",
  "i'm": "i am",
  "i've": "i have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

cantonese_speaking_stop_words = ['哦',
                                 '哋',
                                 '咋',
                                 '噃',
                                 '嚿',
                                 '㗎',
                                 '嘞',
                                 '吖',
                                 '喎',
                                 '囉',
                                 '欸',
                                 '嗱',
                                 '哎',
                                 '吔',
                                 '嘥',
                                 '啩',
                                 '吓',
                                 '揸',
                                 '唉',
                                 '誒',
                                 '呢',
                                 '嚟架',
                                 '嚟㗎',
                                 '嚟',
                                 '咗',
                                 '呀',
                                 '啊',
                                 '咁']

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text.lower())


# need to remove 's after expand contractions??
def remove_s(text):
    return text.replace("'s", "")


def remove_cantonese_speaking_stop_words(text):
    for w in cantonese_speaking_stop_words:
        text = text.replace(w, "")
    return text


def fix_typo(text):
    with open('typo.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            wrong, correct = line.split('\t')[0], line.split('\t')[1]
            text_corrected = text.replace(wrong, correct) if wrong in text else text
            text = text_corrected   
    return text


def preprocess(text, lang=None):
    text = text.lower()
    text = text.strip()
    text = text.replace("\n", "")
    # if lang == "en":
    text = expandContractions(text)  # we'll --> we will
    text = remove_s(text)  # girl's --> girl 
    # elif lang in ["zh-hk", "zh-cn"]:
    text = remove_cantonese_speaking_stop_words(text)
    text = fix_typo(text)
    return text


def node_synonyms_preprocess(text):
    text = text.lower()
    text = converter.convert(text)
    text = text.strip()
    text = expandContractions(text)  # we'll --> we will
    text = remove_s(text)  # girl's --> girl 
    text = remove_cantonese_speaking_stop_words(text)
    text = fix_typo(text)
    text = re.sub(r'[^\w\s+]', '', text)  # remove punctuation
    text = lemmatizer.lemmatize(text)
    text = " ".join(lemmatizer.lemmatize(i) for i in text.split())
    return text


if __name__=="__main__":

    # print(node_synonyms_preprocess("利華超級三文治"))
    print(preprocess("Olympian City这么多期的礼宾处电话几号", "zh"))