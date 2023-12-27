import dill as pickle
import joblib
from paddlenlp.transformers import AutoModel, AutoTokenizer
from configuration import Config
from nlu_infer import ErnieTokenizer, ErnieModel, ErnieMTokenizer, ErnieMModel
from preprocessing import download_nltk
from paddlenlp import Taskflow

download_nltk()

# senta = Taskflow("sentiment_analysis", device_id=-1)
# similarity_en = Taskflow("text_similarity", model="rocketqa-medium-cross-encoder", device_id=-1)
# similarity_zh = Taskflow("text_similarity", model="simbert-base-chinese", device_id=-1)

# sim_transformer_zh = "ernie-3.0-medium-zh"
# sim_transformer_en = "ernie-2.0-base-en"
# nlu_transformer_zh = "ernie-3.0-xbase-zh"
# nlu_transformer_en = "ernie-2.0-base-en"
nlu_transformer_multi = "ernie-m-base"

# nlu_zh_tokenizer = ErnieTokenizer.from_pretrained(nlu_transformer_zh)
# nlu_en_tokenizer = ErnieTokenizer.from_pretrained(nlu_transformer_en)
nlu_multi_tokenizer = ErnieMTokenizer.from_pretrained(nlu_transformer_multi)

# ernie_zh = ErnieModel.from_pretrained(nlu_transformer_zh)
# ernie_en = ErnieModel.from_pretrained(nlu_transformer_en)
ernie_multi = ErnieMModel.from_pretrained(nlu_transformer_multi)
