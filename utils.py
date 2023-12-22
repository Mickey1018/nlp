from pathlib import Path
import os
import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.utils.log import logger
import re
import pandas as pd


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evaluates model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """

    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        labels = batch.pop("labels")
        logits = model(**batch)
        loss = criterion(logits, labels)
        probs = F.sigmoid(logits)
        losses.append(loss.numpy())
        metric.update(probs, labels)

    micro_f1_score, macro_f1_score = metric.accumulate()
    logger.info(
        "eval loss: %.5f, micro f1 score: %.5f, macro f1 score: %.5f"
        % (np.mean(losses), micro_f1_score, macro_f1_score)
    )
    model.train()
    metric.reset()

    return micro_f1_score, macro_f1_score


def preprocess_function(examples, tokenizer, max_seq_length, label_nums=None, is_test=False):
    """
    Builds model inputs from a sequence for sequence classification tasks
    by concatenating and adding special tokens.

    Args:
        examples(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_length(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        label_nums(obj:`int`): The number of the labels.
    Returns:
        result(obj:`dict`): The preprocessed data including input_ids, token_type_ids, labels.
    """
    result = tokenizer(text=examples["sentence"], max_seq_len=max_seq_length)
    
    # One-Hot label
    if not is_test:
        result["labels"] = [float(1) if i in examples["label"] else float(0) for i in range(label_nums)]
    return result


def read_local_dataset(dir, label_list):
    # labels are separated with space in text file
    with open(os.path.join(dir, "label"), "r", encoding="utf-8") as f_label, \
        open(os.path.join(dir, "seq.in"), "r", encoding="utf-8") as f_text:
        for label, text in zip(f_label, f_text):
            labels = [label_list[l.strip()] for l in label.split(" ")]
            text = remove_space_between_chinese_characters(text)
            yield {"sentence": text, "label": labels}


def read_batch_dataset(excel_file):
    df = pd.read_excel(excel_file)
    for i in range(len(df)):
        text = df.loc[i, "text"]
        text = remove_space_between_chinese_characters(text)
        yield {"sentence": text}


def read_one_data(text):
    yield {"sentence": text}


def list_child_directories(path):
    child_directories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return child_directories


def has_child_directories(path, desired_child_directories):
    child_dirs = list_child_directories(path)
    print(child_dirs)
    for desired_child_dir in desired_child_directories:
        if not desired_child_dir in child_dirs:
            return False
    return True


def remove_space_between_chinese_characters(text):
    has_space = re.search(r'([\u4e00-\u9fff]+)\s+([\u4e00-\u9fff]+)', text)
    if has_space:
        end = has_space.end()
        left_part = text[:end]
        right_part = text[end:]
        left_part = re.sub(r'([\u4e00-\u9fff]+)\s+([\u4e00-\u9fff]+)', r'\1\2', left_part)
        return remove_space_between_chinese_characters(left_part + right_part)
    else:
        return text
    

if __name__ == "__main__":
    child_directories = list_child_directories("data/")
    print(child_directories)
    desired_directories = ['excel', 'intent', 'keyword', 'test', 'train']
    result = has_child_directories("data/project/immd", desired_directories)
    print(result)