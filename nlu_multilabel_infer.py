import argparse
import functools
import os

import paddle
import paddle.nn.functional as F
from paddle.io import BatchSampler, DataLoader
from utils import preprocess_function, read_local_dataset, read_batch_dataset, read_one_data

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.utils.log import logger


def load_dict(path):
    label2id = {}
    with open(path, "r", encoding="utf-8") as f:
        labels = f.readlines()
        for idx, label in enumerate(labels):
            label = label.strip()
            label2id[label] = idx
    id2label = dict([(idx, label) for label, idx in label2id.items()])
    return label2id, id2label


@paddle.no_grad()
def multilable_classification_infer(model, tokenizer, id2intent, device='cpu', 
    max_seq_length=1024, batch_size=16, excel_dir='excel', excel_file='dataset.xlsx', use_batch=True,
    text=None):
    """
    Predicts the data labels.
    """
    paddle.set_device(device)
    
    # get intent label
    # label_list = []
    # label_path = os.path.join(project_path, intent_dir, intent_file)
    # with open(label_path, "r", encoding="utf-8") as f:
    #     for i, line in enumerate(f):
    #         label_list.append(line.strip())

    # create dir for predict v.s. put predict data into excel?
    # use other functions instead of read_local_dataset
    if use_batch:
        if text:
            raise ValueError("text should be None")
        else:
            data_ds = load_dataset(
                read_batch_dataset, 
                excel_file=os.path.join(project_path, excel_dir, excel_file), 
                lazy=False
            )
    
    else:
        if not text:
            raise ValueError("text should not be None")
        else:
            data_ds = load_dataset(
                read_one_data,
                text=text,
                lazy=False
            )

    # keep this
    trans_func = functools.partial(
        preprocess_function,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        is_test=True,
    )
    
    data_ds = data_ds.map(trans_func)

    # batchify dataset
    collate_fn = DataCollatorWithPadding(tokenizer)
    data_batch_sampler = BatchSampler(
        data_ds, 
        batch_size=batch_size, 
        shuffle=False
    )

    data_data_loader = DataLoader(
        dataset=data_ds, 
        batch_sampler=data_batch_sampler, 
        collate_fn=collate_fn
    )

    results = []
    model.eval()
    for batch in data_data_loader:
        logits = model(**batch)
        probs = F.sigmoid(logits).numpy()
        # iterate singal result
        for prob in probs:
            temp_result = []
            for i, p in enumerate(prob):
                temp_dict = {}
                if p > 0.5:
                    temp_dict["topic"] = id2intent[i]
                    temp_dict["con"] = p
                    temp_result.append(temp_dict)
            results.append(temp_result)

    print(results)
    print(results[0])

    # write as excel?
    # with open(args.output_file, "w", encoding="utf-8") as f:
    #     f.write("text" + "\t" + "label" + "\n")
    #     for d, result in zip(data_ds.data, results):
    #         label = [label_list[r] for r in result]
    #         f.write(d["sentence"] + "\t" + ", ".join(label) + "\n")
    # logger.info("Prediction results save in {}.".format(args.output_file))

    return results


if __name__ == "__main__":
    project_path='data/project/immd'
    checkpoint_dir='ckpt'
    intent_path = os.path.join(project_path, 'intent', 'intent_label.txt')
    intent2id, id2intent = load_dict(intent_path)

    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(project_path, checkpoint_dir))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(project_path, checkpoint_dir))

    multilable_classification_infer(
        model, 
        tokenizer,
        id2intent=id2intent,
        use_batch=False,
        text='Dear Sir,我德國國籍的同事，已在香港連續工作了七年，今年已踏入第8年，這七年也是在香港納稅的。  請問他是否能申請香港居民身分證。  如要申請，請問有什麼表格要填？  和申請程序？  能否在網上申請？謝謝！'    
    )
