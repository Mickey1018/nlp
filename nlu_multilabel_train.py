# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
import argparse
import functools
import os
import random
import time

import numpy as np
import paddle
from metric import MetricReport
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler
from utils import evaluate, preprocess_function, read_local_dataset

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LinearDecayWithWarmup,
)
from paddlenlp.utils.log import logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def params_saving(save_dir, device):
    with open(os.path.join(save_dir, "setting.txt"), "w") as f:
        f.writelines("------------------ start ------------------" + "\n")
        f.writelines("device" + " : " + str(device) + "\n")
        f.writelines("------------------- end -------------------")


def multilable_classification_train(
    train_id, 
    dataset_dir,
    training_start_time=None, 
    model_name="ernie-3.0-xbase-zh", 
    seed=3, 
    device="gpu", 
    label_file="intent_label.txt", 
    train_dir="train", 
    dev_dir="dev", 
    max_seq_length=1024, 
    batch_size=4, 
    learning_rate=3e-5, 
    epochs=30, 
    init_from_ckpt=None, 
    warmup_steps=0, 
    weight_decay=0.0, 
    early_stop=False, 
    early_stop_nums=3, 
    warmup=True, 
    logging_steps=5, 
    intent_dir="intent"
    ):
    """
    Training a multi label classification model
    Local dataset directory should include train.txt, dev.txt and label.txt
    choices=["ernie-1.0-large-zh-cw", "ernie-3.0-xbase-zh", "ernie-3.0-base-zh", "ernie-3.0-medium-zh", "ernie-3.0-micro-zh", "ernie-3.0-mini-zh", "ernie-3.0-nano-zh", "ernie-2.0-base-en", "ernie-2.0-large-en", "ernie-m-base", "ernie-m-large"])
    init_from_ckpt: "The path of checkpoint to be loaded."
    """
    save_dir = os.path.join(dataset_dir, 'ckpt_intent')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    params_saving(save_dir, device)
    
    set_seed(seed)
    
    paddle.set_device(device)

    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    # load and preprocess dataset
    label_list = {}
    with open(os.path.join(dataset_dir, intent_dir, label_file), "r", encoding="utf-8") as f:
        # Local dataset directory should include train.txt, dev.txt and label.txt
        for i, line in enumerate(f):
            l = line.strip()
            label_list[l] = i
    train_ds = load_dataset(
        read_local_dataset, dir=os.path.join(dataset_dir, train_dir), label_list=label_list, lazy=False
    )
    dev_ds = load_dataset(
        read_local_dataset, dir=os.path.join(dataset_dir, dev_dir), label_list=label_list, lazy=False
    )
    print(dev_ds)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    trans_func = functools.partial(
        preprocess_function, tokenizer=tokenizer, max_seq_length=max_seq_length, label_nums=len(label_list)
    )
    train_ds = train_ds.map(trans_func)
    dev_ds = dev_ds.map(trans_func)
    
    # batchify dataset
    collate_fn = DataCollatorWithPadding(tokenizer)
    
    if paddle.distributed.get_world_size() > 1:
        train_batch_sampler = DistributedBatchSampler(train_ds, batch_size=batch_size, shuffle=True)
    else:
        train_batch_sampler = BatchSampler(train_ds, batch_size=batch_size, shuffle=True)
    dev_batch_sampler = BatchSampler(dev_ds, batch_size=batch_size, shuffle=False)
    
    train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    dev_data_loader = DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=collate_fn)

    # define model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=len(label_list))
    if init_from_ckpt and os.path.isfile(init_from_ckpt):
        state_dict = paddle.load(init_from_ckpt)
        model.set_dict(state_dict)
    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * epochs
    lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_steps)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    criterion = paddle.nn.BCEWithLogitsLoss()
    metric = MetricReport()

    global_step = 0
    best_f1_score = 0
    early_stop_count = 0
    tic_train = time.time()

    for epoch in range(1, epochs + 1):

        if early_stop and early_stop_count >= early_stop_nums:
            logger.info("Early stop!")
            break

        for step, batch in enumerate(train_data_loader, start=1):

            labels = batch.pop("labels")
            logits = model(**batch)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            if warmup:
                lr_scheduler.step()
            optimizer.clear_grad()

            global_step += 1
            if global_step % logging_steps == 0 and rank == 0:
                logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, logging_steps / (time.time() - tic_train))
                )
                tic_train = time.time()

        early_stop_count += 1
        micro_f1_score, macro_f1_score = evaluate(model, criterion, metric, dev_data_loader)

        save_best_path = save_dir
        if not os.path.exists(save_best_path):
            os.makedirs(save_best_path)

        # save models
        if macro_f1_score > best_f1_score:
            early_stop_count = 0
            best_f1_score = macro_f1_score
            model._layers.save_pretrained(save_best_path)
            tokenizer.save_pretrained(save_best_path)
        logger.info("Current best macro f1 score: %.5f" % (best_f1_score))
    logger.info("Final best macro f1 score: %.5f" % (best_f1_score))
    
    now = datetime.now()
    training_end_time = now.strftime("%d/%m/%Y %H:%M:%S")

    training_result = None
    if best_f1_score >= 0.85:
        training_result = 'good'
    elif 0.7 <= best_f1_score < 0.85:
        training_result = 'acceptable'
    elif best_f1_score < 0.7:
        training_result = 'fail'

    # write final results into text file
    if not os.path.exists(os.path.join(dataset_dir, 'train_results')):
        os.makedirs(os.path.join(dataset_dir, 'train_results'))
    if not os.path.exists(os.path.join(dataset_dir, 'train_results', 'intent')):
        os.makedirs(os.path.join(dataset_dir, 'train_results', 'intent'))
    with open(os.path.join(dataset_dir, 'train_results', 'intent', 'results.txt'), 'w', encoding='utf-8') as f:
        f.write("training id: " + str(train_id) + '\n')
        # f.write("Final best macro f1 score: %.5f\n" % (best_f1_score))
        f.write("training status: " + '1 - success' + '\n')
        if training_start_time:
            f.write("training start time: " + training_start_time + '\n')
        f.write("training start time: " + training_end_time + '\n')
        f.write("training result: " + training_result + '\n')

    logger.info("Save best macro f1 text classification model in %s" % (save_dir))

    return training_result, training_end_time

if __name__ == "__main__":
    multilable_classification_train(1, "data/project/immd")
