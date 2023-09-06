#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import os
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import datasets

import utils
import json
import sys

import numpy as np
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from trainer_seq2seq import Seq2SeqTrainer
from arguments import ModelArguments, DataArguments
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

train_file = 'datasets/annotation_v2/train.json'
source_cut_length = 1700
target_cut_length = 300

def setup_dataset():
    data_files = {}
    data_files["train"] = train_file
    extension = train_file.split(".")[-1]
    
    raw_datasets = load_dataset(
        extension,
        data_files=data_files
    )
    return raw_datasets

def load_training_dataset(raw_datasets, tokenizer, name="train"):
    column_names = raw_datasets[name].column_names
    prompt_column = 'input'
    response_column = 'output'
    
    def preprocess_function_train(examples):
        global source_len, target_len
        model_inputs = {
            "source_len": [],
            "target_len": [],
        }
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]
                prompt = query

                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                model_inputs["source_len"].append(len(a_ids))
                model_inputs["target_len"].append(len(b_ids))

        return model_inputs

    train_dataset = raw_datasets[name]
    train_dataset = train_dataset.map(
        preprocess_function_train,
        batched=True,
        num_proc=32,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )
    return train_dataset

tokenizer = AutoTokenizer.from_pretrained(
    'chatglm6b_v10',
    trust_remote_code=True
)
raw_datasets = setup_dataset()
train_dataset = load_training_dataset(raw_datasets, tokenizer)

source_len = np.array(train_dataset['source_len'])
target_len = np.array(train_dataset['target_len'])
print('max source len:', np.max(source_len))
print('max target len:', np.max(target_len))
print('min source len:', np.min(source_len))
print('min target len:', np.min(target_len))
print('mean source len:', np.mean(source_len))
print('mean target len:', np.mean(target_len))
print('source cut num:', np.sum(source_len > source_cut_length))
print('target cut num:', np.sum(target_len > target_cut_length))
