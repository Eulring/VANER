# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import ipdb

from modeling_llama import UnmaskingLlamaForTokenClassification
from seqeval.metrics import precision_score, recall_score, f1_score

from utils_vaner import *
###

def vis(ds, idx):
    print(' '.join(ds['train'][idx]['tokens']))
    print(ds['train'][idx]['ner_tags'])


def find(ds):
    for idx, ele in enumerate(ds['train']):
        if 'linnaeus' in ele['tokens'][0]:
            if 1 in ele['ner_tags']:
                print(idx)
                ipdb.set_trace()


def load_unidev_kgmix2(kg_type):
    ret = {}
    for split_name in ['train']:
        cnt = 0
        data = []
        if kg_type == 'mt':
            for cohort in ["BC2GM","BC4CHEMD","BC5CDR-chem","BC5CDR-disease","JNLPBA","linnaeus","s800","ncbi"]:

                with open(f'./data/vaner_datacohort/{cohort}/{split_name}_df_mix2.jsonl', 'r') as reader:
                    for line in reader:
                        if cohort == "BC2GM":
                            items = parse_mt(json.loads(line), split_name, cohort, 'Gene')
                            data.extend(items)
                        elif cohort == "BC4CHEMD":
                            items = parse_mt(json.loads(line), split_name, cohort, 'Chemical')
                            data.extend(items)
                        elif cohort == "BC5CDR-chem":
                            items = parse_mt(json.loads(line), split_name, cohort, 'Chemical')
                            data.extend(items)
                        elif cohort == "BC5CDR-disease":
                            items = parse_mt(json.loads(line), split_name, cohort, 'Disease')
                            data.extend(items)
                        elif cohort == "JNLPBA":
                            items = parse_mt(json.loads(line), split_name, cohort, 'Gene')
                            data.extend(items)
                        elif cohort == "linnaeus":
                            items = parse_mt(json.loads(line), split_name, cohort, 'Species')
                            data.extend(items)
                        elif cohort == "s800":
                            items = parse_mt(json.loads(line), split_name, cohort, 'Species')
                            data.extend(items)
                        elif cohort == "ncbi":
                            items = parse_mt(json.loads(line), split_name, cohort, 'Disease')
                            data.extend(items)
        else:
            continue
        ret[split_name] = Dataset.from_list(data)
        # print(cnt)
        print(len(data))

    for split_name in ['test']:
        cnt = 0
        data = []
        if kg_type == 'mt':
            for cohort in ["ncbi"]:

                with open(f'./data/vaner_datacohort/{cohort}/{split_name}_df_mix2.jsonl', 'r') as reader:
                    for line in reader:
                        if cohort == "BC2GM":
                            items = parse_mt(json.loads(line), split_name, cohort, 'Gene')
                            data.extend(items)
                        elif cohort == "BC4CHEMD":
                            items = parse_mt(json.loads(line), split_name)
                            data.extend(items)
                        elif cohort == "BC5CDR-chem":
                            items = parse_mt(json.loads(line), split_name)
                            data.extend(items)
                        elif cohort == "BC5CDR-disease":
                            items = parse_mt(json.loads(line), split_name)
                            data.extend(items)
                        elif cohort == "JNLPBA":
                            items = parse_mt(json.loads(line), split_name)
                            data.extend(items)
                        elif cohort == "linnaeus":
                            items = parse_mt(json.loads(line), split_name)
                            data.extend(items)
                        elif cohort == "s800":
                            items = parse_mt(json.loads(line), split_name)
                            data.extend(items)
                        elif cohort == "ncbi":
                            items = parse_mt(json.loads(line), split_name, cohort, 'Disease')
                            data.extend(items)
        else:
            continue

        ret[split_name] = Dataset.from_list(data)
        # print(cnt)
        print(len(data))
    # ipdb.set_trace()

    return DatasetDict(ret)





def load_BC2GM(kg_type):
    ret = {}
    for split_name in ['train', 'test']:
        cnt = 0
        data = []
        if kg_type == 'mt':
            with open(f'./data/vaner_datacohort/BC2GM/{split_name}_df_mix2.jsonl', 'r') as reader:
                for line in reader:
                    # items = parse_mt_species(json.loads(line), split_name)
                    items = parse_mt(json.loads(line), split_name, 'BC2GM', 'Gene')
                    data.extend(items)
        else:
            with open(f'./data/BC2GM/{split_name}.jsonl', 'r') as reader:
                for line in reader:
                    data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
        # print(cnt)
        print(len(data))
    return DatasetDict(ret)


def load_BC4CHEMD(kg_type):
    ret = {}
    for split_name in ['train', 'test']:
        cnt = 0
        data = []
        if kg_type == 'mt':
            with open(f'./data/vaner_datacohort/BC4CHEMD/{split_name}_df_mix2.jsonl', 'r') as reader:
                for line in reader:
                    # items = parse_mt_species(json.loads(line), split_name)
                    items = parse_mt(json.loads(line), split_name, 'BC4CHEMD', 'Chemical')
                    data.extend(items)
        else:
            with open(f'./data/BC4CHEMD/{split_name}.jsonl', 'r') as reader:
                for line in reader:
                    data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
        # print(cnt)
        print(len(data))
    return DatasetDict(ret)



def load_BC5CDR_chem(kg_type):
    ret = {}
    for split_name in ['train', 'test']:
        cnt = 0
        data = []
        if kg_type == 'mt':
            with open(f'./data/vaner_datacohort/BC5CDR-chem/{split_name}_df_mix2.jsonl', 'r') as reader:
                for line in reader:
                    # items = parse_mt_species(json.loads(line), split_name)
                    items = parse_mt(json.loads(line), split_name, 'BC5CDR-chem', 'Chemical')
                    data.extend(items)
        else:
            with open(f'./data/BC5CDR-chem/{split_name}.jsonl', 'r') as reader:
                for line in reader:
                    data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
        # print(cnt)
        print(len(data))
    return DatasetDict(ret)


def load_BC5CDR_disease(kg_type):
    ret = {}
    for split_name in ['train', 'test']:
        cnt = 0
        data = []
        if kg_type == 'mt':
            with open(f'./data/vaner_datacohort/BC5CDR-disease/{split_name}_df_mix2.jsonl', 'r') as reader:
                for line in reader:
                    # items = parse_mt_species(json.loads(line), split_name)
                    items = parse_mt(json.loads(line), split_name, 'BC5CDR-disease', 'Disease')
                    data.extend(items)
        else:
            with open(f'./data/BC5CDR-disease/{split_name}.jsonl', 'r') as reader:
                for line in reader:
                    data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
        # print(cnt)
        print(len(data))
    return DatasetDict(ret)




def load_JNLPBA(kg_type):
    ret = {}
    for split_name in ['train', 'test']:
        cnt = 0
        data = []
        if kg_type == 'mt':
            with open(f'./data/vaner_datacohort/JNLPBA/{split_name}_df_mix2.jsonl', 'r') as reader:
                for line in reader:
                    # items = parse_mt_species(json.loads(line), split_name)
                    items = parse_mt(json.loads(line), split_name, 'JNLPBA', 'Gene')
                    data.extend(items)
        else:
            with open(f'./data/JNLPBA/{split_name}.jsonl', 'r') as reader:
                for line in reader:
                    data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
        # print(cnt)
        print(len(data))
    return DatasetDict(ret)


def load_linnaeus(kg_type):
    ret = {}
    for split_name in ['train', 'test']:
        cnt = 0
        data = []
        if kg_type == 'mt':
            with open(f'./data/vaner_datacohort/linnaeus/{split_name}_df_mix2.jsonl', 'r') as reader:
                for line in reader:
                    # items = parse_mt_species(json.loads(line), split_name)
                    items = parse_mt(json.loads(line), split_name, 'linnaeus', 'Species')
                    data.extend(items)
        else:
            with open(f'./data/linnaeus/{split_name}.jsonl', 'r') as reader:
                for line in reader:
                    data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
        # print(cnt)
        print(len(data))
    return DatasetDict(ret)




def load_s800(kg_type):
    ret = {}
    for split_name in ['train', 'test']:
        cnt = 0
        data = []
        if kg_type == 'mt':
            with open(f'./data/vaner_datacohort/s800/{split_name}_df_mix2.jsonl', 'r') as reader:
                for line in reader:
                    # items = parse_mt_species(json.loads(line), split_name)
                    items = parse_mt(json.loads(line), split_name, 's800', 'Species')
                    data.extend(items)
        else:
            with open(f'./data/s800/{split_name}.jsonl', 'r') as reader:
                for line in reader:
                    data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
        # print(cnt)
        print(len(data))
    return DatasetDict(ret)




def load_ncbi(kg_type):
    ret = {}
    for split_name in ['train', 'test']:
        cnt = 0
        data = []
        if kg_type == 'mt':
            with open(f'./data/vaner_datacohort/ncbi/{split_name}_df_mix2.jsonl', 'r') as reader:
                for line in reader:
                    # items = parse_mt_species(json.loads(line), split_name)
                    items = parse_mt(json.loads(line), split_name, 'ncbi', 'Disease')
                    data.extend(items)
        else:
            with open(f'./data/ncbi/{split_name}.jsonl', 'r') as reader:
                for line in reader:
                    data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
        # print(cnt)
        print(len(data))
    return DatasetDict(ret)

# def load_test():
#     ret = {}
#     for split_name in ['train', 'dev', 'test']:
#         data = []
#         with open(f'./data/test/test.jsonl', 'r') as reader:
#             for line in reader:
#                 data.append(json.loads(line))
#         ret[split_name] = Dataset.from_list(data)
#     return DatasetDict(ret)



task, max_length, kgtype, align_mode = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
print(f'handling task {task}')

epochs = 20
batch_size = 4
learning_rate = 1e-4
lora_r = 12
model_id = './Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_id)

if task == 'ncbi':
    ds = load_ncbi(kgtype)
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}
elif task == 'BC2GM':
    ds = load_BC2GM(kgtype)
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}
elif task == 'BC4CHEMD':
    ds = load_BC4CHEMD(kgtype)
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}
elif task == 'BC5CDR-chem':
    ds = load_BC5CDR_chem(kgtype)
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}
elif task == 'BC5CDR-disease':
    ds = load_BC5CDR_disease(kgtype)
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}
elif task == 'JNLPBA':
    ds = load_JNLPBA(kgtype)
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}
elif task == 'linnaeus':
    ds = load_linnaeus(kgtype)
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}
elif task == 's800':
    ds = load_s800(kgtype)
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}
elif task == 'unidev_kgmix2':
    ds = load_unidev_kgmix2(kgtype)

    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}
else:
    raise NotImplementedError
id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys()) # ds["train"].features[f"ner_tags"].feature.names
model = UnmaskingLlamaForTokenClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
).bfloat16()
peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=lora_r, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ipdb.set_trace()

def tokenize_and_align_labels_try1(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, padding='longest', max_length=max_length, truncation=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []

        # # print("word_ids",word_ids)
        word_ids_adjust=[]
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                word_ids_adjust.append(None)

            elif word_idx ==0:  # Only label the first token of a given word.
                word_ids_adjust.append(0)
            else:
                word_ids_adjust.append(word_idx-1)

        for word_idx in word_ids_adjust:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def tokenize_and_align_labels_vaner(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, padding='longest', max_length=max_length, truncation=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        prompt_len = 1

        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                if word_idx < prompt_len:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx - prompt_len])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs



if align_mode == 'vaner': tokenized_ds = ds.map(tokenize_and_align_labels_vaner, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    predictions = true_predictions
    references = true_labels
    precision = precision_score(references, predictions)
    recall = recall_score(references, predictions)
    f1 = f1_score(references, predictions)
    accuracy = calculate_accuracy([item for sublist in references for item in sublist],
                              [item for sublist in predictions for item in sublist])
    # ipdb.set_trace()
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy":accuracy,
    }


output_dir = 'vaner_{}'.format(task)
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=20,
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
