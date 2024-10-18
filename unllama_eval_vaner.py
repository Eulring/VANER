# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import ipdb

from modeling_llama import UnmaskingLlamaForTokenClassification

# datacohort="JNLPBA"
from utils_vaner import *
###
from seqeval.metrics import precision_score, recall_score, f1_score
def calculate_accuracy(y_true, y_pred):
    total_correct = 0
    total_samples = 0

    for true_sequence, pred_sequence in zip(y_true, y_pred):
        total_samples += len(true_sequence)
        total_correct += sum(1 for true_label, pred_label in zip(true_sequence, pred_sequence) if true_label == pred_label)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return accuracy




def load_ncbi_test(dname):
    # ret = {}
    data = []
    with open(f'./data/vaner_datacohort/ncbi/test_df_mix2.jsonl', 'r') as reader:
            for line in reader:
                items = parse_mt(json.loads(line), "test", "ncbi", 'Disease')
                data.extend(items)

    return data


def load_ncbi():
    ret = {}

    ret['train'] = Dataset.from_list(load_ncbi_test('test')[0:1])
    ret['test'] = Dataset.from_list(load_ncbi_test('test'))


    return DatasetDict(ret)





def load_BC2GM_test(kg_type):
    # ret = {}
    data = []
    with open(f'./data/vaner_datacohort/BC2GM/test_df_mix2.jsonl', 'r') as reader:
            for line in reader:
                items = parse_mt(json.loads(line), "test", "BC2GM", 'Gene')
                data.extend(items)


    return data


def load_BC2GM():
    ret = {}

    ret['train'] = Dataset.from_list(load_BC2GM_test('test')[0:1])
    ret['test'] = Dataset.from_list(load_BC2GM_test('test'))

    return DatasetDict(ret)







def load_JNLPBA_test(dname):
    # ret = {}
    data = []
    with open(f'./data/vaner_datacohort/JNLPBA/test_df_mix2.jsonl', 'r') as reader:
            for line in reader:
                items = parse_mt(json.loads(line), "test", "JNLPBA", 'Gene')
                data.extend(items)

    return data

def load_JNLPBA():
    ret = {}

    ret['train'] = Dataset.from_list(load_JNLPBA_test('test')[0:1])
    ret['test'] = Dataset.from_list(load_JNLPBA_test('test'))

    return DatasetDict(ret)






def load_linnaeus_test(dname):
    # ret = {}
    data = []
    with open(f'./data/vaner_datacohort/linnaeus/test_df_mix2.jsonl', 'r') as reader:
            for line in reader:
                items = parse_mt(json.loads(line),  "test", "linnaeus", 'Species')
                data.extend(items)

    return data


def load_linnaeus():
    ret = {}

    ret['train'] = Dataset.from_list(load_linnaeus_test('test')[0:1])
    ret['test'] = Dataset.from_list(load_linnaeus_test('test'))

    return DatasetDict(ret)




def load_BC5CDR_chem_test(dname):
    # ret = {}
    data = []
    with open(f'./data/vaner_datacohort/BC5CDR-chem/test_df_mix2.jsonl', 'r') as reader:
            for line in reader:
                items = parse_mt(json.loads(line), "test", "BC5CDR-chem", 'Chemical')
                data.extend(items)

    return data


def load_BC5CDR_chem():
    ret = {}

    ret['train'] = Dataset.from_list(load_BC5CDR_chem_test('test')[0:1])
    ret['test'] = Dataset.from_list(load_BC5CDR_chem_test('test'))

    return DatasetDict(ret)




def load_BC5CDR_disease_test(dname):
    # ret = {}
    data = []
    with open(f'./data/vaner_datacohort/BC5CDR-disease/test_df_mix2.jsonl', 'r') as reader:
            for line in reader:
                items = parse_mt(json.loads(line), "test", "BC5CDR-disease", 'Disease')
                data.extend(items)

    return data



def load_BC5CDR_disease():
    ret = {}

    ret['train'] = Dataset.from_list(load_BC5CDR_disease_test('test')[0:1])
    ret['test'] = Dataset.from_list(load_BC5CDR_disease_test('test'))

    return DatasetDict(ret)





def load_BC4CHEMD_test(dname):
    # ret = {}
    data = []
    with open(f'./data/vaner_datacohort/BC4CHEMD/test_df_mix2.jsonl', 'r') as reader:
            for line in reader:
                items = parse_mt(json.loads(line), "test", "BC4CHEMD", 'Chemical')
                data.extend(items)

    return data



def load_BC4CHEMD():
    ret = {}

    ret['train'] = Dataset.from_list(load_BC4CHEMD_test('test')[0:1])
    ret['test'] = Dataset.from_list(load_BC4CHEMD_test('test'))

    return DatasetDict(ret)





def load_s800_test(dname):
    # ret = {}
    data = []
    with open(f'./data/vaner_datacohort/s800/test_df_mix2.jsonl', 'r') as reader:
            for line in reader:
                items = parse_mt(json.loads(line), "test", "s800", 'Species')
                data.extend(items)

    return data


def load_s800():
    ret = {}

    ret['train'] = Dataset.from_list(load_s800_test('test')[0:1])
    ret['test'] = Dataset.from_list(load_s800_test('test'))



    return DatasetDict(ret)





def load_craft_chemicals_test(dname):
    # ret = {}
    data = []
    with open(f'./data/craft/test_craft_chemicals_prompt.jsonl', 'r') as reader:
            for line in reader:
                items = parse_mt(json.loads(line), "test", "craft_chemicals", 'Chemical')
                data.extend(items)

    return data


def load_craft_chemicals():
    ret = {}

    ret['train'] = Dataset.from_list(load_craft_chemicals_test('test')[0:1])
    ret['test'] = Dataset.from_list(load_craft_chemicals_test('test'))




    return DatasetDict(ret)



def load_craft_genes_test(dname):
    # ret = {}
    data = []
    with open(f'./data/craft/test_craft_genes_prompt.jsonl', 'r') as reader:
            for line in reader:
                items = parse_mt(json.loads(line), "test", "test_craft_genes", 'Gene')
                data.extend(items)

    return data


def load_craft_genes():
    ret = {}

    ret['train'] = Dataset.from_list(load_craft_genes_test('test')[0:1])
    ret['test'] = Dataset.from_list(load_craft_genes_test('test'))




    return DatasetDict(ret)




def load_craft_species_test(dname):
    # ret = {}
    data = []
    with open(f'./data/craft/test_craft_species_prompt.jsonl', 'r') as reader:
            for line in reader:
                items = parse_mt(json.loads(line), "test", "craft_species", 'Species')
                data.extend(items)

    return data


def load_craft_species():
    ret = {}

    ret['train'] = Dataset.from_list(load_craft_species_test('test')[0:1])
    ret['test'] = Dataset.from_list(load_craft_species_test('test'))



    return DatasetDict(ret)






if len(sys.argv) != 3:
    print('usage python %.py task model_size')
    sys.exit()

task, lora_path = sys.argv[1], sys.argv[2].lower()


print(f'handling task {task}')

epochs = 1
batch_size = 4
learning_rate = 1e-4
max_length = 128
lora_r = 12
model_id = './Llama-2-7b-hf'

tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# seqeval = evaluate.load("seqeval")
if task == 'ncbi':
    ds = load_ncbi()
    # ipdb.set_trace()
    label2id = {'O': 0, 'B-disease': 1, 'I-disease': 2}

elif task == 's800':
    ds = load_s800()
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}

elif task == 'linnaeus':
    ds = load_linnaeus()
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}

elif task == 'JNLPBA':
    ds = load_JNLPBA()
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}

elif task == 'BC5CDR-chem':
    ds = load_BC5CDR_chem()
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}

elif task == 'BC5CDR-disease':
    ds = load_BC5CDR_disease()
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}

elif task == 'BC4CHEMD':
    ds = load_BC4CHEMD()
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}

elif task == 'BC2GM':
    ds = load_BC2GM()
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}


elif task == 'craft_chemicals':
    ds = load_craft_chemicals()
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}


elif task == 'craft_genes':
    ds = load_craft_genes()
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}


elif task == 'craft_species':
    ds = load_craft_species()
    label2id = {'O': 0, 'B-biomedical': 1, 'I-biomedical': 2}



id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys()) # ds["train"].features[f"ner_tags"].feature.names
model_base = UnmaskingLlamaForTokenClassification.from_pretrained(
    lora_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
).bfloat16()


model = PeftModel.from_pretrained(model_base, lora_path)
model = model.merge_and_unload()




def tokenize_and_align_labels_promptmodel(examples):
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


def tokenize_and_align_labels_new(examples):
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



tokenized_ds = ds.map(tokenize_and_align_labels_new, batched=True)
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


    # ipdb.set_trace()
    predictions = true_predictions
    references = true_labels
    # results = seqeval.compute(predictions=true_predictions, references=true_labels)
    # precision
    precision = precision_score(references, predictions)
    # recall
    recall = recall_score(references, predictions)
    # F1
    f1 = f1_score(references, predictions)
    accuracy = calculate_accuracy([item for sublist in references for item in sublist],
                              [item for sublist in predictions for item in sublist])
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy":accuracy,
    }



training_args = TrainingArguments(
    output_dir="test_model",
    learning_rate=0,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="no",
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
