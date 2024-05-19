import random
import numpy as np
import ipdb
random.seed(10)

def iobidx(num):
    return [1] + [2]*(num-1)

def type_match(t1, t2):
    if t1.lower() in t2.lower(): return True
    if t2.lower() in t1.lower(): return True
    return False

def parse_mt(item, split = 'test', dname = 'ncbi', dtype = 'Disease'):
    token1 = f'If any {dtype} entity is mentioned in the {dname} input, extract it. ##input: '
    item['tokens'][0] = token1
    items = []
    items.append(item)
    
    if split == 'train':
        token2 = f'If any {dtype} entity is mentioned in the input, extract it. ##input: '
        ents_levs = item['umls_kg']
        ents = []
        from random import sample

        # ents = (sample(ents_levs[0], min(5, len(ents_levs[0])))
        #         + sample(ents_levs[1] + ents_levs[3], min(5, len(ents_levs[1] + ents_levs[3])))
        #         + sample(ents_levs[2], min(5, len(ents_levs[2]))))

        ents = (sample(ents_levs[0], min(10, len(ents_levs[0])))
                + sample(ents_levs[1], min(5, len(ents_levs[1]))))
        random.shuffle(ents)
        
        nitem = {'tokens': [token2], 'ner_tags': []}
        for ent in ents:
            nitem['tokens'].extend(ent[0].split(' '))
            n = len(ent[0].split(' '))
            if type_match(ent[1], dtype): nitem['ner_tags'].extend([1] + [2]*(n-1))
            else: nitem['ner_tags'].extend([0]*n)
            
        items.append(nitem)
    # ipdb.set_trace()
    return items



def parse_mt_disease(item, split = 'test'):
    token1 = 'If any disease entity is mentioned in the input, extract it. ##input: '
    # token1 = 'If any biomedical entity is mentioned in the input, extract it. ##input: '
    # token1 = 'you are required to extract biomedical entities from input sentence: . ##input:'
    if split == 'train':
        ents_levs = item['umls_kg']

        items, ents = [], []
        from random import sample

        # ents = (sample(ents_levs[0], min(5, len(ents_levs[0])))
        #         + sample(ents_levs[1] + ents_levs[3], min(5, len(ents_levs[1] + ents_levs[3])))
        #         + sample(ents_levs[2], min(5, len(ents_levs[2]))))

        ents = (sample(ents_levs[0], min(10, len(ents_levs[0])))
                + sample(ents_levs[1], min(5, len(ents_levs[1]))))
        random.shuffle(ents)

    else:
        ents_levs=[]
        items, ents = [], []
    random.shuffle(ents)

    if split == 'test': ents = []

    nitem = {'tokens': [token1], 'ner_tags': []}
    for ent in ents:

        nitem['tokens'].extend(ent[0].split(' '))
        n = len(ent[0].split(' '))
        if ent[1] == 'Disease': nitem['ner_tags'].extend([1] + [2]*(n-1))
        else: nitem['ner_tags'].extend([0]*n)
    items.append(nitem)

    item['tokens'][0] = token1
    items.append(item)

    # if split == 'train':
    #     print(items)
    # # ipdb.set_trace()
    return items



def parse_mt_gene(item, split = 'test'):
    token1 = 'If any gene entity is mentioned in the input, extract it. ##input: '
    # token1 = 'If any biomedical entity is mentioned in the input, extract it. ##input: '
    # token1 = 'you are required to extract biomedical entities from input sentence: . ##input:'
    if split == 'train':
        ents_levs = item['umls_kg']

        items, ents = [], []
        from random import sample

        # ents = (sample(ents_levs[0], min(5, len(ents_levs[0])))
        #         + sample(ents_levs[1] + ents_levs[3], min(5, len(ents_levs[1] + ents_levs[3])))
        #         + sample(ents_levs[2], min(5, len(ents_levs[2]))))

        ents = (sample(ents_levs[0], min(10, len(ents_levs[0])))
                + sample(ents_levs[1], min(5, len(ents_levs[1]))))
        random.shuffle(ents)

    else:
        ents_levs=[]
        items, ents = [], []
    random.shuffle(ents)

    if split == 'test': ents = []

    nitem = {'tokens': [token1], 'ner_tags': []}
    for ent in ents:

        nitem['tokens'].extend(ent[0].split(' '))
        n = len(ent[0].split(' '))
        if ent[1] == 'Gene': nitem['ner_tags'].extend([1] + [2]*(n-1))
        else: nitem['ner_tags'].extend([0]*n)
    items.append(nitem)

    item['tokens'][0] = token1
    items.append(item)

    # if split == 'train':
    #     print(items)
    # # ipdb.set_trace()
    return items




def parse_mt_species(item, split = 'test'):
    token1 = 'If any species entity is mentioned in the input, extract it. ##input: '
    if split == 'train':
        ents_levs = item['umls_kg']

        items, ents = [], []
        from random import sample

        # ents = (sample(ents_levs[0], min(5, len(ents_levs[0])))
        #         + sample(ents_levs[1] + ents_levs[3], min(5, len(ents_levs[1] + ents_levs[3])))
        #         + sample(ents_levs[2], min(5, len(ents_levs[2]))))

        ents = (sample(ents_levs[0], min(10, len(ents_levs[0])))
                + sample(ents_levs[1], min(5, len(ents_levs[1]))))
        random.shuffle(ents)

    else:
        ents_levs=[]
        items, ents = [], []

    random.shuffle(ents)

    if split == 'test': ents = []

    if len(ents) > 0:
        nitem = {'tokens': [token1], 'ner_tags': []}
        for ent in ents:
            nitem['tokens'].extend(ent[0].split(' '))
            n = len(ent[0].split(' '))
            if ent[1] == 'Species': nitem['ner_tags'].extend([1] + [2]*(n-1))
            else: nitem['ner_tags'].extend([0]*n)
        items.append(nitem)

    item['tokens'][0] = token1
    items.append(item)

    # if split == 'train':
    #     print(items)
    # ipdb.set_trace()
    return items



def parse_mt_chemical(item, split = 'test'):
    token1 = 'If any chemical entity is mentioned in the input, extract it. ##input: '
    # token1 = 'If any biomedical entity is mentioned in the input, extract it. ##input: '
    # token1 = 'you are required to extract biomedical entities from input sentence: . ##input:'

    if split == 'train':
        ents_levs = item['umls_kg']

        items, ents = [], []
        from random import sample

        # ents = ents_levs[0]+ents_levs[1]+ents_levs[2]+ents_levs[3]

        # ents = (sample(ents_levs[0], min(5, len(ents_levs[0])))
        #         + sample(ents_levs[1] + ents_levs[3], min(5, len(ents_levs[1] + ents_levs[3])))
        #         + sample(ents_levs[2], min(5, len(ents_levs[2]))))

        ents = (sample(ents_levs[0], min(10, len(ents_levs[0])))
                + sample(ents_levs[1], min(5, len(ents_levs[1]))))
        random.shuffle(ents)

    else:
        ents_levs=[]
        items, ents = [], []

    random.shuffle(ents)

    if split == 'test': ents = []

    nitem = {'tokens': [token1], 'ner_tags': []}
    for ent in ents:

        nitem['tokens'].extend(ent[0].split(' '))
        n = len(ent[0].split(' '))
        if ent[1] == 'Chemical': nitem['ner_tags'].extend([1] + [2]*(n-1))
        else: nitem['ner_tags'].extend([0]*n)
    items.append(nitem)

    item['tokens'][0] = token1
    items.append(item)

    # if split == 'train':
    #     print(items)
    # # ipdb.set_trace()
    return items


def overlap_ft(ents):
    nents = []
    str_buf = ''
    for ent in ents:
        if ent[0] not in str_buf:
            nents.append(ent)
            str_buf += ent[0]
    return nents

def build_kg_prompt(epos, eneg):
    if len(epos) == 0 and len(eneg) == 0: 
        return 'If any disease entity is mentioned in the sentence, extract it. ##sentence: '
    
    kg_allow = ''
    for idx, ep in enumerate(epos):
        if idx > 0: kg_allow += '; '
        kg_allow = kg_allow + "{}".format(ep[0])

    kg_forbid = ''
    for idx, ep in enumerate(eneg):
        if idx > 0: kg_forbid += ', '
        kg_forbid = kg_forbid + "{} is {}".format(ep[0], ep[1])

    return 'If any disease is mentioned in the sentence, extract it. Note: {} ##sentence: '.format(kg_forbid)


def parse_umls2(item, kg_type, split = 'test'):
    ents_levs = item['umls_kg']
    ents_cnt = sum([len(ele) for ele in ents_levs])
    forbid_ent, allow_ent = [], []
    items = []

    if '01' in kg_type:
        forbid_ent = ents_levs[1][:2]
        forbid_ent = forbid_ent[:2]
    if '10' in kg_type:
        allow_ent = sorted(ents_levs[0], key=lambda x: -len(x[0]))[:5]
        allow_ent = overlap_ft(allow_ent)
        p_sample = 0.9
        if split == 'train':
            allow_ent = [ele for ele in allow_ent if random.random()<p_sample]
        allow_ent = allow_ent[:2]
    if '11' in kg_type:
        allow_ent = sorted(ents_levs[0], key=lambda x: -len(x[0]))[:1]
        allow_ent += ents_levs[2]
        allow_ent = allow_ent[:2]
        forbid_ent = ents_levs[1][:2] + ents_levs[3][:2]
        forbid_ent = forbid_ent[:2]

    new_prompt = build_kg_prompt(allow_ent, forbid_ent)
    item['tokens'][0] = new_prompt.replace("  ", ' ')
    # print(new_prompt)
    # ipdb.set_trace()
    return item, len(allow_ent)


def parse_umls(item, split = 'test'):
    ents_levs = item['umls_kg']
    ents_cnt = sum([len(ele) for ele in ents_levs])
    # new_prompt = "You are required to extract biomedical entities, the relevant entities is "
    if ents_cnt == 0:
        new_prompt = 'Without knowledge, you are required to extract disease entities from input sentence: '
    else:
        # ents = ents_levs[1]
        topk = 3
        ents = sorted(ents_levs[1], key=lambda x: -len(x[0]))
        # ipdb.set_trace()
        if len(ents) > topk: 
            if split == 'train': 
                selected_ent = ents[:topk]
                # selected_ent = random.sample(ents, 3)
            else:
                selected_ent = ents[:topk]
        else: selected_ent = ents

        ent_desc = ''
        for idx, ent in enumerate(selected_ent):
            if idx > 0: ent_desc += ' '
            ent_desc = ent_desc + "<{}> {}".format(ent[1], ent[0])
            
        new_prompt = 'Given knowledge: {}, you are required to extract disease entities from input sentence: '.format(ent_desc)
    
    # new_prompt = "You are required to extract biomedical entities from input. Input is: "
    # new_prompt = 'you are required to extract disease entities from input sentence: '
    # ipdb.set_trace()
    item['tokens'][0] = new_prompt
    return item



def calculate_accuracy(y_true, y_pred):
    total_correct = 0
    total_samples = 0

    for true_sequence, pred_sequence in zip(y_true, y_pred):
        total_samples += len(true_sequence)
        total_correct += sum(1 for true_label, pred_label in zip(true_sequence, pred_sequence) if true_label == pred_label)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return accuracy


def parse_gold(item):
    tags = item['ner_tags']
    tokens = item['tokens']
    # token1 = 'you are required to extract disease entities, similar entity: '
    token_buff, entities = [], []
    tags.append(0)
    for i in range(len(tags)):
        if tags[i] != 0: 
            token_buff.append(tokens[i+1])
        if i == len(tags) - 1 or (i>1 and tags[i] == 0 and tags[i-1] !=0):
            if len(token_buff) > 0:
                entities.append(' '.join(token_buff))
            token_buff = []
    if len(entities) > 0: 
        entities = entities[:2]
        # entities = [entities[0]]
        # entities = random.sample(entities, 2)
    # token1 = 'Given knowledge: {} you are required to extract disease entities from input sentence: '.format(' ; '.join(entities))
    token1 = 'If any disease entity {} is mentioned in the sentence, extract it. ##sentence: '.format(' ; '.join(entities))
    
    # ipdb.set_trace()
    item['tokens'][0] = token1
    return item




def tokenize_and_align_labels_try1(examples, tokenizer, max_length):
    tokens_list = [ele['tokens'] for ele in examples]
    tags_list = [ele['ner_tags'] for ele in examples]
    tokenized_inputs = tokenizer(tokens_list, is_split_into_words=True, padding='longest', max_length=max_length, truncation=True)

    labels = []
    for i, label in enumerate(tags_list):
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
        
        examples[i]['labels'] = label_ids
        examples[i]['attention_mask'] = tokenized_inputs['attention_mask'][i]
        examples[i]['input_ids'] = tokenized_inputs['input_ids'][i]
        # labels.append(label_ids)
    return examples