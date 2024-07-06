import json
import pandas as pd
import torch
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score
def load_texts(path):
    l = []
    claims = {}
    evidences = {}
    with open(path,'r') as data:
     lines=data.readlines()
     for line in lines:
        sample=json.loads(line)
        l.append(sample['id'])
        claims[sample['id']]=sample['claim']
        evidences[sample['id']]=sample['evidences']
    return l,claims,evidences

def load_ids(path):
    data=pd.read_csv(path,error_bad_lines=False, sep='\t')
    l=[]
    claims = {}
    labels={}
    for index,item in data.iterrows():
        id = item['id_left']
        if len(l)==0 or id !=l[-1]:
            l.append(id)
        if id not in claims:
            claims[id] = item["claim_text"]
        if id not in labels:
            labels[id] = 1 if item["cred_label"]==True else 0
    return (l,claims,labels)

def load_evidences(path,topk):
    evidences = {}
    with open(path, 'r') as  data:
     lines = data.readlines()
     for line in lines:
        sample = json.loads(line)
        top=sample['evidences'][:topk]
        evidences[sample['id']] = [e[0] for e in top]
    return evidences

def load_claims(path):
    claims={}
    with open(path) as f:
        lines=f.readlines()
    for line in lines:
        e=json.loads(line)
        claims[e['id']]=e
    return claims




def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

# def tok2int_sent(sentence, tokenizer, max_seq_length):
#     """Loads a data file into a list of `InputBatch`s."""
#     sent_a, sent_b = sentence
#     tokens_a = tokenizer.tokenize(sent_a)
#
#     tokens_b = None
#     if sent_b:
#         tokens_b = tokenizer.tokenize(sent_b)
#         _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
#     else:
#         # Account for [CLS] and [SEP] with "- 2"
#         if len(tokens_a) > max_seq_length - 2:
#             tokens_a = tokens_a[:(max_seq_length - 2)]
#
#     tokens =  ["[CLS]"] + tokens_a + ["[SEP]"]
#     segment_ids = [0] * len(tokens)
#     if tokens_b:
#         tokens = tokens + tokens_b + ["[SEP]"]
#         segment_ids += [1] * (len(tokens_b) + 1)
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#     input_mask = [1] * len(input_ids)
#     padding = [0] * (max_seq_length - len(input_ids))
#
#     input_ids += padding
#     input_mask += padding
#     segment_ids += padding
#
#     assert len(input_ids) == max_seq_length
#     assert len(input_mask) == max_seq_length
#     assert len(segment_ids) == max_seq_length
#
#     return input_ids, input_mask, segment_ids
#
# def tok2int_list(src_list, tokenizer, max_seq_length, max_seq_size=-1):
#     inp_padding = list()
#     msk_padding = list()
#     seg_padding = list()
#     for step, sent in enumerate(src_list):
#         input_ids, input_mask, input_seg = tok2int_sent(sent, tokenizer, max_seq_length)
#         inp_padding.append(input_ids)
#         msk_padding.append(input_mask)
#         seg_padding.append(input_seg)
#     return inp_padding, msk_padding, seg_padding

def reverse(labels):
    result=torch.zeros_like(labels)
    for index in range(labels.shape[0]):
        result[index]= 1 if labels[index]==0 else 0
    return result

def _computing_metrics(true_labels , predicted_labels ):
        """
        Computing classifiction metrics for 3 category classification
        Parameters
        ----------
        true_labels: ground truth
        predicted_labels: predicted labels

        Returns
        -------

        """
        assert len(true_labels) == len(predicted_labels)
        results = {}
        f1_macro = f1_score(true_labels, predicted_labels, average='macro')
        f1_micro = f1_score(true_labels, predicted_labels, average='micro')
        f1 = f1_score(true_labels, predicted_labels)
        # this is the normal precision and recall we seen so many times
        precision_true_class = precision_score(true_labels, predicted_labels, labels=[1], average=None)[0]
        recall_true_class = recall_score(true_labels, predicted_labels, labels=[1], average=None)[0]
        f1_true_class = f1_score(true_labels, predicted_labels, labels=[1], average=None)[0]

        precision_false_class = precision_score(true_labels, predicted_labels, labels=[0], average=None)[0]
        recall_false_class = recall_score(true_labels, predicted_labels, labels=[0], average=None)[0]
        f1_false_class = f1_score(true_labels, predicted_labels, labels=[0], average=None)[0]

        results['f1_macro'] = f1_macro
        results['f1_micro'] = f1_micro

        results['PrecisionTrueCls'] = precision_true_class
        results['RecallTrueCls'] = recall_true_class
        results['F1TrueCls'] = f1_true_class  # this must be normal F1

        results['PrecisionFalseCls'] = precision_false_class
        results['RecallFalseCls'] = recall_false_class
        results['F1FalseCls'] = f1_false_class

        return results