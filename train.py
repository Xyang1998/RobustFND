import sys
sys.path.append('./apex')

from datetime import datetime
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from transformers import *
import math
import argparse
import random
import copy
import os
from nltk.tokenize import word_tokenize


#from news_nli.utils.myutils import NLIDataReader
from utils.logging_handler import LoggingHandler
from utils.input_example import InputExample
from bert_nli import BertNLIModel
from test_trained_model import evaluate
from utils.myutils import *
import logging

np.random.seed(123456)
torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
torch.cuda.manual_seed_all(123456)
    
def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
    """
    Returns the correct learning rate scheduler
    """
    scheduler = scheduler.lower()
    if scheduler=='constantlr':
        return get_constant_schedule(optimizer)
    elif scheduler=='warmupconstant':
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler=='warmuplinear':
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler=='warmupcosine':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler=='warmupcosinewithhardrestarts':
        return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))

def map_label(s:str):
    if s=='SUPPORTS':
        return 1
    else: return 0

def load_fever(path,args):
    l=[]
    with open(path,'r') as f:
        datas=f.readlines()
        for data in datas:
            e=json.loads(data)
            label=map_label(e['label'])
            if label==-1:
                continue
            evidences=e['evidence'][:args.topk]
            evidences=[x[2] for x in evidences]
            evidence=''
            for evi in evidences:
                evidence+=evi
            claim=e['claim']
            l.append(InputExample(guid=e['id'], texts=[claim, evidence], label=label))
    return l



def createdata1(data,evidences):
    postdata = []
    index = 0
    for id in data[0]:
        claim = data[1][id]
        curevidences = evidences[str(id)]
        e = ''
        for s in curevidences:
            e += s + '.'
        label = data[2][id]
        postdata.append(InputExample(guid=index, texts=[claim, e], label=label))
    return postdata


def train(model, optimizer, scheduler, train_data, dev_data, batch_size, fp16, checkpoint, gpu, max_grad_norm, best_f1ma,fold):
    loss_fn = nn.CrossEntropyLoss()

    step_cnt = 0
    best_model_weights = None

    for pointer in tqdm(range(0, len(train_data), batch_size),desc='training'):
        model.train() # model was in eval mode in evaluate(); re-activate the train mode
        #optimizer.zero_grad() # clear gradients first
        torch.cuda.empty_cache() # releases all unoccupied cached memory 

        step_cnt += 1
        sent_pairs = []
        labels = []

        for i in range(pointer, pointer+batch_size):
            if i >= len(train_data): break
            sents = train_data[i].get_texts()
            if len(word_tokenize(' '.join(sents))) > 300: continue
            sent_pairs.append(sents)
            labels.append(train_data[i].get_label())
        logits, _,_ = model.ff(sent_pairs,checkpoint)
        if logits is None: continue
        true_labels = torch.LongTensor(labels)
        if gpu:
            true_labels = true_labels.to('cuda')
        loss = loss_fn(logits, true_labels)

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps


        # back propagate
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if step_cnt % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        # update weights 
        #optimizer.step()

        # update training rate


    acc,result = evaluate(model,dev_data,checkpoint,mute=True)
    logging.info('| Best Test F1_macro = %.5f | Best Test F1_micro = %.5f \n'
                                    '| Best Test Precision_True_class = %.5f | Best Test Recall_True_class = %.5f '
                                    '| Best Test F1_True_class = %.5f \n'
                                    '| Best Test Precision_False_class = %.5f | Best Test_Recall_False class = %.5f '
                                    '| Best Test F1_False_class = %.5f \n'
                                    % (result['f1_macro'], result['f1_micro'],
                                       result['PrecisionTrueCls'],  result['RecallTrueCls'], result['F1TrueCls'],
                                       result['PrecisionFalseCls'],  result['RecallFalseCls'], result['F1FalseCls'],))
    logging.info('==> step {} dev acc: {}'.format(step_cnt,acc))
    f1ma=result['f1_macro']
    if f1ma > best_f1ma:
            best_f1ma = f1ma
            best_model_weights = copy.deepcopy(model.cpu().state_dict())
            with open('./logs/%s_%s/Fold_%d' % (args.dataset, args.DAType, fold) + '/best.pt', "wb") as f:
                torch.save(model.state_dict(), f)
            #model.save(model_save_path,best_model_weights,best_acc)
            model.to('cuda')


    return best_model_weights,best_f1ma


def parse_args():
    ap = argparse.ArgumentParser("arguments for bert-nli training")
    ap.add_argument('--lr',type=float,default=2e-5,help='lr')
    ap.add_argument('-b','--batch_size',type=int,default=32,help='batch size')
    ap.add_argument('-ep','--epoch_num',type=int,default=10,help='epoch num')
    ap.add_argument('--fp16',type=int,default=0,help='use apex mixed precision training (1) or not (0); do not use this together with checkpoint')
    ap.add_argument('--check_point','-cp',type=int,default=0,help='use checkpoint (1) or not (0); this is required for training bert-large or larger models; do not use this together with apex fp16')
    ap.add_argument('--gpu',type=int,default=1,help='use gpu (1) or not (0)')
    ap.add_argument('-ss','--scheduler_setting',type=str,default='WarmupLinear',choices=['WarmupLinear','ConstantLR','WarmupConstant','WarmupCosine','WarmupCosineWithHardRestarts'])
    ap.add_argument('-tm','--trained_model',type=str,default='None',help='path to the trained model; make sure the trained model is consistent with the model you want to train')
    ap.add_argument('-mg','--max_grad_norm',type=float,default=1.,help='maximum gradient norm')
    ap.add_argument('-wp','--warmup_percent',type=float,default=0.2,help='how many percentage of steps are used for warmup')
    ap.add_argument('-bt','--bert_type',type=str,default='bert-large',help='transformer (bert) pre-trained model you want to use')
    ap.add_argument('--hans',type=int,default=0,help='use hans data (1) or not (0)')
    ap.add_argument('-rl','--reinit_layers',type=int,default=0,help='reinitialise the last N layers')
    ap.add_argument('-fl','--freeze_layers',type=int,default=0,help='whether to freeze all but the lasat few layers (1) or not (0)')
    ap.add_argument('--path', default='../test_code/ben_data_10_no_body', help='Input data path', type=str)
    ap.add_argument('--dataset', type=str, default='Snopes', help='[Snopes, Politifact]')
    ap.add_argument('--DAType', type=str, default='MyDA', help='[gpt,MyDA]')
    ap.add_argument('--cl', type=bool, default=True, help='Use cl?')
    ap.add_argument('--epochs', default=100, help='Number of epochs to run', type=int)
    #ap.add_argument('--batch_size', default=16, help='Batch size', type=int)
    ap.add_argument('--seed', default=123456, type=float, help='Learning rate')
    ap.add_argument("--bert_hidden_dim", default=768, type=int)
    ap.add_argument('--topk',default=5, type=int)
    #ap.add_argument('--warmup_percent', type=float, default=0.2,help='how many percentage of steps are used for warmup')
    ap.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    ap.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = ap.parse_args()
    return args.batch_size, args.epoch_num, args.fp16, args.check_point, args.gpu,  args.scheduler_setting, args.max_grad_norm, args.warmup_percent, args.bert_type, args.trained_model, args.hans, args.reinit_layers, args.freeze_layers,args


if __name__ == '__main__':

    batch_size, epoch_num, fp16, checkpoint, gpu, scheduler_setting, max_grad_norm, warmup_percent, bert_type, trained_model, hans, reinit_layers, freeze_layers,args = parse_args()
    fp16 = bool(fp16)
    gpu = bool(gpu)
    hans = bool(hans)
    checkpoint = bool(checkpoint)
    if trained_model=='None': trained_model=None



    label_num = 2

    devpath_ori = 'declare_%s/%s/mapped_data/dev_ori.tsv' % (args.DAType, args.dataset)
    devpath_para = 'declare_%s/%s/mapped_data/paraphrase_dev.tsv' % (args.DAType, args.dataset)
    devpath_neg = 'declare_%s/%s/mapped_data/neg_dev.tsv' % (args.DAType, args.dataset)
    evidences = load_evidences('./reoutput/%s.json' % (args.dataset), args.topk)


    dev_ori = load_ids(devpath_ori)
    dev_para = load_ids(devpath_para)
    dev_neg = load_ids(devpath_neg)
    dev_data1=createdata1(dev_ori,evidences)
    dev_data2=createdata1(dev_para,evidences)
    dev_data3=createdata1(dev_neg,evidences)
    dev_data=dev_data1+dev_data2+dev_data3




    for fold in range(5):
     if not os.path.exists('./logs/%s_%s/Fold_%d' % (args.dataset, args.DAType, fold)):
            os.makedirs('./logs/%s_%s/Fold_%d' % (args.dataset, args.DAType, fold))
     handlers = [logging.FileHandler('./logs/%s_%s/Fold_%d/log.txt' % (args.dataset, args.DAType, fold)),
                    logging.StreamHandler()]
     logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',level=logging.DEBUG, handlers=handlers)
     trainpath_ori = 'declare_%s/%s/mapped_data/5fold/train_%d.tsv' % (args.DAType, args.dataset, fold)
     trainpath_para = 'declare_%s/%s/mapped_data/5fold/paraphrase_train_%d.tsv' % (args.DAType, args.dataset, fold)
     trainpath_neg = 'declare_%s/%s/mapped_data/5fold/neg_train_%d.tsv' % (args.DAType, args.dataset, fold)


     train_ori = load_ids(trainpath_ori)
     train_para = load_ids(trainpath_para)
     train_neg = load_ids(trainpath_neg)

     train_data1 = createdata1(train_ori, evidences)
     train_data2 = createdata1(train_para, evidences)
     train_data3 = createdata1(train_neg, evidences)

     train_data = train_data1 + train_data2+train_data3
     np.random.shuffle(train_data)


     #train_data=load_fever('bert_train.json',args)

     logging.info('----------------FOLD%d-----------------'%fold)
     logging.info('train data size {}'.format(len(train_data)))
     logging.info('dev data size {}'.format(len(dev_data)))
     logging.info(args)

     total_steps = math.ceil(epoch_num*len(train_data)*1./(batch_size*args.gradient_accumulation_steps))
     warmup_steps = int(total_steps*warmup_percent)

     model = BertNLIModel(gpu=gpu,batch_size=batch_size,bert_type=bert_type, reinit_num=reinit_layers, freeze_layers=freeze_layers)
     optimizer = AdamW(model.parameters(),lr=args.lr,eps=1e-6,correct_bias=False)
     scheduler = get_scheduler(optimizer, scheduler_setting, warmup_steps=warmup_steps, t_total=total_steps)
     if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

     best_f1ma = -1.
     best_model_dic = None
     for ep in range(epoch_num):
            logging.info('\n=====epoch {}/{}====='.format(ep,epoch_num))
            model_dic,best_f1ma = train(model, optimizer, scheduler, train_data, dev_data, batch_size, fp16, checkpoint, gpu, max_grad_norm, best_f1ma,fold)
            if model_dic is not None:
                best_model_dic = model_dic
     assert best_model_dic is not None


    logging.info('\n=====Training finished. Now start test=====')
