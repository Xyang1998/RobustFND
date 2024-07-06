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



class SCL_mlp_simple(torch.nn.Module):
    def __init__(self,temp):
        super(SCL_mlp_simple, self).__init__()
        self.temp=temp

    def forward(self, query_ori_emb, query_para_emb, query_neg_emb, labels,labels_neg):  # (B,H)
        pos = torch.exp(torch.cosine_similarity(query_ori_emb, query_para_emb) / self.temp) #[B]
        neg = torch.exp(torch.cosine_similarity(query_ori_emb, query_neg_emb) / self.temp) #[B]
        loss = -torch.log(pos / (pos + neg))
        # loss=-torch.log(torch.exp(torch.cosine_similarity(query_ori_emb,query_para_emb)/temp)/expsum)
        # print('prelossshape:')
        # print(loss.shape)
        loss = torch.mean(loss)
        # print('lossshape:')
        # print(loss.shape)
        return loss



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
        postdata.append(InputExample(guid=id, texts=[claim, e], label=label))
    return postdata


def train(model, optimizer, scheduler, train_data, dev_data, batch_size, fp16, checkpoint, gpu, max_grad_norm, best_f1ma,fold, train_para,train_neg,epoch):
    loss_fn = nn.CrossEntropyLoss()

    loss_cl=SCL_mlp_simple(args.temp)

    step_cnt = 0
    best_model_weights = None
    loss_ori_=0
    loss_para_, loss_neg_, cl_loss_=0,0,0


    for pointer in tqdm(range(0, len(train_data), batch_size),desc='training'):
        model.train() # model was in eval mode in evaluate(); re-activate the train mode
        #optimizer.zero_grad() # clear gradients first
        torch.cuda.empty_cache() # releases all unoccupied cached memory 

        step_cnt += 1
        sent_pairs = []
        sent_pairs_p = []
        sent_pairs_n = []


        labels = []
        labels_n=[]

        for i in range(pointer, pointer+batch_size):
            if i >= len(train_data): break
            sents = train_data[i].get_texts()
            if len(word_tokenize(' '.join(sents))) > 300: continue
            sent_pairs.append(sents)
            id=train_data[i].get_guid()

            claim_p=train_para[id]
            sents_p = [claim_p, sents[1]]
            sent_pairs_p.append(sents_p)

            claim_n=train_neg[id]
            sents_n = [claim_n, sents[1]]
            sent_pairs_n.append(sents_n)


            labels.append(train_data[i].get_label())
            if train_data[i].get_label()==0:
             label_n=1
            else:label_n=0
            
            labels_n.append(label_n)


        logits_ori, _ ,emb_ori= model.ff(sent_pairs,checkpoint)
        logits_para, _, emb_para = model.ff(sent_pairs_p, checkpoint)
        logits_neg, _, emb_neg = model.ff(sent_pairs_n, checkpoint)
        if logits_ori is None: continue
        true_labels = torch.LongTensor(labels)
        true_labels_n = torch.LongTensor(labels_n)


        if gpu:
            true_labels = true_labels.to('cuda')
            true_labels_n=true_labels_n.to('cuda')
        loss_ori = loss_fn(logits_ori, true_labels)
        loss_para= loss_fn(logits_para, true_labels)
        loss_neg = loss_fn(logits_neg, true_labels_n)
        clloss=loss_cl(emb_ori,emb_para,emb_neg,None,None)
        loss=loss_ori+0.5*clloss+loss_neg+loss_para
        loss_ori_+=loss_ori.item()
        loss_para_+=loss_para.item()
        loss_neg_+=loss_neg.item()
        cl_loss_+=clloss.item()

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


    logging.info('|Epoch %03d |  Train loss: %.3f'
                 '| Para loss = %.3f | Neg loss = %.3f | CL loss = %.3f'
                 % (epoch, loss_ori_, loss_para_, loss_neg_, cl_loss_))
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
    ap.add_argument('--fp16',type=int,default=0,help='use apex mixed precision training (1) or not (0); do not use this together with checkpoint')
    ap.add_argument('--check_point','-cp',type=int,default=0,help='use checkpoint (1) or not (0); this is required for training bert-large or larger models; do not use this together with apex fp16')
    ap.add_argument('--gpu',type=int,default=1,help='use gpu (1) or not (0)')
    ap.add_argument('-ss','--scheduler_setting',type=str,default='WarmupLinear',choices=['WarmupLinear','ConstantLR','WarmupConstant','WarmupCosine','WarmupCosineWithHardRestarts'])
    ap.add_argument('-tm','--trained_model',type=str,default='None',help='path to the trained model; make sure the trained model is consistent with the model you want to train')
    ap.add_argument('-mg','--max_grad_norm',type=float,default=1.,help='maximum gradient norm')
    ap.add_argument('-wp','--warmup_percent',type=float,default=0.2,help='how many percentage of steps are used for warmup')
    ap.add_argument('-bt','--bert_type',type=str,default='bert-large',help='transformer (bert) pre-trained model you want to use')
    ap.add_argument('-rl','--reinit_layers',type=int,default=0,help='reinitialise the last N layers')
    ap.add_argument('-fl','--freeze_layers',type=int,default=0,help='whether to freeze all but the lasat few layers (1) or not (0)')
    ap.add_argument('--path', default='../test_code/ben_data_10_no_body', help='Input data path', type=str)
    ap.add_argument('--dataset', type=str, default='Snopes', help='[Snopes, Politifact]')
    ap.add_argument('--DAType', type=str, default='MyDA', help='[gpt,MyDA]')
    ap.add_argument('--cl', type=bool, default=True, help='Use cl?')
    ap.add_argument('--epochs', default=100, help='Number of epochs to run', type=int)
    ap.add_argument('--batch_size', default=16, help='Batch size', type=int)
    ap.add_argument('--seed', default=123456, type=float, help='Learning rate')
    ap.add_argument("--bert_hidden_dim", default=768, type=int)
    ap.add_argument('--topk',default=5, type=int)
    ap.add_argument('--warmup_percent', type=float, default=0.2,help='how many percentage of steps are used for warmup')
    ap.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    ap.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    ap.add_argument('--temp',
                        type=float,
                        default=0.3,
                        help="Temperature")
    args = ap.parse_args()
    return args.batch_size, args.epoch_num, args.fp16, args.check_point, args.gpu,  args.scheduler_setting, args.max_grad_norm, args.warmup_percent, args.bert_type, args.trained_model, args.hans, args.reinit_layers, args.freeze_layers,args


if __name__ == '__main__':

    batch_size, epoch_num, fp16, checkpoint, gpu, scheduler_setting, max_grad_norm, warmup_percent, bert_type, trained_model, hans, reinit_layers, freeze_layers,args = parse_args()
    fp16 = bool(fp16)
    gpu = bool(gpu)
    hans = bool(hans)
    checkpoint = bool(checkpoint)
    if trained_model=='None': trained_model=None
    devpath_ori = 'declare_%s/%s/mapped_data/dev_ori.tsv' % (args.DAType, args.dataset)
    devpath_para = 'declare_%s/%s/mapped_data/paraphrase_dev.tsv' % (args.DAType, args.dataset)
    devpath_neg = 'declare_%s/%s/mapped_data/neg_dev.tsv' % (args.DAType, args.dataset)
    evidences = load_evidences('./reoutput/%s.json' % (args.dataset), args.topk)
    other='Snopes' if args.dataset=='PolitiFact' else 'PolitiFact'
    evidences_other = load_evidences('./reoutput/%s.json' % (other), args.topk)

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
     print(args.dataset)
     trainpath_ori = 'declare_%s/%s/mapped_data/5fold/train_%d.tsv' % (args.DAType, args.dataset, fold)
     trainpath_para = 'declare_%s/%s/mapped_data/5fold/paraphrase_train_%d.tsv' % (args.DAType, args.dataset, fold)
     trainpath_neg = 'declare_%s/%s/mapped_data/5fold/neg_train_%d.tsv' % (args.DAType, args.dataset, fold)
     train_ori = load_ids(trainpath_ori)
     train_para = load_ids(trainpath_para)
     train_neg = load_ids(trainpath_neg)
     train_data1 = createdata1(train_ori, evidences)
     train_data = train_data1
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
    
            model_dic,best_f1ma = train(model, optimizer, scheduler, train_data, dev_data, batch_size, fp16, checkpoint, gpu, max_grad_norm, best_f1ma,fold,train_para[1],train_neg[1],ep)
            
            
            if model_dic is not None:
                best_model_dic = model_dic
     assert best_model_dic is not None


    #logging.info('\n=====Training finished. Now start test=====')

