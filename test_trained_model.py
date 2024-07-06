import sys
sys.path.append('../')
sys.path.append('../apex')

import torch
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import argparse

from bert_nli import BertNLIModel
from utils.nli_data_reader import NLIDataReader
from sklearn.metrics import f1_score, precision_score, recall_score
from utils.myutils import *
from utils.input_example import InputExample



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
        
def evaluate(model, test_data, checkpoint, mute=False, test_bs=10):
    model.eval()
    all_ids=[test_data[i].get_guid() for i in range(len(test_data))]
    sent_pairs = [test_data[i].get_texts() for i in range(len(test_data))]
    
    all_labels = [test_data[i].get_label() for i in range(len(test_data))]
    with torch.no_grad():
        _, probs = model(sent_pairs,checkpoint,bs=test_bs)
    all_predict = [np.argmax(pp) for pp in probs]


    assert len(all_predict) == len(all_labels)
    result=_computing_metrics(all_labels,all_predict)

    acc = len([i for i in range(len(all_labels)) if all_predict[i]==all_labels[i]])*1./len(all_labels)
    with open('./res.txt','w' ) as f:
     for i in range(len(all_ids)):
       if int(all_predict[i])==int(all_labels[i]):
         res='T'
       else: res='F'
       f.write('id:%d TorF:%s \n'%(all_ids[i],res))


    return acc,result


def parse_args():
    ap = argparse.ArgumentParser("arguments for bert-nli evaluation")
    ap.add_argument('--filename',type=str)
    ap.add_argument('-b','--batch_size',type=int,default=32,help='batch size')
    ap.add_argument('-g','--gpu',type=int,default=1,help='run the model on gpu (1) or not (0)')
    ap.add_argument('-cp','--checkpoint',type=int,default=0,help='run the model with checkpointing (1) or not (0)')
    ap.add_argument('-tm','--trained_model',type=str,default='default',help='path to the trained model you want to test; if set as "default", it will find in output xx.state_dict, where xx is the bert-type you specified')
    ap.add_argument('-bt','--bert_type',type=str,default='bert-large',help='model you want to test; make sure this is consistent with your trained model')
    ap.add_argument('--hans',type=int,default=0,help='use hans dataset (1) or not (0)')
    ap.add_argument('--dataset', type=str, default='PolitiFact', help='[Snopes, Politifact]')
    ap.add_argument('--DAType', type=str, default='gpt', help='[gpt,MyDA]')
    ap.add_argument('--epochs', default=100, help='Number of epochs to run', type=int)
    ap.add_argument('--batch_size', default=16, help='Batch size', type=int)
    ap.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    ap.add_argument('--seed', default=123456, type=float, help='Learning rate')
    ap.add_argument("--bert_hidden_dim", default=768, type=int)
    ap.add_argument('--topk',default=5, type=int)
    args = ap.parse_args()
    return args.batch_size, args.gpu, args.trained_model, args.checkpoint, args.bert_type,args

if __name__ == '__main__':
    batch_size, gpu, mpath, checkpoint, bert_type ,args= parse_args()

    if mpath == 'default': mpath = 'output/{}.state_dict'.format(bert_type)


    gpu = bool(gpu)

    checkpoint = bool(checkpoint)


    res=[]
    # Read the dataset
    for fold in range(5):
        testpath_ori = 'declare_%s/%s/mapped_data/5fold/test_%d_ori.tsv' % (args.DAType, args.dataset, fold)
        mpath='./logs/%s_%s/%s/Fold_%d' % (args.dataset, args.DAType,args.filename,fold) + '/best.pt'
        testpath_para = 'declare_%s/%s/mapped_data/5fold/paraphrase_test_%d.tsv' % (args.DAType, args.dataset, fold)
        testpath_neg = 'declare_%s/%s/mapped_data/5fold/neg_test_%d.tsv' % (args.DAType, args.dataset, fold)
        test_ori = load_ids(testpath_ori)
        test_para=load_ids(testpath_para)
        test_neg=load_ids(testpath_neg)
        evidences = load_evidences('./reoutput/%s.json' % (args.dataset), args.topk)
        #dev_para = load_ids(devpath_para)
        #dev_neg = load_ids(devpath_neg)

        test_data1=createdata1(test_ori,evidences)
        test_data2 = createdata1(test_para, evidences)
        test_data3 = createdata1(test_neg, evidences)
        test_data=test_data3+test_data2+test_data1
        model = BertNLIModel(model_path=mpath,batch_size=batch_size,bert_type=bert_type)
        print('test data size: {}'.format(len(test_data)))
        _,result=evaluate(model,test_data,checkpoint,test_bs=batch_size)
        print('| Best Test F1_macro = %.5f | Best Test F1_micro = %.5f \n'
                                        '| Best Test Precision_True_class = %.5f | Best Test Recall_True_class = %.5f '
                                        '| Best Test F1_True_class = %.5f \n'
                                        '| Best Test Precision_False_class = %.5f | Best Test_Recall_False class = %.5f '
                                        '| Best Test F1_False_class = %.5f \n'
                                        % (result['f1_macro'], result['f1_micro'],
                                           result['PrecisionTrueCls'],  result['RecallTrueCls'], result['F1TrueCls'],
                                           result['PrecisionFalseCls'],  result['RecallFalseCls'], result['F1FalseCls'],))
        res.append(result)
    avg_results = {}
    for metric in res[0].keys():
        ls = [fold[metric] for fold in res]
        avg, std = np.mean(ls), np.std(ls)
        avg_results[metric] = {"avg": avg, "std": std,
                               "all_5_folds" : " ".join([str(e) for e in ls])}
    print(avg_results)

    


