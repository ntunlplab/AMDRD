import csv, os
import argparse
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import *
from tqdm.auto import tqdm, trange
from transformers import BertModel, BertForTokenClassification, BertTokenizer

from model import *
from preprocess import *

parser = argparse.ArgumentParser()
parser.add_argument("LREC_PATH")
parser.add_argument("SAVE_PATH")
args = parser.parse_args()

LREC_PATH = args.LREC_PATH
SAVE_PATH = args.SAVE_PATH
MAX_SEN_LEN = 128

def f1_score():
  results = []
  for batch in tqdm(test_dataloader):
      model.eval()
      batch = tuple(t.to(device) for t in batch)
      with torch.no_grad():
        inputs = {
          "input_ids": batch[0],
          "input_mask": batch[1],
          "segment_ids": batch[2],
        }
        outputs = model(**inputs)
        for i in outputs[1]:
          results.append(i)
  t_tp = 0
  t_fp = 0
  t_fn = 0
  t_tn = 0

  for tag in list(bio_dict): 
    if tag == 'X':
        continue
    tp = 0.00001
    fp = 0.00001
    fn = 0.00001
    tn = 0.00001
    for i,j in zip(results, test_feature):
      #pre = [torch.argmax(x).item() for x in i]
      pre = [x for x in i]
      tru = [x for x in j.labels]
      for t, p in zip(tru, pre):
          if t == bio_dict['X']:
            continue
          if t == p and t == bio_dict[tag]:
            tp+=1
          elif t != p and p == bio_dict[tag]:
            fp+=1
          elif t != p and t == bio_dict[tag]:
            fn+=1
          elif t != p and t != bio_dict[tag]:
            tn+=1
      precision = tp/(tp+fp)
      recall = tp/(tp+fn)
      f1 = 2 * precision * recall / (precision + recall)
    print(tp, fp, fn)
    print(tag, precision, recall, f1)
    t_tp += tp
    t_fp += fp
    t_fn += fn
    t_tn += tn
  precision = t_tp/(t_tp+t_fp)
  recall = t_tp/(t_tp+t_fn)
  f1 = 2 * precision * recall / (precision + recall)
  print('Micro:', precision, recall, f1)
  print('ACC:', (t_tp+t_tn)/(t_tp+t_fp+t_tn+t_fn))

relation_path = LREC_PATH+'brat_format'
boundary_path = LREC_PATH+'conll_format/post_level/'
MAX_SEN_LEN = 128

boundary_data = []
for fname in os.listdir(boundary_path):
  with open(boundary_path+fname) as f:
    res = []
    rows = csv.reader(f, delimiter='\t')
    for row in rows:
      res.append(row)
    boundary_data.append(res)

sentence_data = []
for post in boundary_data:
  sen = []
  for token in post:
    sen.append(token)
    if token[1] == '.':
      sentence_data.append(sen)
      sen = []
print(len(sentence_data))
bio_dict = {'X':0, '[CLS]':1, '[SEP]':2}
for i in sentence_data:
  for w in i:
    if w[4] not in bio_dict:
      bio_dict[w[4]] = len(bio_dict)
bio_fre = {key: 0 for key in bio_dict}
for i in sentence_data:
  for s in i:
    bio_fre[s[4]] +=1
print(bio_fre)
bio_balance = {key: 1 for key in bio_dict if key[0]=='B'}
for t in bio_balance:
  bio_balance[t] = int(3100/bio_fre[t])
print(bio_balance)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_data, test_data= train_test_split(sentence_data, test_size=0.1, random_state=42)
#train_data, valid_data= train_test_split(train_data, test_size=0.2, random_state=42)

train_batch_size = 32
test_batch_size = 128
train_feature = data_to_bio_bert(train_data, tokenizer, True, bio_dict, bio_balance)
train_dataloader = prepare_dataloader(train_feature, train_batch_size, 'random')
print(len(train_feature))
#valid_feature = data_to_bio_bert(valid_data, tokenizer, True)
#valid_dataloader = prepare_dataloader(valid_feature, train_batch_size, 'random')
test_feature = data_to_bio_bert(test_data, tokenizer, False, bio_dict, bio_balance)
test_dataloader = prepare_dataloader(test_feature, test_batch_size, 'sequential')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertModel.from_pretrained('bert-base-uncased')
model = BERT_CRF_NER(bert_model, bio_dict['[CLS]'], bio_dict['[SEP]'], len(bio_dict), MAX_SEN_LEN, train_batch_size, device)
model.to(device)
print(device)

num_train_epochs = 15
num_train_optimization_steps = int(len(train_feature) / train_batch_size) * num_train_epochs
learning_rate = 5e-5
warmup_proportion = 0.1


param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
        and not any(nd in n for nd in new_param)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
        and not any(nd in n for nd in new_param)], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n in ('transitions','hidden2label.weight')] \
        , 'lr':2e-4, 'weight_decay': 0.005},
    {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'] \
        , 'lr':2e-4, 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_train_optimization_steps*warmup_proportion, num_training_steps=num_train_optimization_steps
)

global_step = 0
nb_tr_steps = 0
tr_loss = 0

model.zero_grad()
train_iterator = trange(0, int(num_train_epochs), desc="Epoch")
for _ in train_iterator:
    tr_loss = 0
    total_step = len(train_data) // train_batch_size
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        model.train()
        neg_log_likelihood = model.neg_log_likelihood(batch[0], batch[1], batch[2], batch[3])
        neg_log_likelihood.backward()

        tr_loss += neg_log_likelihood.item()
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1
        if step % 50 == 1:
          print("train_loss:",tr_loss)
    f1_score()
    torch.save(model.state_dict(), SAVE_PATH+'bert_bio_crf.pkl.'+str(global_step))





