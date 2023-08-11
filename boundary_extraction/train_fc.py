import csv, os
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
import argparse
MAX_SEN_LEN = 128
parser = argparse.ArgumentParser()
parser.add_argument("LREC_PATH")
parser.add_argument("SAVE_PATH")
args = parser.parse_args()
LREC_PATH = args.LREC_PATH
SAVE_PATH = args.SAVE_PATH

relation_path = LREC_PATH+'brat_format'
boundary_path = LREC_PATH+'conll_format/post_level/'

def span_ex(s, e):
  has_b = False
  span = []
  p = []
  for n in range(len(s)-e):
    if s[n] in bio_b:
      p.append(n)
      has_b = True
    if (s[n+1] not in bio_i or n == len(s)-e-1) and has_b:
      p.append(n)
      has_b = False
  for i in range(0, len(p), 2):
    span.append((s[p[i]], p[i], p[i+1]))
  return span

def f1_score_token():
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
      pre = [torch.argmax(x).item() for x in i]
      #pre = [x for x in i]
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

def f1_score_span():

  tp = 0
  fp = 0
  fn = 0

  for i,j in zip(results, test_feature):
    pre = [torch.argmax(x).item() for x in i]
    #pre = [x.item() for x in i]
    tru = [x for x in j.labels]
    t_span = span_ex(tru, MAX_SEN_LEN-sum(j.mask_ids))
    p_span = span_ex(pre, MAX_SEN_LEN-sum(j.mask_ids))
    print(t_span, p_span)
    for t in t_span:
      if t in p_span:
        bio_res[t[0]][0] += 1
      else:
        bio_res[t[0]][1] += 1
    for p in p_span:
      if p not in t_span:
        bio_res[p[0]][2] += 1

  for k in list(bio_res):
    tp += bio_res[k][0]
    fp += bio_res[k][1]
    fn += bio_res[k][2]
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1 = 2 * precision * recall / (precision + recall)
  print(tp, fp, fn)
  print(k, precision, recall, f1)

boundary_data = []
for fname in os.listdir(boundary_path):
  with open(boundary_path+fname, encoding = 'utf8') as f:
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

bio_balance = {key: 1 for key in bio_dict if key[0]=='B'}
for t in bio_balance:
  bio_balance[t] = int(3100/bio_fre[t])


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_data, test_data= train_test_split(sentence_data, test_size=0.1, random_state=42)
train_batch_size = 32
test_batch_size = 128

train_feature = data_to_bio_bert(train_data, tokenizer, True, bio_dict, bio_balance)
train_dataloader = prepare_dataloader(train_feature, train_batch_size, 'random')
test_feature = data_to_bio_bert(test_data, tokenizer, False, bio_dict, bio_balance)
test_dataloader = prepare_dataloader(test_feature, test_batch_size, 'sequential')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(bio_dict))
model.to(device)
print(device)


num_train_epochs = 5
num_train_optimization_steps = int(len(train_feature) / train_batch_size) * num_train_epochs
learning_rate = 2e-5
warmup_proportion = 0.1
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
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
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3]
         }
        outputs = model(**inputs)
        loss = outputs[0]
        loss.backward()
        tr_loss += loss.item()
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1
        if step % 50 == 1:
            print("train_loss:",tr_loss)
results = []
for batch in tqdm(test_dataloader):
     model.eval()
     batch = tuple(t.to(device) for t in batch)
     with torch.no_grad():
         inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3]
         }
         outputs = model(**inputs)
         for i in outputs[1]:
             results.append(i)
bio_b = [bio_dict[key] for key in bio_dict if key[0]=='B']
bio_i = [bio_dict[key] for key in bio_dict if key[0]=='I' or key[0]=='X']
bio_res = {key: [0,0,0] for key in bio_b} #tp,fp,fn
f1_score_token()
f1_score_span()
