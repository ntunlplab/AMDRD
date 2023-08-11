import csv, os
import numpy as np
import xgboost as xgb
from utils import *
from preprocess import *

def prepare_result(all_data, mode):
  all_feature = data_to_relation_bert(all_data, tokenizer, mode, False)
  all_dataloader = prepare_dataloader(all_feature, all_data, test_batch_size, 'sequential', False)
  results = []
  for batch in tqdm(all_dataloader):
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
  return results, all_feature

def get_feature(i, relation_data, seg):
  p = len([x for x in relation_data[i[1]] if x[0][0] != 'R'])
  a1 = strategy_dict[relation_data[i[1]][i[2]][1].split(' ')[0]]
  a2 = strategy_dict[relation_data[i[1]][i[3]][1].split(' ')[0]]
  if i[4] == 'inner':
    s = i[2]
    t = i[3]
  else:
    s = (i[2] - seg[i[1]])
    t = i[3]  
  #r.append(r1)
  return a1, a2, s, t     

def prepare_xgb(all_data, results, mode):
  data_x = []
  data_y = []

  for i,j in zip(all_data, results):
    a = [x.item() for x in j] #bert
    a += [int(i[4] == 'inner')]
    if mode == 'train':
      if i[4] == 'inner': #strategy
        a1, a2, s, t = get_feature(i, train_inner_relation_data, train_seg)
      else:
        a1, a2, s, t = get_feature(i, train_inter_relation_data, train_seg)
    else:
      if i[4] == 'inner':
        a1, a2, s, t = get_feature(i, test_inner_relation_data, test_seg)
      else:
        a1, a2, s, t = get_feature(i, test_inter_relation_data, test_seg)    
    a += [s, t, int(s>t), s-t] #position     
    a += [int(i==a1) for i in range(len(strategy_dict))] + [int(i==a2) for i in range(len(strategy_dict))] #strategy
    #a += [int(i==r1) for i in range(len(relation_dict))] #RST, comment out if do not use RST features
         
    if i[0] == 'no relation':
      data_y.append(0)
    elif i[0] == 'Support':
      data_y.append(1)
    elif i[0] == 'Attack':
      data_y.append(2)
    else:
      continue
    data_x.append(a)
  
  return np.array(data_x), data_y

with open(PRETRAIN_PATH+'inner_rst.pkl', 'rb') as f:
  inner_rst = pickle.load(f)
with open(PRETRAIN_PATH+'inter_rst.pkl', 'rb') as f:
  inter_rst = pickle.load(f)
train_inner_rst, test_inner_rst= train_test_split(inner_rst, test_size=0.1, random_state=42)
train_inter_rst, test_inter_rst= train_test_split(inter_rst, test_size=0.1, random_state=42)


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load(PRETRAIN_PATH+'bert_all_relation_all/pytorch_model.bin'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

neg_data = []
for _ in range(1):  
  for i in positive_data:
    neg_data += neg_sample(i[1], i[4], i[2], i[3])
all_data = positive_data + neg_data
test_data = prepare_test_data(test_inner_relation_data, 'inner', test_seg)
test_data += prepare_test_data(test_inter_relation_data, 'inter', test_seg)

MAX_SEN_LEN = 64
train_results, train_feature = prepare_result(all_data, 'train')
test_results, test_feature = prepare_result(test_data, 'test')
  
train_x, train_y = prepare_xgb(all_data, train_results, 'train')
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
test_x, test_y = prepare_xgb(test_data, test_results, 'test')

d_train = xgb.DMatrix(train_x, label=train_y)
d_valid = xgb.DMatrix(valid_x, label=valid_y)
d_test = xgb.DMatrix(test_x, label=test_y)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
xgb_model = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=300, verbose_eval=10)
xgb_pre = xgb_model.predict(d_test)

results = xgb_pre

t_tp = 0
t_fp = 0
t_fn = 0
t_tn = 0
relation_types = ['Support','Attack']
for n,r in enumerate(relation_types):
  tp = 0.00001
  fp = 0.00001
  fn = 0.00001
  tn = 0.00001
  for i,j in zip(results, test_feature):
    pre = np.argmax(i)
    tru = j.labels
    if tru == pre and tru == n+1:
      tp+=1
    elif tru != pre and pre == n+1:
      fp+=1
    elif tru != pre and tru == n+1:
      fn+=1
    else:
      tn += 1
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1 = 2 * precision * recall / (precision + recall)
  print(tp, fp, fn)
  print(r, precision, recall, f1)
  t_tp += tp
  t_fp += fp
  t_fn += fn
  t_tn += tn
precision = t_tp/(t_tp+t_fp)
recall = t_tp/(t_tp+t_fn)
f1 = 2 * precision * recall / (precision + recall)
acc = (t_tp+t_tn)/(t_tp+t_fp+t_fn+t_tn)

print('Micro:', precision, recall, f1, acc)

#xgb_model.save_model(PRETRAIN_PATH+'xgb_relation.model')
