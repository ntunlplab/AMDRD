import random
import argparse
from sklearn.model_selection import train_test_split
import pickle
import random
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import *
from tqdm.auto import tqdm, trange
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn as nn
MAX_SEN_LEN = 64
INTRA_WINDOW = 5
INTER_C_WINDOW = 100
INTER_T_WINDOW = 100
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_batch_size = 32
test_batch_size = 128

parser = argparse.ArgumentParser()
parser.add_argument("PRETRAIN_PATH")
parser.add_argument("SAVE_PATH")
args = parser.parse_args()

PRETRAIN_PATH = args.PRETRAIN_PATH
SAVE_PATH = args.SAVE_PATH

params = {'booster': 'gbtree',
            'objective': 'multi:softprob',
            'gamma': 0.1,
            #'lambda': 2,
            'max_depth': 12,
            'scale_pos_weight':1,
            'eta': 0.04,
            'eval_metric': 'mlogloss',
            'num_class': 3
            }
strategy_dict = {
    'Value':0,
    'Fact':1,
    'Policy':2,
    'Testimony':3,
    'Rhetorical_Statement':4,
    'Major_Claim':5
}

relation_dict = {
 'antithesis': 18,
 'attribution': 10,
 'circumstance': 7,
 'comment': 23,
 'comparison': 11,
 'concession': 15,
 'condition': 6,
 'contrast': 17,
 'disjunction': 22,
 'elaboration': 1,
 'example': 21,
 'explanation': 13,
 'list': 2,
 'manner': 20,
 'means': 12,
 'purpose': 9,
 'reason': 19,
 'same_unit': 8,
 'sequence': 4,
 'span': 5,
 'temporal': 16,
 'textualorganization': 3,
 'topic': 14}

def neg_sample(post_num, relation_type, c, t):
  if relation_type == 'inner':
    p = len([x for x in train_inner_relation_data[post_num] if x[0][0] != 'R'])
    x = min(p-1, c+INTRA_WINDOW)
    y = max(t-INTRA_WINDOW, 0)
    r1 = random.randint(0, x)
    r2 = random.randint(y, p-1)
    while (c, r1) in inner_positive_dict:
      r1 = random.randint(0, x)
    while (r2, t) in inner_positive_dict:
      r2 = random.randint(y, p-1)
    return [('no relation', post_num, c, r1, relation_type), ('no relation', post_num, r2, t, relation_type)]
  else:
    p = len([x for x in train_inter_relation_data[post_num] if x[0][0] != 'R'])
    x = min(train_seg[post_num]-1, INTER_T_WINDOW)
    y = min(train_seg[post_num]-1+INTER_C_WINDOW, p-1)
    r1 = random.randint(0, x)
    r2 = random.randint(train_seg[post_num]-1, y)
    while (c, r1) in inter_positive_dict:
      r1 = random.randint(0, x)
    while (r2, t) in inter_positive_dict:
      r2 = random.randint(y, p-1)
    return [('no relation', post_num, c, r1, relation_type), ('no relation', post_num, r2, t, relation_type)]     

def prepare_positive_data(relation_data, relation_type):
  positive_data = []
  wrong_ann = 0
  for n,i in enumerate(relation_data):
    for j in i:
      if j[0][0] == 'T':
        continue
      r = j[1].split(' ')
      relation = r[0]
      try:
        callout_idx = [n for n, t in enumerate(i) if r[1][5:] == t[0]][0]
        target_idx = [n for n, t in enumerate(i) if r[2][5:] == t[0]][0]
        if target_idx - callout_idx < 5:
          positive_data.append((relation, n, callout_idx, target_idx, relation_type))
      except IndexError:
        wrong_ann +=1
        continue 
  positive_dict = {}
  for i in range(len(relation_data)):
    positive_dict[i] = [(x[2],x[3]) for x in positive_data if x[1] == i]  
  return positive_data, positive_dict

def prepare_test_data(test_relation_data, relation_type, test_seg):
  test_data = []
  for n,i in enumerate(test_relation_data):
    if relation_type == 'inner':
      for num in range(len(i)):
        if i[num][0][0] == 'R':
          break
        else:
          for x in range(num):
            test_data.append(('no relation', n, num, x, relation_type))
            if num - x < INTRA_WINDOW:
              test_data.append(('no relation', n, x, num, relation_type))
    else:
      for num in range(test_seg[n]):
        for x in range(test_seg[n], len(i)):
          if i[x][0][0] == 'R':
            break
          else:
            if (num < INTER_T_WINDOW or num > test_seg[n] - INTER_T_WINDOW) and (x-test_seg[n]) < INTER_C_WINDOW or x > len(i) - INTER_C_WINDOW:
              test_data.append(('no relation', n, x, num, relation_type))
    for j in i:
      if j[0][0] == 'T':
        continue
      r = j[1].split(' ')
      relation = r[0]
      try:
        callout_idx = [n for n, t in enumerate(i) if r[1][5:] == t[0]][0]
        target_idx = [n for n, t in enumerate(i) if r[2][5:] == t[0]][0]
        try:
            test_data.remove(('no relation', n, callout_idx, target_idx, relation_type))
        except:
            print((n, callout_idx, target_idx, relation_type))
        if relation == 'Support' or relation == 'Attack':
          test_data.append((relation, n, callout_idx, target_idx, relation_type))
      except IndexError:
        continue
  return test_data

with open(PRETRAIN_PATH+'inner_relation_data.pkl', 'rb') as f:
  inner_relation_data = pickle.load(f)
with open(PRETRAIN_PATH+'inter_relation_data.pkl', 'rb') as f:
  inter_relation_data = pickle.load(f)
#with open(PRETRAIN_PATH+'inner_rst.pkl', 'rb') as f:
  #inner_rst = pickle.load(f)
#with open(PRETRAIN_PATH+'inter_rst.pkl', 'rb') as f:
  #inter_rst = pickle.load(f)

#train_inner_relation_data, test_inner_relation_data, train_inner_rst, test_inner_rst= train_test_split(inner_relation_data, inner_rst, test_size=0.1, random_state=42)
train_inner_relation_data, test_inner_relation_data= train_test_split(inner_relation_data, test_size=0.1, random_state=42)
seg = []
for i in range(0, len(inter_relation_data), 2):
  for n,x,y in zip(range(1000), inter_relation_data[i], inter_relation_data[i+1]):
    if x[1] != y[1]:
      seg.append(n)
      seg.append(n)
      break
#train_inter_relation_data, test_inter_relation_data, train_seg, test_seg, train_inter_rst, test_inter_rst= train_test_split(inter_relation_data, seg, inter_rst, test_size=0.1, random_state=42)
train_inter_relation_data, test_inter_relation_data, train_seg, test_seg= train_test_split(inter_relation_data, seg, test_size=0.1, random_state=42)

inner_positive_data, inner_positive_dict = prepare_positive_data(train_inner_relation_data, 'inner')
inter_positive_data, inter_positive_dict = prepare_positive_data(train_inter_relation_data, 'inter')
#positive_data = inner_positive_data
positive_data = inner_positive_data + inter_positive_data


