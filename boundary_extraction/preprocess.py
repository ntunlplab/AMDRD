import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import *
from tqdm.auto import tqdm, trange
from transformers import BertModel, BertForTokenClassification, BertTokenizer
MAX_SEN_LEN = 128

class Bio_Features(object):
  def __init__(self, unique_id, query_ids, mask_ids, segment_ids, labels):
    self.unique_id = unique_id
    self.query_ids = query_ids
    self.mask_ids = mask_ids
    self.segment_ids = segment_ids
    self.labels = labels

def data_to_bio_bert(inputs, tokenizer, weighted, bio_dict, bio_balance):
  outputs = []
  unique_id = 0
  for (i,j) in enumerate(inputs):
    weight = 0
    query_tokens = ['[CLS]']
    labels = [bio_dict['[CLS]']]
    #query_tokens = ['O']
    #labels = [bio_dict['O']]
    mask_ids = [1]
    for word in j:
      if word[4][0] == 'B':
        if weight < bio_balance[word[4]]:
          weight = bio_balance[word[4]]
      t = tokenizer.tokenize(word[1])
      l = [bio_dict[word[4]]] + [bio_dict['X']]*(len(t)-1)
      query_tokens += t
      labels += l
      mask_ids += [1] + [0]*(len(t)-1)
    query_tokens = query_tokens[:MAX_SEN_LEN-1]
    labels = labels[:MAX_SEN_LEN-1]
    query_tokens.append('[SEP]')
    labels.append(bio_dict['[SEP]'])
    #query_tokens.append('O')
    #labels.append(bio_dict['O'])
    mask_ids.append(1)
    mask_ids = mask_ids[:MAX_SEN_LEN]
    mask_ids += (MAX_SEN_LEN - len(mask_ids))*[0]
    query_tokens += (MAX_SEN_LEN - len(query_tokens))*['[PAD]']
    labels += (MAX_SEN_LEN - len(labels))*[bio_dict['X']]
    query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
    segment_ids = MAX_SEN_LEN*[0] 
    feature = Bio_Features(unique_id,query_ids,mask_ids,segment_ids,labels)
    unique_id += 1

    if weighted:
      for i in range(weight): 
        outputs.append(feature)
    else:
      outputs.append(feature)

  return outputs

def prepare_dataloader(features, test_batch_size, sampler):
  test_ids = torch.tensor([f.query_ids for f in features], dtype=torch.long)
  test_mask = torch.tensor([f.mask_ids for f in features], dtype=torch.long)
  test_segment = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
  test_labels = torch.tensor([f.labels for f in features], dtype=torch.long)
  test_data = TensorDataset(test_ids, test_mask, test_segment, test_labels)
  if sampler == 'sequential':
    test_sampler = SequentialSampler(test_data)
  if sampler == 'random':
    test_sampler = RandomSampler(test_data) 
  test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)
  return test_dataloader