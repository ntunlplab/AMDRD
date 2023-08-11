from utils import *

class Relation_Features(object):
  def __init__(self, unique_id, query_ids, mask_ids, segment_ids, labels):
    self.unique_id = unique_id
    self.query_ids = query_ids
    self.mask_ids = mask_ids
    self.segment_ids = segment_ids
    self.labels = labels


def data_to_relation_bert(inputs, tokenizer, mode, weighted):
  outputs = []
  unique_id = 0
  for (i,j) in enumerate(inputs):
    if j[0] == 'Support':
      labels = 1
    elif j[0] == 'Attack':
      labels = 2
    elif j[0] == 'no relation':
      labels = 0
    else:
      continue
    if mode == 'train':
      if j[4] == 'inner':
        q1 = tokenizer.tokenize(train_inner_relation_data[j[1]][j[2]][2])[:MAX_SEN_LEN-2]
        q2 = tokenizer.tokenize(train_inner_relation_data[j[1]][j[3]][2])[:MAX_SEN_LEN-2]
      else:
        q1 = tokenizer.tokenize(train_inter_relation_data[j[1]][j[2]][2])[:MAX_SEN_LEN-2]
        q2 = tokenizer.tokenize(train_inter_relation_data[j[1]][j[3]][2])[:MAX_SEN_LEN-2]        
    else:
      if j[4] == 'inner':
        q1 = tokenizer.tokenize(test_inner_relation_data[j[1]][j[2]][2])[:MAX_SEN_LEN-2]
        q2 = tokenizer.tokenize(test_inner_relation_data[j[1]][j[3]][2])[:MAX_SEN_LEN-2]
      else:
        q1 = tokenizer.tokenize(test_inter_relation_data[j[1]][j[2]][2])[:MAX_SEN_LEN-2]
        q2 = tokenizer.tokenize(test_inter_relation_data[j[1]][j[3]][2])[:MAX_SEN_LEN-2]      
    query_tokens = ['[CLS]'] + q1 + ['[SEP]'] + q2 + ['[SEP]']
    segment_ids = (len(q1)+2)*[0] + (len(q2)+1)*[1] + (2*MAX_SEN_LEN-len(q1)-len(q2)-3)*[0]
    mask_ids = (len(q1)+2)*[1] + (len(q2)+1)*[1] + (2*MAX_SEN_LEN-len(q1)-len(q2)-3)*[0]
    query_tokens += (2*MAX_SEN_LEN - len(query_tokens))*['[PAD]']
    query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
    feature = Relation_Features(unique_id,query_ids,mask_ids,segment_ids,labels)
    unique_id += 1
    outputs.append(feature)            

  return outputs

def prepare_dataloader(features, pos_data, test_batch_size, sampler, add_neg):
  neg_features = []
  if add_neg:
    for _ in range(1):
      neg_data = []
      for i in pos_data:
        neg_data += neg_sample(i[1], i[4], i[2], i[3])
      output = data_to_relation_bert(neg_data, tokenizer, 'train', False)
      neg_features += output
  all_features = features + neg_features
  test_ids = torch.tensor([f.query_ids for f in all_features], dtype=torch.long)
  test_mask = torch.tensor([f.mask_ids for f in all_features], dtype=torch.long)
  test_segment = torch.tensor([f.segment_ids for f in all_features], dtype=torch.long)
  test_labels = torch.tensor([f.labels for f in all_features], dtype=torch.long)
  test_data = TensorDataset(test_ids, test_mask, test_segment, test_labels)
  if sampler == 'sequential':
    test_sampler = SequentialSampler(test_data)
  if sampler == 'random':
    test_sampler = RandomSampler(test_data)
  test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)
  return test_dataloader
