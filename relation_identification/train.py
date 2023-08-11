from utils import *
from preprocess import *

def f1_score(test_dataloader):
  results = []
  for batch in tqdm(test_dataloader):
      model.eval()
      batch = tuple(t.to(device) for t in batch)
      with torch.no_grad():
        inputs = {
          "input_ids": batch[0],
          "attention_mask": batch[1],
          "token_type_ids": batch[2]
        }
        outputs = model(**inputs)
        for i in outputs[0]:
          results.append(i)
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
      pre = torch.argmax(i).item()
      #pre = i
      tru = j.labels
      #print(pre, i, tru)
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

train_feature = data_to_relation_bert(positive_data, tokenizer, 'train', True)
test_data = prepare_test_data(test_inner_relation_data, 'inner', test_seg)
test_data += prepare_test_data(test_inter_relation_data, 'inter', test_seg)
test_feature = data_to_relation_bert(test_data, tokenizer, 'test', False)
test_dataloader = prepare_dataloader(test_feature, test_data, test_batch_size, 'sequential', False)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)
num_train_epochs = 15
num_train_optimization_steps = int(len(train_feature) / train_batch_size) * num_train_epochs
learning_rate = 2e-5
warmup_proportion = 0.1
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_train_optimization_steps*warmup_proportion, num_training_steps=num_train_optimization_steps
)

#negtive sampling

global_step = 0
nb_tr_steps = 0
tr_loss = 0

model.zero_grad()
train_iterator = trange(0, int(num_train_epochs), desc="Epoch")
for _ in train_iterator:
    tr_loss = 0
    train_dataloader = prepare_dataloader(train_feature, positive_data, train_batch_size, 'random', True)
    #valid_dataloader = prepare_dataloader(valid_feature, valid_data, train_batch_size, 'random', True)   
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
            }
        model.train()
        outputs = model(**inputs)
        loss = outputs[0]
        loss.backward()

        tr_loss += loss.item()
        optimizer.step()
        scheduler.step()  
        model.zero_grad()
        global_step += 1
        if step % 50 == 1:       
          print("train_loss:",tr_loss)
    f1_score(test_dataloader)
    model.save_pretrained(SAVE_PATH + 'bert_all_relation_all')
    tokenizer.save_pretrained(SAVE_PATH + 'bert_all_relation_all')






