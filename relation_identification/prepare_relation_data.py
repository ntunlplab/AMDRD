import csv, os
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("LREC_PATH")
parser.add_argument("SAVE_PATH")
args = parser.parse_args()
LREC_PATH = args.LREC_PATH
SAVE_PATH = args.SAVE_PATH
inner_path = LREC_PATH + 'brat_format/inner_post/'
inter_path = LREC_PATH + 'brat_format/inter_post/'
MAX_SEN_LEN = 64

def read_relation_data(dir):
  with open(dir) as f:
    res = []
    rows = csv.reader(f, delimiter='\t')
    for row in rows:
      res.append(row)
  return res   

inner_relation_data = []
for fname in os.listdir(inner_path):
  if fname == '.DS_Store':
    continue
  inner_relation_data.append(read_relation_data(inner_path+fname+'/negative.ann'))
  inner_relation_data.append(read_relation_data(inner_path+fname+'/op.ann'))
  inner_relation_data.append(read_relation_data(inner_path+fname+'/positive.ann'))

inter_relation_data = []
for fname in os.listdir(inter_path):
  if fname == '.DS_Store':
    continue
  inter_relation_data.append(read_relation_data(inter_path+fname+'/negative.ann'))
  inter_relation_data.append(read_relation_data(inter_path+fname+'/positive.ann'))


with open(SAVE_PATH + 'inner_relation_data.pkl', 'wb') as f:
  pickle.dump(inner_relation_data, f)
with open(SAVE_PATH + 'inter_relation_data.pkl', 'wb') as f:
  pickle.dump(inter_relation_data, f)