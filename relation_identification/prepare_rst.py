#prepare RST feature
import pickle
with open('./pretrained_models/inner_relation_data.pkl', 'rb') as f:
  inner_relation_data = pickle.load(f)
with open('./pretrained_models/inter_relation_data.pkl', 'rb') as f:
  inter_relation_data = pickle.load(f)


import csv, os, re
inner_rst = []
inter_rst = []
for n in range(len(inter_relation_data)):
  with open('./data/rst/inner'+str(n)+'.merge' ,'r') as f:
    rows = csv.reader(f, delimiter='\t')
    du_dict = {i[-1]: i[0] for i in rows if len(i) != 0}
  with open('./data/rst/inner'+str(n)+'.brackets' ,'r') as f:
    rows = f.read()
    rows = re.sub('\(|\)|\,|\'', '', rows).split('\n')
    a = [i.split(' ') for i in rows if len(i) != 0]
    res = {(du_dict[i[0]], du_dict[i[1]]): i[3] for i in a if du_dict[i[0]] != du_dict[i[1]]}
  inter_rst.append(res)

for n in range(len(inner_relation_data)):
  with open('./data/rst/inter'+str(n)+'.merge' ,'r') as f:
    rows = csv.reader(f, delimiter='\t')
    du_dict = {i[-1]: i[0] for i in rows if len(i) != 0}
  with open('./data/rst/inter'+str(n)+'.brackets' ,'r') as f:
    rows = f.read()
    rows = re.sub('\(|\)|\,|\'', '', rows).split('\n')
    a = [i.split(' ') for i in rows if len(i) != 0]
    res = {(du_dict[i[0]], du_dict[i[1]]): i[3] for i in a if du_dict[i[0]] != du_dict[i[1]]}
  inner_rst.append(res)

with open('./pretrained_models/inner_rst.pkl', 'wb') as f:
  pickle.dump(inner_rst, f)
with open('./pretrained_models/inter_rst.pkl', 'wb') as f:
  pickle.dump(inter_rst, f)