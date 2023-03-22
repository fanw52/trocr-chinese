from utils import strQ2B
import random

import os
path = "/data/wufan/data/open_dataset_ocr_line/BAIDU/train.list"
save_dir = "/data/wufan/data/open_dataset_ocr_line/BAIDU"
res = []


vocab = set()
with open(path,encoding="utf-8" ) as rfile:
    for line  in rfile.readlines():
        line = strQ2B(line)
        line = line.strip().split("\t")
        w,h,imname,label = line
        res.append(["train_images/"+imname, label])
        vocab = vocab.union(set(list(label)))

random.seed(0)
random.shuffle(res)

with open(os.path.join(save_dir,"train.txt"),'w',encoding='utf-8') as  w:
    for line  in res[:200000]:
        w.writelines("\t".join(line)+"\n")


with open(os.path.join(save_dir,"val.txt"),'w',encoding='utf-8') as  w:
    for line  in res[200000:]:
        w.writelines("\t".join(line)+"\n")


vocab_list = sorted(list(vocab))
with open(os.path.join(save_dir,"vocab.txt"),'w') as w:
    for line  in vocab_list:
        w.writelines(line+"\n")
