import matplotlib.pyplot as plt
from PIL import Image


import os
import cv2
import json
data_dir = "/data/wufan/data/OCRData/open_hand_writing"
path_list = [
    "./train_data/chineseocr-data/rec_hand_line_all_label_train.txt",
    "./train_data/handwrite/HWDB2.0Train_label.txt",
    "./train_data/handwrite/HWDB2.1Train_label.txt",
    "./train_data/handwrite/HWDB2.2Train_label.txt",
    "./train_data/handwrite/hwdb_ic13/handwriting_hwdb_train_labels.txt",
    "./train_data/handwrite/HW_Chinese/train_hw.txt"
]

c = 0
from tqdm import tqdm

result = []
for filename in path_list:
    path = os.path.join(data_dir,filename)
    with open(path,encoding="utf-8") as rfile:
        for line  in tqdm(rfile.readlines()):
            try:
                imname, label = line.strip().split("\t")
                impath = os.path.join(data_dir,"train_data",imname)
                image = cv2.imread(impath)
                shape = image.shape
                result.append({"path": impath, "label": label, "shape": [shape[0],shape[1]]})
            except:
                print(line)

with open("output.json",'w',encoding="utf-8") as wfile:
    json.dump(result,wfile,ensure_ascii=False)
