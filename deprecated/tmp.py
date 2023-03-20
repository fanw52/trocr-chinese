# import json
# import os
#
# data_dir = "/Users/admin/Desktop/数据集/文字识别/公司标注/手写日期已标注"
# output_dir = "/Users/admin/Desktop/github/synth-data/datasets/label.json"
#
# count = 0
# result = []
# char_set = set()
# data_set = {}
# for filename in os.listdir(data_dir):
#     if not filename.endswith("json"):
#         continue
#     path = os.path.join(data_dir, filename)
#
#     with open(path, encoding="utf-8") as rfile:
#         data = json.load(rfile)
#         for line in data:
#             file_name = line["filename"]
#             wordList = line["objectList"]["wordList"][0]
#             label = wordList["labels"]
#             for c in label:
#                 char_set.add(c)
#             count += 1
#             result.append({"filename": os.path.basename(file_name), "label": label})
#
#             if label not in data_set:
#                 data_set[label] = 0
#             data_set[label] += 1
#
# print(c)
# print(data_set)
#
# # for c in  data_set:
#
# import  pprint
# pprint.pprint(char_set)
# pprint.pprint(data_set)
#
# with open(output_dir, 'w', encoding="utf-8") as wfile:
#     json.dump(result, wfile, ensure_ascii=False, indent=2)
#
# import time
# # 获取当前时间
# current_time = int(time.time())
# print(current_time) # 1631186249
# # 转换为localtime
# localtime = time.localtime(current_time)
# print(
#     localtime
# )
# from datetime import datetime
# currentDateAndTime = datetime.now()
# print(currentDateAndTime)


path = "/data/wufan/data/OCRData/open_hand_writing/train_data/handwrite/HW_Chinese/train_hw.txt"

maxl = 0
with open(path,encoding="utf-8") as rfile:
    for line  in rfile.readlines():
        imname,label = line.strip().split("\t")
        l = len(label)
        if l> maxl:
            maxl = l
print(maxl)