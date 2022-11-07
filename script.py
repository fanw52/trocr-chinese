#
# import os
# import cv2
# data_dir = "/Users/admin/Desktop/数据集/文字识别/公司标注/手写日期"
# for filename in os.listdir(data_dir):
#     path = os.path.join(data_dir,filename)
#     image = cv2.imread(path)
#     image


# x = {"filename": "business_license/image_00000188.jpg", "objectList": {"peopleList": [], "wordList": [
#     {"leftTop": "461,1025", "rightBottom": "697,1042", "leftBottom": "461,1042", "imageType": "wordList",
#      "rightTop": "697,1025", "labels": "1"}}}


import json

result = []

import os
data_dir = "/data/sharedVolume/mtdata/annotator-mt/data"
output_dir = "./detection_hwqz"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(data_dir):
    if not filename.endswith("jpg"):
        continue
    result.append({"path":f"文书样例0916/{filename}"})

print(len(result))


N = len(result)
x = N // 7
for i in range(7):
    start = i * x
    end = (i + 1) * x
    if i + 1 == 7:
        end = N
    with open(f"./{output_dir}/20221009_hwqz_{i}.json", 'w', encoding="utf-8") as wfile:
        json.dump(result[start:end], wfile, ensure_ascii=False)
