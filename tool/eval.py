import argparse
import json
import logging
import os
from datetime import datetime
from glob import glob

import torch
from rapidfuzz.distance import Levenshtein
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, TrOCRForCausalLM

from dataset import decode_text,decode_textv1
from dataset import trocrDatasetV2


def get_metirc(correct_num, all_num, norm_edit_dis, eps):
    acc = correct_num / (all_num + eps),
    norm_edit_dis = 1 - norm_edit_dis / (all_num + eps)
    return {"acc": acc, "norm_edit_dis": norm_edit_dis}
'''
python eval.py --cust_data_init_weights_path /data1/wufan/model/my-trocr-chinese

'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr 模型评估')
    parser.add_argument('--cust_data_init_weights_path', default='./cust-data/weights', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0', type=str, help="GPU设置")
    parser.add_argument('--dataset_path', default='dataset/HW-hand-write/HW_Chinese/*/*.[j|p]*', type=str,
                        help="img path")
    parser.add_argument('--random_state', default=None, type=int, help="用于训练集划分的随机数")
    parser.add_argument('--max_target_length', default=256, type=int, help="用于训练集划分的随机数")
    parser.add_argument('--per_device_eval_batch_size', default=16, type=int, help="用于训练集划分的随机数")
    parser.add_argument('--output', default="./output", type=str, help="用于训练集划分的随机数")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    paths = glob(args.dataset_path)

    currentDateAndTime = datetime.now()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)
    vocab = processor.tokenizer.get_vocab()

    vocab_inp = {vocab[key]: key for key in vocab}

    model = VisionEncoderDecoderModel.from_pretrained(args.cust_data_init_weights_path)
    model.eval()
    model.cuda()

    pred_str, label_str = [], []

    data_dir = "/data/wufan/data/OCRData/open_hand_writing/train_data"
    test_path_list = [f"{data_dir}/handwrite/HW_Chinese/test_hw.txt"]
    # test_path_list = [f"{data_dir}/handwriteHWDB2.2Test_label.txt"]
    # test_path_list = [f"{data_dir}/handwrite/hwdb_ic13/handwriting_ic13_test_labels_strQ2B.txt"]

    # data_dir = "/data/wufan/ocr-server-paddle/doc/手写数据line"
    # test_path_list = [f"/data/wufan/ocr-server-paddle/doc/手写数据line(已改).txt"]

    transformer = lambda x: x
    eval_dataset = trocrDatasetV2(data_dir=data_dir, path_list=test_path_list, processor=processor,
                                  max_target_length=args.max_target_length, transformer=transformer)

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size)
    norm_edit_dis = 0.0
    correct_num = 0
    all_num = 0
    eps = 1e-5

    result = []
    process_bar = tqdm(eval_dataloader)
    import time
    t1 = time.time()
    print("start")
    for batch in process_bar:
        with torch.no_grad():
            pixel_values = batch["pixel_values"]
            label_list = batch["text"]
            imname_list = batch["imname"]
            generated_ids = model.generate(pixel_values.cuda())

            generated_text = decode_textv1(generated_ids.cpu().numpy(), vocab, vocab_inp)
            for pred, label, imname in zip(generated_text, label_list, imname_list):
                norm_edit_dis += Levenshtein.normalized_distance(pred, label)
                if pred == label:
                    correct_num += 1
                else:
                    logging.error(f"\n{imname}: pred: {pred} : label: {label}")
                all_num += 1
                result.append({"imname": imname, "pred": pred, "label": label})
            res = get_metirc(correct_num, all_num, norm_edit_dis, eps)
            # process_bar.set_description(f"{res}")

    print(time.time()-t1)
    res = get_metirc(correct_num, all_num, norm_edit_dis, eps)
    print(res)
    # with open(os.path.join(args.output, "output.json"), 'w', encoding="utf-8") as wfile:
    #     json.dump(result, wfile, ensure_ascii=False)
