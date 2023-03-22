import argparse
import logging
import os
from datetime import datetime
from glob import glob

import torch
from rapidfuzz.distance import Levenshtein
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from dataset import decode_textv1
from dataset import trocrDatasetV2
from utils import Profile
from utils import load_config

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(name)s:%(asctime)s %(message)s')



'''
python eval.py --cust_data_init_weights_path /data1/wufan/model/my-trocr-chinese --configs_path ./configs/eval.yaml

'''


def get_metirc(correct_num, all_num, norm_edit_dis, eps):
    acc = correct_num / (all_num + eps),
    norm_edit_dis = 1 - norm_edit_dis / (all_num + eps)
    return {"acc": acc, "norm_edit_dis": norm_edit_dis}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr 模型评估')
    parser.add_argument('--cust_data_init_weights_path', default='./cust-data/weights', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--device', default='0', type=str, help="GPU设置")
    parser.add_argument('--dataset_path', default='dataset/HW-hand-write/HW_Chinese/*/*.[j|p]*', type=str,
                        help="img path")
    parser.add_argument('--random_state', default=None, type=int, help="用于训练集划分的随机数")
    parser.add_argument('--max_target_length', default=256, type=int, help="用于训练集划分的随机数")
    parser.add_argument('--per_device_eval_batch_size', default=16, type=int, help="用于训练集划分的随机数")
    parser.add_argument('--output', default="./output", type=str, help="用于训练集划分的随机数")
    parser.add_argument('--half', default=False, type=bool, help="是否使用半精度评估")
    parser.add_argument('--config_path', default="./configs/eval.yaml", type=str, help="评估配置")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    paths = glob(args.dataset_path)

    currentDateAndTime = datetime.now()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)
    vocab = processor.tokenizer.get_vocab()

    vocab_inp = {vocab[key]: key for key in vocab}

    half = args.half
    model = VisionEncoderDecoderModel.from_pretrained(args.cust_data_init_weights_path)
    model.eval()
    model.cuda()
    if half:
        model = model.half()
    pred_str, label_str = [], []
    config = load_config(args.config_path)
    logging.info(config)
    data_dir_list = config["data_dir"]
    label_path_list = config["label_path"]
    for data_dir, label_path in zip(data_dir_list, label_path_list):
        transformer = lambda x: x
        eval_dataset = trocrDatasetV2(data_dir=data_dir, path_list=[label_path], processor=processor,
                                      max_target_length=args.max_target_length, transformer=transformer)

        eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size)
        norm_edit_dis = 0.0
        correct_num = 0
        all_num = 0
        eps = 1e-5
        count = 0

        process_bar = tqdm(eval_dataloader)
        dt = (Profile(), Profile(), Profile())

        with dt[0]:
            for batch in process_bar:
                with torch.no_grad():
                    with dt[1]:
                        pixel_values = batch["pixel_values"]
                        label_list = batch["text"]
                        imname_list = batch["imname"]
                        data = pixel_values.cuda()
                        if half:
                            data = data.half()
                        generated_ids = model.generate(data)
                    with dt[2]:
                        generated_text = decode_textv1(generated_ids.cpu().numpy(), vocab, vocab_inp)
                        for pred, label, imname in zip(generated_text, label_list, imname_list):
                            count += len(label)
                            norm_edit_dis += Levenshtein.normalized_distance(pred, label)
                            if pred == label:
                                correct_num += 1
                            all_num += 1

                        res = get_metirc(correct_num, all_num, norm_edit_dis, eps)
        res = get_metirc(correct_num, all_num, norm_edit_dis, eps)
        logging.info(f"数据集:{os.path.basename(label_path)}")
        logging.info(f"是否是半精度推理:{args.half}")
        logging.info(f"行准确率:{res['acc'][0]:.4f}")
        logging.info(f"单字准确率:{res['norm_edit_dis']:.4f}")
        logging.info(f"总时间:{dt[0].t}")
        logging.info(f"模型预测时间:{dt[1].t}")
        logging.info(f"后处理时间:{dt[2].t}")
        logging.info(f"总字符数:{count}")
        logging.info(f"平均每秒预测的字符数:{count / dt[0].t}")
