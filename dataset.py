import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def resize_norm_img(img,
                    image_shape,
                    padding=True,
                    interpolation=cv2.INTER_LINEAR):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    if not padding:
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=interpolation)
        resized_w = imgW
    else:
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


class SVTRRecResizeImg(object):
    def __init__(self, image_shape, padding=True, **kwargs):
        self.image_shape = image_shape
        self.padding = padding

    def __call__(self, img):
        # img = data['image']

        norm_img, valid_ratio = resize_norm_img(img, self.image_shape,
                                                self.padding)
        # data['image'] = norm_img
        # data['valid_ratio'] = valid_ratio
        return norm_img


class trocrDatasetV2(Dataset):
    def __init__(self, data_dir, path_list, processor, max_target_length=128, transformer=lambda x: x,
                 image_shape=(3, 32, 320), process_type="patch"):
        self.processor = processor
        self.transformer = transformer
        self.max_target_length = max_target_length
        self.vocab = processor.tokenizer.get_vocab()
        self.data = []
        self.data_dir = data_dir
        for path in path_list:
            with open(path, encoding="utf-8") as rfile:
                for line in rfile.readlines():
                    imname, label = line.strip().split("\t")
                    self.data.append({'imname': imname, "label": label})

        self.nsamples = len(self.data)
        self.process_type = process_type

        self.resize = SVTRRecResizeImg(image_shape)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):

        line = self.data[idx]
        imname = line["imname"]
        text = line["label"]

        # image = self.transformer(image)  ##图像增强函数
        if self.process_type == "patch":
            image = Image.open(os.path.join(self.data_dir, imname)).convert("RGB")
            image = self.transformer(image)  ##图像增强函数
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            labels = encode_text(text, max_target_length=self.max_target_length, vocab=self.vocab)
            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
            encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels) , "text": text, "imname": imname}

        else:
            image = cv2.imread(os.path.join(self.data_dir, imname))

            pixel_values = self.resize(image)
            labels = encode_text(text, max_target_length=self.max_target_length, vocab=self.vocab)
            length = min(len(text), self.max_target_length)

            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
            encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels),
                        "length": torch.tensor(length)}

        return encoding


def encode_text(text, max_target_length=128, vocab=None):
    """
    ##自持自定义 list: ['<td>',"3","3",'</td>',....]
    {'input_ids': [0, 1092, 2, 1, 1],
    'attention_mask': [1, 1, 1, 0, 0]}
    """
    if type(text) is not list:
        text = list(text)

    text = text[:max_target_length - 2]
    tokens = [vocab.get('<s>')]
    unk = vocab.get('<unk>')
    pad = vocab.get('<pad>')
    mask = []
    for tk in text:
        token = vocab.get(tk, unk)
        tokens.append(token)
        mask.append(1)

    tokens.append(vocab.get('</s>'))
    mask.append(1)

    if len(tokens) < max_target_length:
        for i in range(max_target_length - len(tokens)):
            tokens.append(pad)
            mask.append(0)

    return tokens
    # return {"input_ids": tokens, 'attention_mask': mask}


def decode_text(tokens_list, vocab, vocab_inp):
    ##decode trocr
    s_start = vocab.get('<s>')
    s_end = vocab.get('</s>')
    unk = vocab.get('<unk>')
    pad = vocab.get('<pad>')

    decode_strs =  []
    for tokens in tokens_list:
        text = ''

        for tk in tokens:
            if tk not in [s_end, s_start, pad, unk]:
                text += vocab_inp[tk]
        decode_strs.append(text)

    return decode_strs
