import math
import os

import cv2
import lmdb
import numpy as np
import six
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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
                    line = line.strip()
                    if len(line)==0:
                        continue
                    imname, label = line.strip().split("\t")
                    if label=="###":
                        continue
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
            encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels), "text": text,
                        "imname": imname}

        else:
            image = cv2.imread(os.path.join(self.data_dir, imname))

            pixel_values = self.resize(image)
            labels = encode_text(text, max_target_length=self.max_target_length, vocab=self.vocab)
            length = min(len(text), self.max_target_length)

            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
            encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels),
                        "length": torch.tensor(length)}

        return encoding


class LMDBDataSet(Dataset):
    def __init__(self, data_dir,processor, max_target_length=128, transformer=lambda x: x,
                 image_shape=(3, 32, 320)):
        super(LMDBDataSet, self).__init__()

        # global_config = config['Global']
        # dataset_config = config[mode]['dataset']
        # loader_config = config[mode]['loader']
        # batch_size = loader_config['batch_size_per_card']
        # data_dir = dataset_config['data_dir']
        # self.do_shuffle = loader_config['shuffle']

        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir)
        # logger.info("Initialize indexs of datasets:%s" % data_dir)
        self.data_idx_order_list = self.dataset_traversal()
        self.transformer = transformer
        self.processor = processor
        self.vocab = processor.tokenizer.get_vocab()
        self.max_target_length = max_target_length
        # if self.do_shuffle:
        #     np.random.shuffle(self.data_idx_order_list)
        # # self.ops = create_operators(dataset_config['transforms'], global_config)
        # self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx",
        #                                                1)
        #
        # ratio_list = dataset_config.get("ratio_list", [1.0])
        # self.need_reset = True in [x < 1 for x in ratio_list]

        # self.transformer = transformer
        self.decode_image =  DecodeImage()

    def load_hierarchical_lmdb_dataset(self, data_dir):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, dirnames, filenames in os.walk(data_dir + '/'):
            if not dirnames:
                env = lmdb.open(dirpath, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
                txn = env.begin(write=False)
                num_samples = int(txn.get('num-samples'.encode()))
                lmdb_sets[dataset_idx] = {"dirpath": dirpath, "env": env, "txn": txn, "num_samples": num_samples}
                dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def get_img_data(self, value):
        """get_img_data"""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype='uint8')
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    #
    # def get_ext_data(self):
    #     ext_data_num = 0
    #     for op in self.ops:
    #         if hasattr(op, 'ext_data_num'):
    #             ext_data_num = getattr(op, 'ext_data_num')
    #             break
    #     load_data_ops = self.ops[:self.ext_op_transform_idx]
    #     ext_data = []
    #
    #     while len(ext_data) < ext_data_num:
    #         lmdb_idx, file_idx = self.data_idx_order_list[np.random.randint(
    #             len(self))]
    #         lmdb_idx = int(lmdb_idx)
    #         file_idx = int(file_idx)
    #         sample_info = self.get_lmdb_sample_info(
    #             self.lmdb_sets[lmdb_idx]['txn'], file_idx)
    #         if sample_info is None:
    #             continue
    #         img, label = sample_info
    #         data = {'image': img, 'label': label}
    #         # data = transform(data, load_data_ops)
    #         if data is None:
    #             continue
    #         ext_data.append(data)
    #     return ext_data

    def get_lmdb_sample_info(self, txn, index):
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'], file_idx)
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        img, label = sample_info
        data = {'image': img, 'label': label}
        data = self.decode_image(data)
        data["src"] = os.path.basename(self.lmdb_sets[lmdb_idx]["dirpath"])

        # data['ext_data'] = self.get_ext_data()
        # outs = self.transform(data, self.ops)
        # if outs is None:
        #     return self.__getitem__(np.random.randint(self.__len__()))
        # return outs
        image = data["image"]
        text = data["label"]
        image = self.transformer(image)  ##图像增强函数
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = encode_text(text, max_target_length=self.max_target_length, vocab=self.vocab)
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels), "text": text,
                    }
        return encoding

    def __len__(self):
        return self.data_idx_order_list.shape[0]


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

def decode_text(tokens, vocab, vocab_inp):
    ##decode trocr
    s_start = vocab.get('<s>')
    s_end = vocab.get('</s>')
    unk = vocab.get('<unk>')
    pad = vocab.get('<pad>')

    decode_strs = ""
    for tk in tokens:
        if tk not in [s_end, s_start, pad, unk]:
            decode_strs += vocab_inp[tk]

    return decode_strs

def decode_textv1(tokens_list, vocab, vocab_inp):
    ##decode trocr
    s_start = vocab.get('<s>')
    s_end = vocab.get('</s>')
    unk = vocab.get('<unk>')
    pad = vocab.get('<pad>')

    decode_strs = []
    for tokens in tokens_list:
        text = ''
        for tk in tokens:
            if tk not in [s_end, s_start, pad, unk]:
                text += vocab_inp[tk]
        decode_strs.append(text)

    return decode_strs


class DecodeImage(object):
    """ decode image """

    def __init__(self,
                 img_mode='RGB',
                 channel_first=False,
                 ignore_orientation=False,
                 **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first
        self.ignore_orientation = ignore_orientation

    def __call__(self, data):
        img = data['image']
        if six.PY2:
            assert type(img) is str and len(img) > 0, "invalid input 'img' in DecodeImage"
        else:
            assert type(img) is bytes and len(img) > 0, "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype='uint8')
        if self.ignore_orientation:
            img = cv2.imdecode(img, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        else:
            img = cv2.imdecode(img, 1)
        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data['image'] = img
        return data


if __name__ == '__main__':
    from transformers import TrOCRProcessor
    from tqdm import tqdm
    data_dir = "/data/wufan/data/lmdb_rec_data/train"
    cust_data_init_weights_path =  "/data1/wufan/model/cust-data"
    processor = TrOCRProcessor.from_pretrained(cust_data_init_weights_path)
    vocab = processor.tokenizer.get_vocab()
    dataset = LMDBDataSet(data_dir,processor)
    per_device_eval_batch_size = 128
    dataloader = DataLoader(dataset, batch_size=per_device_eval_batch_size,num_workers=32)

    charset = set()
    for line in tqdm(dataloader):
        text = line["text"]
        tmp_set = set(list("".join(text)))
        charset.union(tmp_set)
    char_list = sorted(list(charset))
    with open("./mychar.txt",'w',encoding="utf-8") as wfile:
        for line  in char_list:
            wfile.writelines(line+"\n")
