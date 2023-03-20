import os
from PIL import Image
import time
import torch
import argparse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dataset import decode_text
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr fine-tune训练')
    parser.add_argument('--cust_data_init_weights_path', default='/Users/admin/Desktop/模型/cust-data', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0', type=str, help="GPU设置")
    parser.add_argument('--test_img_dir', default='./img', type=str, help="img path")


    args = parser.parse_args()

    processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)
    vocab = processor.tokenizer.get_vocab()

    vocab_inp = {vocab[key]: key for key in vocab}
    model = VisionEncoderDecoderModel.from_pretrained(args.cust_data_init_weights_path)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}

    for filename in  os.listdir(args.test_img_dir):
        t = time.time()
        if not filename.endswith("jpg"):
            continue

        path = os.path.join(args.test_img_dir,filename)

        img = Image.open(path).convert('RGB')
        pixel_values = processor([img], return_tensors="pt").pixel_values
        with torch.no_grad():
            if torch.cuda.is_available():
                data = pixel_values[:, :, :].cuda()
                data = data.repeat(64,1,1,1)
            else:
                data = pixel_values[:, :, :].cpu()

            t2 = time.time()
            generated_ids = model.generate(data)
            t3 = time.time()

        generated_text = decode_text(generated_ids[0].cpu().numpy(), vocab, vocab_inp)
        print('file: ',filename,'time take:', round(time.time() - t, 2), "s ocr:", [generated_text])
        image = cv2.imread(path)
        cv2.imshow("win",image)
        cv2.waitKey()
