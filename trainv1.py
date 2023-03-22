import argparse
import os
import pprint

import yaml
from datasets import load_metric
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import default_data_collator

from dataset import decode_text, trocrDatasetV2

'''
python trainv1.py --config_file ./configs/base.yaml --device 0,1

'''

def compute_metrics(pred):
    """
    计算cer,acc
    :param pred:
    :return:
    """
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = [decode_text(pred_id, vocab, vocab_inp) for pred_id in pred_ids]
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = [decode_text(labels_id, vocab, vocab_inp) for labels_id in labels_ids]
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    acc = [pred == label for pred, label in zip(pred_str, label_str)]
    print([pred_str[0], label_str[0]])
    acc = sum(acc) / (len(acc) + 0.000001)

    return {"cer": cer, "acc": acc}


def load_config(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr fine-tune训练')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help="训练文字字符数")
    parser.add_argument('--config_file', default="./configs/base.yaml", type=str, help="训练数据路径")
    parser.add_argument('--device', default='6', type=str, help="GPU设置")

    args = parser.parse_args()
    pprint.pprint(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["WANDB_DISABLED"] = "false"
    print("loading data .................")
    config = load_config(args.config_file)
    data_dir = config["data_dir"]
    train_path_list = config["train_data"]
    test_path_list = config["test_data"]
    ##图像预处理
    processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)
    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}
    transformer = lambda x: x  ##图像数据增强函数，可自定义

    train_dataset = trocrDatasetV2(data_dir=data_dir, path_list=train_path_list, processor=processor,
                                   max_target_length=config["max_target_length"], transformer=transformer)
    transformer = lambda x: x  ##图像数据增强函数

    eval_dataset = trocrDatasetV2(data_dir=data_dir, path_list=test_path_list, processor=processor,
                                  max_target_length=config["max_target_length"], transformer=transformer)

    model = VisionEncoderDecoderModel.from_pretrained(args.cust_data_init_weights_path)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.config.vocab_size = model.config.decoder.vocab_size

    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 256
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    cer_metric = load_metric("./cer.py")

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=8,
        fp16=True,
        dataloader_num_workers=8,
        output_dir=config["checkpoint_path"],
        logging_steps=10,
        num_train_epochs=config["num_train_epochs"],
        save_steps=config["eval_steps"],
        eval_steps=config["eval_steps"],
        save_total_limit=5
    )

    # seq2seq trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(os.path.join(config["checkpoint_path"], 'last'))
    processor.save_pretrained(os.path.join(config["checkpoint_path"], 'last'))
