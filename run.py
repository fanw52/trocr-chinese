import logging
import math
import os

import datasets
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
)
from transformers import TrOCRProcessor

from dataset import trocrDatasetV2
from metric import RecMetric
from model import FastTrOCR
from tool.train import TrOCRTrainer
from utils import parse_args

logger = get_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def main():
    args = parse_args()
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

        # download the dataset.
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    data_dir = "/data/wufan/data/OCRData/open_hand_writing/train_data"
    train_path_list = [f"{data_dir}/handwrite/HW_Chinese/train_hw.txt"]
    test_path_list = [f"{data_dir}/handwrite/HW_Chinese/test_hw.txt"]

    transformer = lambda x: x  ##图像数据增强函数，可自定义
    processor = TrOCRProcessor.from_pretrained(args.model_name_or_path)
    metric = RecMetric(processor=processor)

    train_dataset = trocrDatasetV2(data_dir=data_dir, path_list=train_path_list, processor=processor,
                                   max_target_length=args.max_target_length, transformer=transformer)
    transformer = lambda x: x  ##图像数据增强函数
    eval_dataset = trocrDatasetV2(data_dir=data_dir, path_list=test_path_list, processor=processor,
                                  max_target_length=args.max_target_length, transformer=transformer)
    logger.info(f"  Num examples = {len(train_dataset)}")

    config = AutoConfig.from_pretrained(args.model_name_or_path)

    # config.num_hidden_layers = config.encoder.num_hidden_layers
    # config.hidden_size =  config.encoder.hidden_size
    # config.num_attention_heads = config.encoder.num_attention_heads
    # config.qkv_bias = config.encoder.qkv_bias

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size)
    logger.info(config)

    model = FastTrOCR(config)

    model.to(accelerator.device)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    trainer = TrOCRTrainer(args, model, train_dataloader, eval_dataloader, accelerator, metric, logger)
    trainer.train()


if __name__ == '__main__':
    main()
