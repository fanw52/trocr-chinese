import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from transformers import get_scheduler


class CTCLoss(nn.Module):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        predicts = predicts.permute((1, 0, 2))
        predicts = F.log_softmax(predicts, dim=2)
        N, B, _ = predicts.shape
        preds_lengths = torch.tensor([N] * B,dtype=torch.int64)
        labels = batch["labels"]
        input_lengths = torch.full(size=(B,), fill_value=N, dtype=torch.int64)
        lengths = batch["length"]
        loss = self.loss_func(predicts, labels, input_lengths, lengths)
        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = torch.subtract(torch.Tensor([1.0]), weight)
            weight = torch.square(weight)
            loss = torch.multiply(loss, weight)
        loss = loss.mean()
        return loss

class TrOCRTrainer():
    def __init__(self, args, model, train_dataloader, eval_dataloader,
                 accelerator,
                 metric,
                 logger
                 ):
        self.accelerator = accelerator
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # Prepare everything with our `accelerator`.
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
        self.metric = metric
        self.args = args
        self.logger = logger
        self.loss = CTCLoss()

    def train(self):
        checkpointing_steps = self.args.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)

        total_batch_size = self.args.per_device_train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0
        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint is not None or self.args.resume_from_checkpoint != "":
                self.accelerator.print(f"Resumed from checkpoint: {self.args.resume_from_checkpoint}")
                self.accelerator.load_state(self.args.resume_from_checkpoint)
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                resume_step = int(training_difference.replace("step_", ""))
                starting_epoch = resume_step // len(self.train_dataloader)
                resume_step -= starting_epoch * len(self.train_dataloader)

        for epoch in range(starting_epoch, self.args.num_train_epochs):
            self.model.train()
            if self.args.with_tracking:
                total_loss = 0
            for step, batch in enumerate(self.train_dataloader):
                # We need to skip steps until we reach the resumed step
                if self.args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue

                outputs = self.model(batch["pixel_values"])
                loss = self.loss(outputs,batch)
                loss = loss / self.args.gradient_accumulation_steps
                self.accelerator.backward(loss)
                progress_bar.set_description(f"loss:{loss:.4f}")
                if step % self.args.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1


                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if self.args.output_dir is not None:
                            output_dir = os.path.join(self.args.output_dir, output_dir)
                        self.accelerator.save_state(output_dir)

                if completed_steps >= self.args.max_train_steps:
                    break

            self.model.eval()
            samples_seen = 0
            for step, batch in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                # 假设这里是ids
                labels = batch["labels"]
                self.metric(predictions, labels)
            self.accelerator.print(f"epoch {epoch}:", self.metric.get_metric())
            if self.args.with_tracking:
                self.accelerator.log(
                    {
                        "seqeval": self.metric.get_metric(),
                        "train_loss": total_loss.item() / len(self.train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

            # if self.args.push_to_hub and epoch < self.args.num_train_epochs - 1:
            #     self.accelerator.wait_for_everyone()
            #     unwrapped_model = self.accelerator.unwrap_model(self.trocr)
            #     unwrapped_model.save_pretrained(
            #         self.args.output_dir, is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save
            #     )
            #     if self.accelerator.is_main_process:
            #         tokenizer.save_pretrained(self.args.output_dir)
            #         repo.push_to_hub(
            #             commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
            #         )

            if self.args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if self.args.output_dir is not None:
                    output_dir = os.path.join(self.args.output_dir, output_dir)
                self.accelerator.save_state(output_dir)

        if self.args.with_tracking:
            self.accelerator.end_training()

        if self.args.output_dir is not None:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                self.args.output_dir, is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save
            )
            if self.accelerator.is_main_process:
                # TODO:
                # tokenizer.save_pretrained(self.args.output_dir)
                # if self.args.push_to_hub:
                #     repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

                all_results = {f"eval_{k}": v for k, v in self.metric.get_metric()}
                if self.args.with_tracking:
                    all_results.update({"train_loss": total_loss.item() / len(self.train_dataloader)})
                with open(os.path.join(self.args.output_dir, "all_results.json"), "w") as f:
                    json.dump(all_results, f)
