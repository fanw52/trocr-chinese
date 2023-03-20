from datasets import load_metric
from transformers import TrOCRProcessor

from dataset import decode_text
from rapidfuzz.distance import Levenshtein
import string


class Metric():
    def __init__(self, config):
        self.processor = TrOCRProcessor.from_pretrained(config.cust_data_init_weights_path)
        self.vocab = self.processor.tokenizer.get_vocab()
        self.vocab_inp = {self.vocab[key]: key for key in self.vocab}
        self.cer_metric = load_metric("./cer.py")

    def compute_metrics(self, pred):
        """
        计算cer,acc
        :param pred:
        :return:
        """
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = [decode_text(pred_id, self.vocab, self.vocab_inp) for pred_id in pred_ids]
        labels_ids[labels_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = [decode_text(labels_id, self.vocab, self.vocab_inp) for labels_id in labels_ids]
        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)

        acc = [pred == label for pred, label in zip(pred_str, label_str)]
        print([pred_str[0], label_str[0]])
        acc = sum(acc) / (len(acc) + 0.000001)

        return {"cer": cer, "acc": acc}



class RecMetric(object):
    def __init__(self,
                 processor,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=False,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()
        self.processor = processor
        self.vocab = self.processor.tokenizer.get_vocab()
        self.vocab_inp = {self.vocab[key]: key for key in self.vocab}
        self.cer_metric = load_metric("./cer.py")


    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()

    def __call__(self, pred_label, *args, **kwargs):
        labels_ids = pred_label.label_ids
        pred_ids = pred_label.predictions

        pred_str = [decode_text(pred_id, self.vocab, self.vocab_inp) for pred_id in pred_ids]
        labels_ids[labels_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = [decode_text(labels_id, self.vocab, self.vocab_inp) for labels_id in labels_ids]
        # preds, labels = pred_label

        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        # for (pred, pred_conf), (target, _) in zip(preds, labels):
        for pred,target in zip(pred_str,label_str):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        return {
            'acc': correct_num / (all_num + self.eps),
            'norm_edit_dis': 1 - norm_edit_dis / (all_num + self.eps)
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        self.reset()
        return {'acc': acc, 'norm_edit_dis': norm_edit_dis}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0

