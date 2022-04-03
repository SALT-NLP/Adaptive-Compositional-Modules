import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
import re
import csv
import json
import uuid
import pickle as pkl
import numpy as np
import random
from copy import deepcopy
import os
from glob import glob
import logging
import pathlib
from collections import OrderedDict
from settings import args, TASK_DICT, SPECIAL_TOKENS, SPECIAL_TOKEN_IDS, FILL_VAL
from settings import TOKENIZER, LEN_FACTOR, DATA_ATTRS, MEMORY_FACTOR, MODEL_CONFIG, MODEL_CLASS
from multiprocessing import Pool
import sys
import time
import quadprog
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
logger = logging.getLogger(__name__)
import pdb


def make_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def get_gen_token(task):
    if args.use_eos_as_sos:
        return '<|endoftext|>'
    elif args.add_task_tokens:
        return '__' + task + '__'
    else:
        return '__gen__'


def get_model_dir(tasks):
    return os.path.join(args.model_dir_root, tasks[0]) if args.seq_train_type not in ["multitask", "multilm"] else args.model_dir_root


def get_losses(parallel_model, cqa, Y, gen_X, gen_Y, loss_fct):
    # logger.info(cqa)
    # logger.info(Y)
    if "lll" in args.seq_train_type or "multilm" in args.seq_train_type:
        qa_logits, _ = parallel_model(cqa)
        lm_logits, _ = parallel_model(gen_X)
        # logger.info(len(qa_logits))
        Y = Y.view(-1)
        qa_logits = qa_logits.view(Y.shape[0], -1)
        gen_Y = gen_Y.view(-1)
        lm_logits = lm_logits.view(gen_Y.shape[0], -1)

        qa_loss = loss_fct(qa_logits, Y)
        lm_loss = loss_fct(lm_logits, gen_Y)
        return torch.mean(qa_loss), args.lm_lambda * torch.mean(lm_loss)
    else:
        qa_logits, _ = parallel_model(cqa)
        Y = Y.view(-1)
        qa_logits = qa_logits.view(Y.shape[0], -1)
        qa_loss = loss_fct(qa_logits, Y)
        return torch.mean(qa_loss), torch.tensor(0.)

def repeat_last_logits(target, repeat_times=0):
    if repeat_times <= 0:
        return target
    else:
        for t in range(len(target)):
            rep = target[t][:, -1].unsqueeze(-1).expand(-1, repeat_times)
            target[t] = torch.cat([target[t], rep], dim=-1)
        return target




def get_distil_losses(teacher_model, parallel_model, cqa, Y, gen_X, gen_Y, is_extra, kldiv_loss_fct, ce_loss_fct, temperature=2.0, pad_idx=-1, weighting=False, clamp=50260):
    '''
    Compute KL-div between teacher and student models.
    loss_fct should be nn.KLDivLoss(reduction="batchmean")
    '''
    if "lll" in args.seq_train_type or "multilm" in args.seq_train_type:
        qa_mask = [(y != pad_idx).unsqueeze(-1) for y in Y]
        lm_mask = [(gen_y != pad_idx).unsqueeze(-1) for gen_y in gen_Y]
        extra_num = sum([l.sum() for l in is_extra])
        total_num = sum([l.size(0) for l in is_extra])
        if clamp>0:
            cqa_th = [(torch.clamp(l[0], 0, clamp),) for l in cqa]
            gen_X_th = [(torch.clamp(l[0], 0, clamp),) for l in gen_X]
        else:
            cqa_th = cqa
            gen_X_th = gen_X
        if extra_num == 0:
            # all example are current example, use distillation for all.
            qa_logits = [torch.nn.functional.log_softmax(torch.masked_select(l, qa_mask[i].expand_as(l)).view(-1, l.size(-1)) / temperature, dim=-1) for i, l in enumerate(parallel_model(cqa))]
            lm_logits = [torch.nn.functional.log_softmax(torch.masked_select(l, lm_mask[i].expand_as(l)).view(-1, l.size(-1)) / temperature, dim=-1) for i, l in enumerate(parallel_model(gen_X))]
            qa_target = [torch.nn.functional.softmax(torch.masked_select(l.detach(), qa_mask[i].expand_as(l)).view(-1, l.size(-1)) / temperature, dim=-1) for i, l in enumerate(teacher_model(cqa_th))]
            lm_target = [torch.nn.functional.softmax(torch.masked_select(l.detach(), lm_mask[i].expand_as(l)).view(-1, l.size(-1)) / temperature, dim=-1) for i, l in enumerate(teacher_model(gen_X_th))]
            size_diff = qa_logits[0].shape[-1] - qa_target[0].shape[-1]
            if size_diff > 0:
                qa_target = repeat_last_logits(qa_target, size_diff)
                lm_target = repeat_last_logits(lm_target, size_diff)

            #qa_target = torch.nn.functional.softmax(teacher_model(cqa).detach()*qa_mask[i] / temperature, dim=-1)
            #lm_target = torch.nn.functional.softmax(teacher_model(gen_X).detach()*lm_mask[i] / temperature, dim=-1)
            qa_loss = kldiv_loss_fct(qa_logits, qa_target) * (temperature) ** 2
            lm_loss = kldiv_loss_fct(lm_logits, lm_target) * (temperature) ** 2
        else:
            qa_curr_mask = [(mask*(1-ext).unsqueeze(-1).unsqueeze(-1).bool()) for mask, ext in zip(qa_mask, is_extra)]
            lm_curr_mask = [(mask*(1-ext).unsqueeze(-1).unsqueeze(-1).bool()) for mask, ext in zip(lm_mask, is_extra)]
            qa_extr_mask = [(mask*ext.unsqueeze(-1).unsqueeze(-1).bool()) for mask, ext in zip(qa_mask, is_extra)]
            lm_extr_mask = [(mask*ext.unsqueeze(-1).unsqueeze(-1).bool()) for mask, ext in zip(lm_mask, is_extra)]
            qa_stud_logits = parallel_model(cqa)
            lm_stud_logits = parallel_model(gen_X)
            # loss for current training example
            qa_curr_logits = [torch.nn.functional.log_softmax(torch.masked_select(l, qa_curr_mask[i].expand_as(l)).view(-1, l.size(-1)) / temperature, dim=-1) for i, l in enumerate(qa_stud_logits)]
            lm_curr_logits = [torch.nn.functional.log_softmax(torch.masked_select(l, lm_curr_mask[i].expand_as(l)).view(-1, l.size(-1)) / temperature, dim=-1) for i, l in enumerate(lm_stud_logits)]
            qa_curr_target = [torch.nn.functional.softmax(torch.masked_select(l.detach(), qa_curr_mask[i].expand_as(l)).view(-1, l.size(-1)) / temperature, dim=-1) for i, l in enumerate(teacher_model(cqa_th))]
            lm_curr_target = [torch.nn.functional.softmax(torch.masked_select(l.detach(), lm_curr_mask[i].expand_as(l)).view(-1, l.size(-1)) / temperature, dim=-1) for i, l in enumerate(teacher_model(gen_X_th))]
            size_diff = qa_curr_logits[0].shape[-1] - qa_curr_target[0].shape[-1]
            if size_diff > 0:
                qa_curr_target = repeat_last_logits(qa_curr_target, size_diff)
                lm_curr_target = repeat_last_logits(lm_curr_target, size_diff)
            qa_curr_loss = kldiv_loss_fct(qa_curr_logits, qa_curr_target) * (temperature) ** 2
            lm_curr_loss = kldiv_loss_fct(lm_curr_logits, lm_curr_target) * (temperature) ** 2
            # loss for extra example (no distillation)
            qa_extr_loss = ce_loss_fct([torch.transpose(l, 1, 2) for l in qa_stud_logits], [((y+1)*ext.unsqueeze(-1)-1) for y, ext in zip(Y, is_extra)])
            lm_extr_loss = ce_loss_fct([torch.transpose(l, 1, 2) for l in lm_stud_logits], [((y+1)*ext.unsqueeze(-1)-1) for y, ext in zip(gen_Y, is_extra)])
            if weighting:
                qa_loss = ((total_num - extra_num) * qa_curr_loss + extra_num * qa_extr_loss) / total_num
                lm_loss = ((total_num - extra_num) * lm_curr_loss + extra_num * lm_extr_loss) / total_num
            else:
                qa_loss = qa_curr_loss + qa_extr_loss
                lm_loss = lm_curr_loss + lm_extr_loss

        return torch.mean(qa_loss), args.lm_lambda * torch.mean(lm_loss)
    else:
        #qa_mask = (Y != pad_idx).unsqueeze(-1)
        qa_mask = [(y != pad_idx).unsqueeze(-1) for y in Y]
        if clamp>0:
            cqa_th = [(torch.clamp(l[0], 0, clamp),) for l in cqa]
        else:
            cqa_th = cqa
        qa_logits = [torch.nn.functional.log_softmax(torch.masked_select(l, qa_mask[i].expand_as(l)).view(-1, l.size(-1)) / temperature, dim=-1) for i, l in enumerate(parallel_model(cqa))]
        qa_target = [torch.nn.functional.softmax(torch.masked_select(l.detach(), qa_mask[i].expand_as(l)).view(-1, l.size(-1)) / temperature, dim=-1) for i, l in enumerate(teacher_model(cqa_th))]
        size_diff = qa_logits[0].shape[-1] - qa_target[0].shape[-1]
        if size_diff > 0:
            qa_target = repeat_last_logits(qa_target, size_diff)
        qa_loss = kldiv_loss_fct(qa_logits, qa_target) * (temperature) ** 2
        return torch.mean(qa_loss), torch.tensor(0.)

def pad_to_max_len(l, pad_len, val):
    return l + [val] * pad_len


def pad_all_to_max_len(ls, val):
    max_len = max(len(l) for l in ls)
    return [pad_to_max_len(l, max_len-len(l), val) for l in ls]


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # if top_p > 0.0:
    #     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    #     # Remove tokens with cumulative probability above the threshold
    #     sorted_indices_to_remove = cumulative_probs > top_p
    #     # Shift the indices to the right to keep also the first token above the threshold
    #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #     sorted_indices_to_remove[..., 0] = 0

    #     indices_to_remove = sorted_indices[sorted_indices_to_remove]
    #     logits[indices_to_remove] = filter_value
    return logits


def varlen_collate_fn(data):
    batch_size = (len(data) + args.n_gpus - 1) // args.n_gpus
    cqs = torch.tensor(pad_all_to_max_len([datum[0] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    len_cqs = torch.tensor([datum[1] for datum in data]).split(batch_size)
    cqas = torch.tensor(pad_all_to_max_len([datum[2] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    len_cqas = torch.tensor([datum[3] for datum in data]).split(batch_size)
    Ys = torch.tensor(pad_all_to_max_len([datum[4] for datum in data], FILL_VAL)).split(batch_size)
    gen_Xs = torch.tensor(pad_all_to_max_len([datum[5] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    gen_Ys = torch.tensor(pad_all_to_max_len([datum[6] for datum in data], FILL_VAL)).split(batch_size)
    is_extra = torch.tensor([datum[-1] for datum in data]).split(batch_size)
    return list(cqs), list(len_cqs), list(cqas), list(len_cqas), list(Ys), list(gen_Xs), list(gen_Ys), list(is_extra)


def dynamic_collate_fn(data, batch_size):

    def local_collate():
        null_counter = 0
        _cqs, _len_cqs, _cqas, _len_cqas, _Ys, _gen_Xs, _gen_Ys, _is_extra, _idx = [], [], [], [], [], [], [], [], []
        Y_max_len = max(len(data[j][4]) for j in range(st, ed))
        cq_max_len = max(len(data[j][0]) for j in range(st, ed))
        cqa_max_len = max(len(data[j][2]) for j in range(st, ed))
        for j in range(st, ed):
            if None in data[j] or [] in data[j]:
                null_counter += 1
                logger.warning('null example in collate_fn, count: {}'.format(null_counter))
                continue

            pad_len = cqa_max_len - len(data[j][2])

            _cqs.append(pad_to_max_len(data[j][0], cq_max_len-len(data[j][0]), SPECIAL_TOKEN_IDS["pad_token"]))
            _len_cqs.append(data[j][1])
            _cqas.append(pad_to_max_len(data[j][2], pad_len, SPECIAL_TOKEN_IDS["pad_token"]))
            _len_cqas.append(data[j][3])
            _Ys.append(pad_to_max_len(data[j][4], Y_max_len - len(data[j][4]), FILL_VAL))
            _gen_Xs.append(pad_to_max_len(data[j][5], pad_len, SPECIAL_TOKEN_IDS["pad_token"]))
            _gen_Ys.append(pad_to_max_len(data[j][6], pad_len, FILL_VAL))
            _is_extra.append(data[j][-2])
            _idx.append(data[j][-1])

        cqs.append(torch.tensor(_cqs))
        len_cqs.append(torch.tensor(_len_cqs))
        cqas.append(torch.tensor(_cqas))
        len_cqas.append(torch.tensor(_len_cqas))
        Ys.append(torch.tensor(_Ys))
        gen_Xs.append(torch.tensor(_gen_Xs))
        gen_Ys.append(torch.tensor(_gen_Ys))
        is_extra.append(torch.tensor(_is_extra))
        idx.append(torch.tensor(_idx))

    cqs, len_cqs, cqas, len_cqas, Ys, gen_Xs, gen_Ys, is_extra, idx = [], [], [], [], [], [], [], [], []
    st, ed = 0, len(data)
    local_collate()

    return cqs, len_cqs, cqas, len_cqas, Ys, gen_Xs, gen_Ys, is_extra, idx


class QADataset(Dataset):
    def __init__(self, data_paths, data_type, gen_token, extra_data=[]):
        self.data_type = data_type
        self.gen_token = gen_token
        if args.use_sep:
            self.sep_token = SPECIAL_TOKEN_IDS["sep_token"]
        self.ans_token = SPECIAL_TOKEN_IDS["ans_token"]
        self.eos_token = SPECIAL_TOKEN_IDS["eos_token"]
        self.pad_token = SPECIAL_TOKEN_IDS["pad_token"]

        if not isinstance(data_paths, list):
            data_paths = [data_paths]

        data = []

        if args.upsample_data is not None:
            upsample_rate = [int(x) for x in args.upsample_data.split('_')]
            assert len(upsample_rate) == len(data_paths)
        else:
            upsample_rate = None
        if args.round_robin:
            temp_data = []
            total_len = 0
            for i in range(len(data_paths)):
                temp_data.append([])

        self.multitask_specific = []
        for i, data_path in enumerate(data_paths):
            if not data_path:
                continue
            with open(data_path, "r") as f:
                raw_ds = json.load(f)
                if not args.test_training_set:
                    raw_ds = map(lambda x: x["paragraphs"], raw_ds["data"])
                else:
                    new_raw_ds = []
                    for i1 in range(len(raw_ds["data"])):
                        for i2 in range(len(raw_ds["data"][i1]["paragraphs"])):
                            raw_ds["data"][i1]["paragraphs"][i2]['pid'] = "%d_%d"%(i1, i2)
                        new_raw_ds.append(raw_ds["data"][i1]["paragraphs"])
                    raw_ds = new_raw_ds
            d = []
            for raw_d in raw_ds:
                d.extend(raw_d)
            if not args.round_robin:
                if upsample_rate is None:
                    data += d
                    self.multitask_specific += [i]*len(d)
                else:
                    data += d * upsample_rate[i]
                    self.multitask_specific += [i]*(len(d)*upsample_rate[i])
                    logger.info(f"Upsample dataset {data_path} to {upsample_rate[i]} times with length: {len(d)} x {upsample_rate[i]}.")
            else:
                if upsample_rate is None:
                    temp_data[i] += d
                    total_len += len(d)
                else:
                    temp_data[i] += d * upsample_rate[i]
                    total_len += len(d * upsample_rate[i])
                    logger.info(f"Upsample dataset {data_path} to {upsample_rate[i]} times with length: {len(d)} x {upsample_rate[i]}.")

        if args.round_robin:
            for i in range(len(temp_data)):
                random.shuffle(temp_data[i])
            while not len(data) == total_len:
                for i in range(len(temp_data)):
                    try:
                        data.append(temp_data[i].pop())
                        self.multitask_specific += [i]
                    except:
                        pass
            logger.info("Round Robin shuffle done!")
        self.data = []
        self.max_a_len = 0
        if len(data_paths)==1 and data_paths[0] is not None and ('wiki' in data_paths[0] or 'woz' in data_paths[0]):
            #data = self._sort_by_index(data)
            #args.n_workers = 1
            if 'wiki' in data_paths[0]:
                answers_file = "wikisql_answers.json" 
            elif 'woz' in data_paths[0]:
                answers_file = "woz.en_answers.json" 
            with open(os.path.join(args.data_dir,answers_file),"r") as f:
                self.answers = json.load(f)
        if len(data) > 0:
            self.data_tokenization(data)

        self.c_len = len(self.data)

        if len(extra_data) > 0:
            extra_data = map(lambda x: self.etl_single_extra_data(x), extra_data)
            extra_data = list(filter(lambda x:x, extra_data))
            if args.gen_lm_sample_percentage > 0. and len(extra_data) == 0:
                logger.warning("No good extra data but sample percentage > 0!")
            self.is_extra = [0]*len(self.data) + [1]*len(extra_data)
            self.data += extra_data
        else:
            self.is_extra = [0]*len(self.data)

        self.example_idx = [i for i in range(len(self.data))]

    def etl_single_extra_data(self, data):
        gen_token = data[0]
        data = ' '.join([str(datum) for datum in data[1:]])
        try:
            if args.use_sep:
                context, qa = re.split(str(SPECIAL_TOKEN_IDS["sep_token"]), data)
            else:
                context = ""
                qa = data
            question, answer = re.split(str(SPECIAL_TOKEN_IDS["ans_token"]), qa)
            context = [int(c) for c in context.strip().split()]
            question = [int(q) for q in question.strip().split()]
            answer = [int(a) for a in re.sub(str(SPECIAL_TOKEN_IDS["eos_token"]), "", answer).strip().split()]
            uid = uuid.uuid1().hex
            data = self.parse_example(gen_token, context, question, answer, uid)
        except ValueError:
            return
        return data

    def concat_example(self, gen_token, c, sep_token, q, ans_token, a, eos_token):
        example = sep_token + q + ans_token + a
        if len(example) + 1 > args.max_len:
            logger.warning('an example with len {} is longer than max_len {}!'.format(len(example) + 1, args.max_len))
            limit_len = len(a) - (len(example) + 1 - args.max_len) - 128
            example = sep_token + q + ans_token + a[:limit_len]
            logger.warning('reduce A from {} to {}, total to len {}!'.format(len(a), limit_len, len(example) + 1))
        example = gen_token + c[:args.max_len-len(example)-1] + example + eos_token
        return example

    def parse_example(self, gen_token, context, question, answer, idx):
        if args.use_sep:
            cq_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], [], [])
            cqa_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], answer, [])
        else:
            cq_example = self.concat_example([], context, [], question, [self.ans_token], [], [])
            cqa_example = self.concat_example([], context, [], question, [self.ans_token], answer, [])
        Y_example = self.concat_example([], [], [], [], [], answer, [self.eos_token])
        Y_example = [FILL_VAL] * (len(cqa_example) - len(Y_example)) + Y_example
        if args.use_sep:
            gen_X_example = self.concat_example([gen_token], context, [self.sep_token], question, [self.ans_token], answer, [])
            gen_Y_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], answer, [self.eos_token])
        else:
            gen_X_example = self.concat_example([gen_token], context, [], question, [self.ans_token], answer, [])
            gen_Y_example = self.concat_example([], context, [], question, [self.ans_token], answer, [self.eos_token])
        return cq_example, len(cq_example), cqa_example, len(cqa_example), Y_example, gen_X_example, gen_Y_example, idx

    def parallel_tokenization(self, d):
        examples = []
        context = TOKENIZER.encode(d["context"])
        max_a_len = 0
        for i3, qa in enumerate(d["qas"]):
            question = TOKENIZER.encode(qa["question"])

            raw_answers = qa["answers"]
            if len(raw_answers) == 0:
                assert qa["is_impossible"]
                raw_answers.append({"text": ""})

            if self.data_type == "test":
                answer = []
                for i, raw_answer in enumerate(raw_answers):
                    answer.extend(TOKENIZER.encode(raw_answer["text"]))
                    if i != len(raw_answers) - 1:
                        answer.append(self.pad_token)
                max_a_len = max(max_a_len, len(answer))

                examples.append(self.parse_example(self.gen_token, context, question, answer, qa.get("id", 0 if not args.test_training_set else d["pid"]+"_%d"%i3)))
            else:
                answers = []
                for i, raw_answer in enumerate(raw_answers):
                    answer = TOKENIZER.encode(raw_answer["text"])
                    max_a_len = max(max_a_len, len(answer))
                    answers.append(answer)
                for j in range(len(answers)):
                    examples.append(self.parse_example(self.gen_token, context, question, answers[j], qa.get("id", 0)))
        return examples, max_a_len

    def data_tokenization(self, data):
        if args.debug:
            data = data[:10]
            new_data = []
            for datum in data:
                new_data.append(self.parallel_tokenization(datum))
            data = new_data
        else:
            with Pool(args.n_workers) as pool:
                data = pool.map(self.parallel_tokenization, data)
        for datum, max_a_len in data:
            self.data.extend(datum)
            self.max_a_len = max(self.max_a_len, max_a_len)

    def sort(self):
        self.data.sort(key=lambda x: len(x[0]))
        return self

    def sort_by_index(self):
        self.data.sort(key=lambda x: x[-1])

    def get_indices(self):
        return [d[-1] for d in self.data]

    #def _sort_by_index(self,data):
    #    datum = []
    #    for d in data:
    #        for qa in d["qas"]:
    #            datum.append({"context":d["context"], "qas":[qa]})
    #    datum.sort(key=lambda x:x["qas"][0]["id"])
    #    return datum

    def __len__(self):
        return len(self.data)

    def get_c_len(self):
        return self.c_len

    def __getitem__(self, index):
        if not args.multitask_specific:
            return self.data[index] + (self.is_extra[index], self.example_idx[index])
        else:
            return self.data[index] + (self.multitask_specific[index], )


class EarlyStopping:
    def __init__(self, logger,  patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.logger = logger

    def __call__(self, val_loss, model, model_dir):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir)
        elif score < self.best_score:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_dir):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save_pretrained(model_dir)
        TOKENIZER.save_pretrained(model_dir)
        self.val_loss_min = val_loss


def getsecond(obj):
    return obj[1]


def get_imp_dim(tgt_list, thres):
    tgt_list_sum = np.sum(tgt_list)
    tgt_list = [(j, tgt_list[j] / tgt_list_sum) for j in range(len(tgt_list))]
    tgt_list.sort(key=getsecond, reverse=True)
    acc = 0
    imp = []
    for i in range(len(tgt_list)):
        acc += tgt_list[i][1]
        imp.append((tgt_list[i][0], tgt_list[i][1]))
        if acc > thres:
            break
    return imp


class TrainStep:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.pasts_imp = np.zeros(1536 * 12 * args.preseqlen)

    def __call__(self, loss, scheduler_steps, pasts=None):

        # if pasts is not None:
            # pasts.retain_grad()
        # logger.info(loss)
        if args.seq_train_type in args.REG_TYPE_KEYS:
            reg_lambda = self.model.reg_params.get('lambda')
            for param in self.model.parameters():
                if param in self.model.reg_params:
                    reg_param = self.model.reg_params.get(param)
                    # logger.info(param)
                    # logger.info(reg_param.get('init_val'))
                    # logger.info(reg_param.get('omega'))

                    # This is not the approach of original LAMOL repo.
                    # but it is used by Mi et al. in https://github.com/MiFei/Continual-Learning-for-NLG
                    loss += args.reg_lambda * torch.sum(reg_param.get('omega') * (param - reg_param.get('init_val')) ** 2)

            # if torch.sum(reg_param.get('omega')) > 0:
            #     logger.info(loss)

        if not args.fp32:
            self.optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

        if not args.fp32:
            self.optimizer.update_master_grads()
            self.optimizer.clip_master_grads(args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

        if "gem" in args.seq_train_type and self.model.task_id >0: 
            store_grad(self.model.parameters, self.model.grads, self.model.grad_dims,self.model.task_id)
            indx = torch.cuda.LongTensor([i for i in range(self.model.task_id)])
            dotp = torch.mm(self.model.grads[:, self.model.task_id].unsqueeze(0),
                            self.model.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.model.grads[:, self.model.task_id].unsqueeze(1),
                              self.model.grads.index_select(1, indx), args.qp_margin)
                # copy gradients back
                overwrite_grad(self.model.parameters,
                               self.model.grads[:, self.model.task_id],
                               self.model.grad_dims)

        if args.seq_train_type in args.REG_TYPE_KEYS:
            self.optimizer.step(self.model.reg_params)
        else:
            self.optimizer.step()
        if args.fp32 or (not self.optimizer.overflow):
            for i in range(scheduler_steps):
                self.scheduler.step()
        self.optimizer.zero_grad()


class GEMStep:
    def __init__(self, model, parallel_model, train_loss_fct, optimizer):
        self.model = model
        self.parallel_model = parallel_model
        self.train_loss_fct = train_loss_fct
        self.optimizer = optimizer

    def __call__(self,current_task_id):
        for past_task_id, md in enumerate(args.memory_data):
            # Not saving current task's grads.
            if past_task_id >= current_task_id: return
            qadata = QADataset(None, "test", "gen", md)
            dataloader = create_dataloader(qadata, "test")
            grads_tmp = torch.zeros(sum(self.model.grad_dims),).cuda()
            if not args.fp32:
                grads_tmp = grads_tmp.half() 
            for _, _, cqa, _, Y, gen_X, gen_Y in dataloader:
                #CHECK
                n_inputs = sum(_cqa.shape[0] for _cqa in cqa)
                self.optimizer.zero_grad()
                for i in range(len(cqa)):
                    cqa[i] = (cqa[i].to(args.device_ids[i]),)
                    Y[i] = Y[i].to(args.device_ids[i])
                    gen_X[i] = (gen_X[i].to(args.device_ids[i]),)
                    gen_Y[i] = gen_Y[i].to(args.device_ids[i])

                losses = get_losses(self.parallel_model, cqa, Y, gen_X, gen_Y, self.train_loss_fct)
                loss = sum(losses)
                if not args.fp32:
                    self.optimizer.backward(loss, update_master_grads=False)
                else:
                    loss.backward()

                if not args.fp32:
                    #copy fp16 grads to fp32 grads  
                    self.optimizer.update_master_grads()
                    self.optimizer.clip_master_grads(args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                i = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        beg = 0 if i == 0 else sum(self.model.grad_dims[:i])
                        end = sum(self.model.grad_dims[:i+1])
                        grads_tmp[beg: end] += param.grad.data.view(-1)*n_inputs
                    i += 1

            grads_tmp /= len(qadata)
            self.model.grads[:, past_task_id].copy_(grads_tmp)
            self.optimizer.zero_grad()


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, data_type, max_batch_size, normal=False):
        self.dataset = dataset
        self.data_type = data_type
        self.normal = normal
        if data_type == "train":
            self.batch_size = args.train_batch_size
        else:
            self.batch_size = args.test_batch_size
        self.n_samples = len(dataset)
        self.max_batch_size = max_batch_size
        try:
            self.extra_start = dataset.is_extra.index(1)
        except ValueError:
            self.extra_start = self.n_samples
            logger.info('Attention, no extra data detected by sampler')

    def __iter__(self):
        if self.normal or not args.lamaml or (args.lamaml and self.extra_start == self.n_samples) or self.data_type == "test":
            if args.debug or self.data_type == "test" or args.z_debug_noshuff:
                indices = range(self.n_samples)
            else:
                indices = np.random.permutation(self.n_samples)
            max_len, cnt, st = 0, 0, 0
            batch = []
            for ed, idx in enumerate(indices):
                ln = len(self.dataset[idx][2])
                if max(max_len, ln)**LEN_FACTOR * (ed - st + 1) > self.batch_size[cnt] and len(batch) > 0:
                    st = ed
                    cnt += 1
                    max_len = 0
                    if cnt == args.n_gpus:
                        yield batch
                        cnt = 0
                        batch = []
                max_len = max(max_len, ln)
                # Modified for meta optimization
                # make sure that extra data are in the same half
                if args.random_batch:
                    batch.append(idx)
                else:
                    if idx < self.extra_start:
                        batch.append(idx)
                    else:
                        batch.insert(0, idx)
                if len(batch) == self.max_batch_size and self.data_type == "train":
                    yield batch
                    cnt, max_len, st = 0, 0, ed
                    batch = []
            if len(batch) > 0:
                yield batch
        # For fair comparision with our approach
        # it gaurantees the frequency of replaying old examples are more than usual
        # averagely speaking, the frequency of learning new examples and replaying old examples is 1:1
        else:
            if args.debug or self.data_type == "test" or args.z_debug_noshuff:
                c_indices = range(self.extra_start)
            else:
                c_indices = np.random.permutation(self.extra_start)
            # initialize extra_indices
            e_indices = []
            times = len(c_indices) // (self.n_samples - self.extra_start) + 1
            for _ in range(times):
                e_indices += (np.random.permutation(self.n_samples - self.extra_start) + self.extra_start).tolist()

            batch = []
            max_len, cnt, st = 0, 0, 0
            for ed, idx in enumerate(c_indices):
                ln = max(len(self.dataset[idx][2]), len(self.dataset[e_indices[ed]][2]))
                if max(max_len, ln)**LEN_FACTOR * 2 * (ed - st + 1) > self.batch_size[cnt] or (len(batch) == self.max_batch_size and self.data_type == "train"):
                    yield batch
                    cnt, max_len, st = 0, 0, ed
                    batch = []
                max_len = max(max_len, ln)
                if args.random_first:
                    if np.random.rand() > 0.5:
                        batch.insert(0, e_indices[ed])
                        batch.append(idx)
                    else:
                        batch.insert(0, idx)
                        batch.append(e_indices[ed])
                elif args.replay_first:
                    batch.insert(0, e_indices[ed])
                    batch.append(idx)
                else:
                    batch.insert(0, idx)
                    batch.append(e_indices[ed])

    def __len__(self):
        raise NotImplementedError


def create_dataloader(dataset, data_type, max_batch_size=1000000000, normal=False):
    if data_type == "train":
        batch_size = args.train_batch_size
    else:
        batch_size = args.test_batch_size

    if isinstance(batch_size, list):
        collate_fn=lambda x,bs=batch_size: dynamic_collate_fn(x, bs)
        shuffle = False
        batch_size = 1
        batch_sampler = DynamicBatchSampler(dataset, data_type, max_batch_size, normal)
        logger.info("Dynamic sampler and collate for {}".format(data_type))
    else:
        collate_fn=lambda x: varlen_collate_fn(x)
        shuffle = not (data_type != "train" or args.debug)
        batch_sampler = None

    if args.round_robin:
        shuffle = False
    dataloader =  DataLoader(dataset, num_workers=args.n_workers,
                             collate_fn=collate_fn,
                             shuffle=shuffle,
                             batch_size=batch_size,
                             batch_sampler=batch_sampler)
    return dataloader


class WrapModel(torch.nn.Module):
    def __init__(self, model):
        super(WrapModel, self).__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return outputs[0]


def remove_id(idx, need_process, all_pasts):
    assert idx in need_process
    del need_process[idx]
    for layer_id in range(MODEL_CONFIG.n_layer):
        all_pasts[layer_id][idx] = 0


def sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens):
    while len(need_process) > 0:
        first_id = next(iter(need_process))
        shortest_len = len(qa_results[first_id])
        # decode_batch_size = int(args.memory_sizes[0] * MEMORY_FACTOR[args.seq_train_type] // (shortest_len+1)**LEN_FACTOR)
        # set to a fixed number
        decode_batch_size = 32
        it = iter(need_process)
        stop = False
        remove_ids = []
        while not stop:
            batch_ids, input_ids, past = [], [], [[] for _ in range(MODEL_CONFIG.n_layer)]
            while True:
                try:
                    cur_id = next(it)
                    if len(qa_results[cur_id]) > shortest_len:
                        stop = True
                        break
                    batch_ids.append(cur_id)
                    if args.model_name == "gpt2":
                        input_ids.append(qa_results[cur_id][-1:])
                        for layer_id in range(MODEL_CONFIG.n_layer):
                            past[layer_id].append(all_pasts[layer_id][cur_id])
                    else:
                        input_ids.append(qa_results[cur_id])
                    if len(input_ids) == decode_batch_size:
                        break
                except StopIteration:
                    stop = True
                    break

            n_inputs = len(input_ids)
            if n_inputs == 0:
                break
            input_ids = torch.stack(input_ids)
            if args.model_name == "gpt2":
                for layer_id in range(MODEL_CONFIG.n_layer):
                    # after 538: fix bug!
                    past[layer_id] = torch.stack(past[layer_id], dim=1).cuda()
                all_outputs = model(input_ids=input_ids.cuda(), past=past)
            else:
                all_outputs = model(input_ids=input_ids.cuda())

            outputs = all_outputs[0]
            if args.model_name == "gpt2":
                pasts = all_outputs[1]

            next_logits = outputs[..., -1, :] / args.temperature_qa
            next_tokens = logits_to_tokens(next_logits).cpu()

            for i, cur_id in enumerate(batch_ids):
                if next_tokens[i] == SPECIAL_TOKEN_IDS["eos_token"]:
                    remove_ids.append(cur_id)
                else:
                    qa_results[cur_id] = torch.cat((qa_results[cur_id], next_tokens[i]))
                    if len(qa_results[cur_id]) in [max_tot_lens[cur_id], args.max_len]:
                        remove_ids.append(cur_id)
                    elif args.model_name == "gpt2":
                        for layer_id in range(MODEL_CONFIG.n_layer):
                            # after 537: trans to cpu
                            all_pasts[layer_id][cur_id] = pasts[layer_id][:, i].type(torch.float if args.fp32 else torch.half).cpu()
        for idx in remove_ids:
            remove_id(idx, need_process, all_pasts)


def write_extra_data(dump_path, qa_results):
    logger.info(f"writing extra data in {dump_path} ...")
    with open(dump_path,"w",newline="",encoding="utf-8") as f:
        lm_writer = csv.writer(f,delimiter=',')
        lm_writer.writerow(["gen"])
        for l in qa_results:
            lm_writer.writerow([l])


def parse_single_real_data(data,task):
    c = data["paragraphs"][0]["context"]
    q = data["paragraphs"][0]["qas"][0]["question"]
    a = data["paragraphs"][0]["qas"][0]["answers"][0]["text"]
    if args.use_sep:
        data = "{}{}{}{}{}{}{}".format(SPECIAL_TOKENS[task],c,SPECIAL_TOKENS["sep_token"],q,SPECIAL_TOKENS["ans_token"],a,SPECIAL_TOKENS["eos_token"])
    else:
        data = "{}{} {}{}{}{}".format(SPECIAL_TOKENS[task],c,q,SPECIAL_TOKENS["ans_token"],a,SPECIAL_TOKENS["eos_token"])
    return data


def get_real_data(task, train_extra_data, accum=True, encode=True):
    task_idx = args.tasks.index(task)
    gen_size = DATA_ATTRS[task]["train"]["data_size"]
    if accum:
        prev_tasks = args.tasks[:task_idx]
        gen_size = int(np.ceil(gen_size * args.gen_lm_sample_percentage))//len(prev_tasks)
    else:
        prev_tasks = [args.tasks[task_idx-1]]
        gen_size = int(gen_size * args.gen_lm_sample_percentage)

    datum = []
    for prev_task in prev_tasks:
        with open(TASK_DICT[prev_task]["train"],"r") as f:
            data = data_expand(json.load(f)["data"])
        indices = np.random.choice(range(len(data)), gen_size)
        for i in indices:
            d = parse_single_real_data(data[i],prev_task)
            datum.append(d)
            if encode:
                train_extra_data.append(TOKENIZER.encode(d))

    model_dir = get_model_dir([prev_task])
    dump_path = os.path.join(model_dir,"real.csv")
    write_extra_data(dump_path, datum)
    return dump_path


def read_extra_data(gen_path, train_extra_data):
    with open(gen_path,"r") as lm_file:
        reader = csv.reader(lm_file,delimiter=',')
        next(reader)
        for row in reader:
            row = TOKENIZER.encode(row[0].strip()) 
            train_extra_data.append(row)


def create_extra_data(task, prev_task, model, train_extra_data, prefix=None, gen_path=None):
    if args.real_sample:
        logger.info(f"using real data as extra data")
        return get_real_data(task, train_extra_data)
    task_cnt = args.tasks.index(task)
    model_dir = get_model_dir([prev_task])
    if gen_path is None:
        gen_path = os.path.join(model_dir,"lm.csv")
    if os.path.exists(gen_path):
        logger.info(f"extra data exists in {gen_path}, read it!")
        return read_extra_data(gen_path, train_extra_data) 
    gen_size = DATA_ATTRS[task]["train"]["data_size"]
    # generate 3 time more examples
    gen_size = int(np.ceil(gen_size * args.gen_lm_sample_percentage))
    gen_size -= (gen_size % task_cnt)

    if args.debug:
        gen_size = task_cnt

    model.eval()

    need_process = OrderedDict()
    qa_results = []
    for task_name in args.tasks[:task_cnt]:
        qa_results.extend([torch.tensor([SPECIAL_TOKEN_IDS[task_name]]) for _ in range(gen_size//task_cnt)])
    if prefix is None:
        all_pasts = [[
            torch.empty(2, MODEL_CONFIG.n_head, 0, MODEL_CONFIG.n_embd//MODEL_CONFIG.n_head,
                dtype=torch.float if args.fp32 else torch.half)
            for _ in range(gen_size)
        ] for __ in range(MODEL_CONFIG.n_layer)]
    else:
        all_pasts = [[prefix[i] for j in range(gen_size)] for i in range(MODEL_CONFIG.n_layer)]
    max_tot_lens = [args.max_len for _ in range(gen_size)]

    with torch.no_grad():
        for i in range(gen_size):
            need_process.update([[i, None]])
            """ trace back 0505
            if i % 100 == 0:
                sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)
            """
        sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)

    model.train()

    qa_results = [res.tolist() for res in qa_results]
    """
    # -------------------- Filtering bad examples---------------------------#
    print("before filtering: {}".format(len(qa_results)))
    model.cpu()
    from pytorch_transformers import GPT2Tokenizer
    from pytorch_transformers import GPT2LMHeadModel, GPT2Config

    cnt = 0
    train_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    clean_qa = []
    for task_name in args.tasks[:task_cnt]:
        print("Task: {} filtering, cnt : {}".format(task_name, cnt))

        c_cnt = 0
        model_dir = get_model_dir([task_name])
        local_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        with open(os.path.join(model_dir, "added_tokens.json"), "r") as f:
            raw_ds = json.load(f)
            for i in raw_ds:
                local_tokenizer.add_tokens([i])
        local_tokenizer_len = len(local_tokenizer)

        local_model_config = GPT2Config.from_json_file(os.path.join(model_dir, "config.json"))
        local_model = GPT2LMHeadModel(local_model_config).cuda().eval()
        state_dict = torch.load(os.path.join(model_dir, "model-finish"), map_location='cuda:0')
        local_model.load_state_dict(state_dict)

        for _ in range(gen_size // task_cnt):
            if c_cnt >= int(0.25 * gen_size // task_cnt):
                cnt += 1
                continue

            gen_cqa = qa_results[cnt]
            cnt += 1

            legal = True
            for token_id in gen_cqa:
                if token_id >= local_tokenizer_len:
                    legal = False
                    print("Illegal Example!!! local_tokenizer_len: {}, token_id : {}".format(local_tokenizer_len, token_id))
                    print(gen_cqa)
                    print(TOKENIZER.decode(gen_cqa))
                    break
            if not legal:
                continue

            gen_first = gen_cqa[0]
            should_first = local_tokenizer.convert_tokens_to_ids(get_gen_token(task_name))
            assert should_first == gen_first, "Wrong Filtering models!!! {} {}".format(should_first, gen_first)
            tgt_cqa = gen_cqa[1:] + [local_tokenizer.convert_tokens_to_ids('<|endoftext|>')]
            inputs = torch.tensor(gen_cqa).unsqueeze(0).cuda()
            labels = torch.tensor(tgt_cqa).cuda()
            output = local_model(inputs)
            loss = train_loss_fct(output[0].squeeze(0), labels)
            this_ppl = torch.exp(loss.mean()).item()
            if this_ppl <= args.ppl_thr:
                c_cnt += 1
                clean_qa.append(gen_cqa)
            else:
                pass
                # print("Filtering out! ppl: {}".format(this_ppl))
                # print(gen_cqa)
                # print(local_tokenizer.decode(gen_cqa))
        print("Task: {}, generated examples: {}".format(task_name, c_cnt))
    qa_results = clean_qa
    model.cuda()
    """

    qa_text = []
    qa_final_res = []
    for res in qa_results:
        try:
            tmp = TOKENIZER.decode(res)
            qa_text.append(tmp)
            qa_final_res.append(res)
        except TypeError:
            logger.info("TypeError {}".format(res))

    train_extra_data.extend(qa_final_res)
    # qa_results = [TOKENIZER.decode(res) for res in qa_results]

    write_extra_data(gen_path, qa_text)


def logits_to_tokens(next_logits):
    filtered_logits = top_k_top_p_filtering(next_logits, top_k=args.top_k_qa, top_p=args.top_p_qa)
    log_probs = F.softmax(filtered_logits, dim=-1)
    next_tokens = torch.multinomial(log_probs, num_samples=1)
    return next_tokens

 
def lll_unbound_setting(split_size=10,data_type="train",test_target="self"):
    data_dir = os.path.join(args.data_dir,"{}_{}".format("_".join(args.tasks),args.gen_lm_sample_percentage))
    if data_type == "test":
        args.splitted_tasks = [f"task_{i}" for i in range(split_size)]
        args.n_train_epochs = {task: args.n_train_epochs for task in args.splitted_tasks}
        if test_target in ["self","all"]:
            for no in range(split_size):  
                task = f"task_{no}" 
                test_data_path = os.path.join(data_dir,f"{task}-test.json")
                TASK_DICT[task] = {}
                TASK_DICT[task]["test"] = test_data_path
            if test_target == "all":
                args.tasks += args.splitted_tasks
            else:
                args.tasks = args.splitted_tasks
    elif data_type == "train":
        create_lll_unbound_data(split_size)
        args.n_train_epochs = {task: args.n_train_epochs for task in args.tasks}
    return TASK_DICT


def create_lll_unbound_data(split_size=10): 
    data_dir = os.path.join(args.data_dir,"{}_{}".format("_".join(args.tasks),args.gen_lm_sample_percentage))
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    datum = [] 
    test_datum = []
    data_sizes = [] 
    chunk_sizes = []
    for task in args.tasks:
        train_data_path = TASK_DICT[task]["train"]
        with open(train_data_path, "r") as f:
            data = json.load(f)["data"]
            data = data_expand(data)
            data_sizes.append(len(data))
            datum += data
        test_data_path = TASK_DICT[task]["test"]
        with open(test_data_path, "r") as f:
            data = json.load(f)["data"]
            data = data_expand(data)
            test_datum.append(data) 
    chunk_size = int(np.ceil(len(datum)/split_size))

    tasks = []
    for no, i in enumerate(range(0, len(datum), chunk_size)):  
        task = f"task_{no}" 
        tasks.append(task)
        chunk = datum[i:i + chunk_size] if i < len(datum)-chunk_size else datum[i:]
        chunk_sizes.append(len(chunk))
        DATA_ATTRS[task] = {"train":{"data_size":None}}
        DATA_ATTRS[task]["train"]["data_size"] = len(chunk)
        train_data_path = os.path.join(data_dir,f"{task}-train.json")
        with open(train_data_path,"w") as f:
            json.dump({"data":chunk},f)
        TASK_DICT[task] = {}
        TASK_DICT[task]["train"] = train_data_path
    args.tasks = tasks

    sis = get_split_indices(data_sizes,chunk_sizes)
    test_split = []
    for dic in sis.values():
        merged_data = []
        for k,v in dic.items():
            from_index = int(len(test_datum[k])*v[0])
            to_index = int(len(test_datum[k])*v[1])
            merged_data+= test_datum[k][from_index:to_index]
        test_split.append(merged_data)

    for no, chunk in enumerate(test_split):  
        task = f"task_{no}" 
        test_data_path = os.path.join(data_dir,f"{task}-test.json")
        with open(test_data_path,"w") as f:
            json.dump({"data":chunk},f)
        TASK_DICT[task]["test"] = test_data_path


def data_expand(data):
    datum = []
    for d in data:
        para = d["paragraphs"]
        for p in para: 
            for qa in p["qas"]:
                d = {"context": p["context"], "qas": [qa]}
                datum.append({"paragraphs":[d]})
    return datum


def get_split_indices(data_sizes,chunk_sizes):
    ds = deepcopy(data_sizes)
    records = {}
    tmp = {}
    order = 0 # data_sizes index
    i = 0 # chunk_sizes index
    while len(data_sizes)>0:
        d0 = data_sizes[0]
        c0 = chunk_sizes[0]
        if d0>c0:
            val = c0/ds[order]
        else:
            val = d0/ds[order]

        if order not in tmp:
            rec = (0,val)
            tmp[order] = val
        else:
            rec = (tmp[order],tmp[order]+val)
            tmp[order] += val
        if i in records:
            records[i][order] = rec
        else:
            records[i] = {order: rec}

        if d0>c0:
            data_sizes[0]-=c0
            chunk_sizes.pop(0)
            i+=1
        else:
            chunk_sizes[0]-=d0
            data_sizes.pop(0)
            order+=1
            if d0==c0:
                chunk_sizes.pop(0)
                i+=1
    return records


def store_grad(get_ps, grads, grad_dims, task_id): 
    i = 0
    for param in get_ps():
        if param.grad is not None:
            beg = 0 if i == 0 else sum(grad_dims[:i])
            end = sum(grad_dims[:i+1])
            grads[beg: end, task_id].copy_(param.grad.data.view(-1))
        i += 1


def overwrite_grad(pp, newgrad, grad_dims):
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))
