import torch
from torch.utils.data import DataLoader
from torch import nn
from pytorch_transformers import AdamW, WEIGHTS_NAME, WarmupLinearSchedule
import csv
import random
import numpy as np
import os
import logging
from fp16 import FP16_Module, FP16_Optimizer
from parallel import DataParallelModel, DataParallelCriterion
from collections import OrderedDict
from utils import *
from settings import args, TASK_DICT, init_logging, MODEL_CONFIG, MODEL_CLASS, SPECIAL_TOKENS, CONFIG_CLASS
from settings import TOKENIZER, SPECIAL_TOKEN_IDS, FILL_VAL, SAVE_NAME, FINAL_SAVE_NAME, TOKENS_WEIGHT, CONFIG_NAME
from scheduler import AnnealingLR
from regularizers import REG_TYPES, REG_TYPE_KEYS, Weight_Regularized_AdamW, Weight_Regularized_SGD
from torch.nn import CrossEntropyLoss
logger = logging.getLogger(__name__)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def swap_name(org_name, seq_distil, ref1):
    # swap_name(TASK_DICT[t]["train"], args.seq_distil, args.ref1)
    if not seq_distil and not ref1:
        return org_name
    if seq_distil:
        return org_name.replace("train", "distil")
    if ref1:
        return org_name.replace("train", "ref1")

def validation(p_model, valid_dataloader, train_loss_fct):
    cum_loss = 0.0
    cum_qa_loss = 0.0
    cum_lm_loss = 0.0
    cur_n_inputs = 0
    with torch.no_grad():
        p_model.eval()
        for (_, _, cqa, _, Y, gen_X, gen_Y, is_extra, idx) in valid_dataloader:
            torch.cuda.empty_cache()
            n_inputs = cqa[0].shape[0]
            # qa_loss, lm_loss = p_model.observe_val([cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda()], train_loss_fct)
            qa_loss, lm_loss = get_losses(p_model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), train_loss_fct)
            cum_loss += (qa_loss + lm_loss) * n_inputs
            cum_qa_loss += qa_loss * n_inputs
            cum_lm_loss += lm_loss * n_inputs
            cur_n_inputs += n_inputs
    return cum_loss / cur_n_inputs, cum_qa_loss / cur_n_inputs, cum_lm_loss / cur_n_inputs

def train(task_ids, model):
            
    tasks = [args.tasks[task_id] for task_id in task_ids]

    logger.info("start to train { task: %s, seq train type: %s }" % (tasks, args.seq_train_type))
    model_dir = get_model_dir(tasks)
    make_dir(model_dir)

    #train_dataset = [(TASK_DICT[t]["train"] if not args.seq_distil else TASK_DICT[t]["train"].replace("train", "distil")) for t in tasks]
    train_dataset = [swap_name(TASK_DICT[t]["train"], args.seq_distil, args.ref1) for t in tasks]
    valid_dataset = [TASK_DICT[t]["test"] for t in tasks]
    train_extra_data = []
    if "lll" in args.seq_train_type and task_ids[0] > 0 and not args.skip_tasks:
        prev_task = args.tasks[task_ids[0]-1]
        with torch.no_grad():
            create_extra_data(tasks[0], prev_task, model, train_extra_data)
    elif "gem" in args.seq_train_type and task_ids[0] > 0: 
        get_real_data(tasks[0], train_extra_data, accum=False, encode=True)
        args.memory_data.append(train_extra_data)
        train_extra_data = []
    logger.info('extra training data size: {}'.format(len(train_extra_data)))

    if not model:
        # which_model_to_load = model_dir if os.path.isfile(os.path.join(model_dir, FINAL_SAVE_NAME)) else args.model_name
        model = MODEL_CLASS.from_pretrained('./PModel').cuda()
        model.resize_token_embeddings(len(TOKENIZER))
        if not args.fp32:
            model = FP16_Module(model)
    
    gen_token = get_gen_token(tasks[0])
    TOKENIZER.add_tokens([gen_token])
    TOKENIZER.save_pretrained(model_dir)
    SPECIAL_TOKENS[tasks[0]] = gen_token
    SPECIAL_TOKEN_IDS[tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
    
    # no need for reg
    if args.seq_train_type not in REG_TYPE_KEYS:
        logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[tasks[0]]))

    global TOKENS_WEIGHT
    if args.seq_train_type not in REG_TYPE_KEYS:
        if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
            TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))

    MODEL_CONFIG.vocab_size = TOKENS_WEIGHT.shape[0]
    MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
    model.vocab_size = TOKENS_WEIGHT.shape[0]

    if args.skip_tasks and len(tasks) == 1:
        logger.info("*********** skip task: {} ***********".format(tasks[0]))
        if tasks[0] in args.skip_tasks:
            if len(args.skip_tasks) == 1:
                model_dir = get_model_dir(tasks)
                model_path = os.path.join(model_dir, FINAL_SAVE_NAME)
                config_path = os.path.join(model_dir,CONFIG_NAME)
                model_config = CONFIG_CLASS.from_json_file(config_path)
                model = MODEL_CLASS(model_config).cuda()
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
                if not args.fp32:
                    model = FP16_Module(model)
                if args.seq_train_type in REG_TYPE_KEYS:
                    logger.info("calulating reg_params ...")
                    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
                    max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
                    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
                    parallel_model = DataParallelModel(WrapModel(model), args.device_ids)
                    regularizer = REG_TYPES[args.seq_train_type](model, parallel_model, [train_dataloader], tasks[0])
                    regularizer.task_start_do()
                    regularizer.task_end_do()
                    torch.save(model.state_dict(), os.path.join(model_dir, FINAL_SAVE_NAME))
                    logger.info("done reg_params!")
            args.skip_tasks.remove(tasks[0])
            return model

    # no need for reg
    if args.seq_train_type not in REG_TYPE_KEYS:
        model.resize_token_embeddings(len(TOKENIZER) if not args.multitask_specific else len(TOKENIZER)+4)

    if args.multitask_specific:
        for i in range(4):
            TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
    if args.distil:
        teacher_model = MODEL_CLASS.from_pretrained('./PModel').cuda()
        teacher_vocab_size = json.load(open("models/gpt2/lll/{task}_0.2/{task}/config.json".format(task=tasks[0])))['vocab_size']
        teacher_model.resize_token_embeddings(teacher_vocab_size)
        print("load teacher model from {}".format("models/gpt2/lll/{task}_0.2/{task}/model-finish".format(task=tasks[0])))
        teacher_model.load_state_dict(torch.load("models/gpt2/lll/{task}_0.2/{task}/model-finish".format(task=tasks[0])))
        if not args.fp32:
            teacher_model = FP16_Module(teacher_model)
        teacher_model.eval()
        teacher_model = DataParallelModel(WrapModel(teacher_model), args.device_ids)

    if not args.fp32:  # again because resize_token_embeddings makes embedding layer fp32
        model = FP16_Module(model)

    # parallel_model = DataParallelModel(WrapModel(model), args.device_ids)
    parallel_model = model

    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
    valid_qadata = QADataset(valid_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]])

    # max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
    max_train_batch_size = 4

    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
    valid_dataloader = create_dataloader(valid_qadata, "test")

    if not args.unbound and args.seq_train_type not in ["multitask", "multilm"]:
        # n_train_epochs = TASK_DICT[tasks[0]]["n_train_epochs"]
        n_train_epochs = args.n_train_epochs[tasks[0]]
    else:
        n_train_epochs = args.n_train_epochs['_'.join(tasks)]
    n_train_optimization_steps = len(train_qadata) * n_train_epochs
    logger.info('len of train dataset: {} , max train batch size {} , num of opt steps: {}'.format(
        len(train_qadata), max_train_batch_size, n_train_optimization_steps))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
                
    if "gem" in args.seq_train_type:
        model.task_id = task_ids[0]
        if not hasattr(model, "grad_dims"):
            model.grad_dims = []
            for param in model.parameters():
                model.grad_dims.append(param.data.numel())
        if not hasattr(model, "grads"):
            model.grads = torch.zeros(sum(model.grad_dims),len(args.tasks))
            model.grads = model.grads.cuda()

    if args.seq_train_type in REG_TYPE_KEYS:
        optimizer = Weight_Regularized_AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if not args.fp32:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                   dynamic_loss_args={'scale_window': 100, 'min_scale': 1, 'delayed_shift': 2})

    if not args.lamaml or (args.lamaml and task_ids[0] == 0):
        scheduler = AnnealingLR(optimizer, start_lr=args.learning_rate, warmup_iter=int(args.n_warmup_ratio*len(train_qadata)), 
                                num_iters=int(n_train_optimization_steps), decay_style=args.decay_style)
    else:
        scheduler = AnnealingLR(optimizer, start_lr=args.learning_rate, warmup_iter=int(args.n_warmup_ratio*len(train_qadata)), 
                                num_iters=int(train_qadata.get_c_len() * 2 * n_train_epochs) + 100, decay_style=args.decay_style)

    # train_loss_fct = DataParallelCriterion(CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT), args.device_ids)
    train_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT)
    if args.distil:
        kd_loss_fct = DataParallelCriterion(nn.KLDivLoss(reduction="batchmean"), args.device_ids)

    if args.seq_train_type in REG_TYPE_KEYS:
        copy_train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
        prev_task = args.tasks[task_ids[0]-1]
        regularizer = REG_TYPES[args.seq_train_type](model, parallel_model, [copy_train_dataloader], tasks[0], prev_task)
        regularizer.task_start_do()

    tot_n_steps = 0
    train_once = TrainStep(model, optimizer, scheduler)
    if "gem" in args.seq_train_type and task_ids[0] != 0:
        gem_step = GEMStep(model, parallel_model, train_loss_fct, optimizer)
    
    show_list = []
    for param in model.parameters():
        if param.requires_grad:
            # logger.info(param)
            # logger.info(model.reg_params.get(param))
            show_list.append([param, model.reg_params.get(param)])
    logger.info(show_list[0])

    for group in optimizer.param_groups:
        print(group['lr'])

    for ep in range(n_train_epochs):
        model.train()
        cum_loss, cum_qa_loss, cum_lm_loss, cur_n_inputs = 0, 0, 0, 0
        for n_steps, (_, _, cqa, _, Y, gen_X, gen_Y, is_extra, idx) in enumerate(train_dataloader):
            n_inputs = cqa[0].shape[0]
            if cqa[0].shape[1] > 700:
                logger.info(cqa[0].shape)
                continue
            if args.multitask_specific:
                for i in range(len(is_extra)):
                    gen_X[i][:, 0] += is_extra[i]
                    is_extra[i] = is_extra[i] * 0

            if args.distil:
                losses = get_distil_losses(teacher_model, parallel_model, cqa, Y, gen_X, gen_Y, is_extra, kd_loss_fct, train_loss_fct, args.temperature_kd, pad_idx=FILL_VAL)
            else:
                losses = get_losses(parallel_model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), train_loss_fct)
            loss = sum(losses)
            if "gem" in args.seq_train_type and task_ids[0] != 0:
                gem_step(task_ids[0])
            train_once(loss, n_inputs)

            qa_loss = losses[0].item() * n_inputs
            lm_loss = losses[1].item() * n_inputs
            cum_loss += (qa_loss + lm_loss)
            cum_qa_loss += qa_loss
            cum_lm_loss += lm_loss
            cur_n_inputs += n_inputs

            if (n_steps + 1 ) % args.logging_steps == 0:
                logger.info('progress {:.3f} , lr {:.1E} , loss {:.3f} , qa loss {:.3f} , lm loss {:.3f} , avg batch size {:.1f}'.format(
                    ep + cur_n_inputs/len(train_qadata), scheduler.get_lr(), cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs,
                    cur_n_inputs/(n_steps + 1)
                ))
        if ep + 1 == 9:
            torch.save(model.state_dict(), os.path.join(model_dir, SAVE_NAME+str(ep+1)))
            logger.info("MODEL SAVED!")
        tot_n_steps += (n_steps + 1)
        val_loss, val_qa_loss, val_lm_loss = validation(parallel_model, valid_dataloader, train_loss_fct)
        logger.info('epoch {}/{} done , tot steps {} , lr {:.1E} , loss {:.2f} , qa loss {:.2f} , lm loss {:.2f}, val loss {:.2f}, vqa loss {:.2f}, vlm loss {:.2f}, avg batch size {:.1f}'.format(
            ep+1, n_train_epochs, tot_n_steps, scheduler.get_lr(), cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs, val_loss, 
            val_qa_loss, val_lm_loss, cur_n_inputs/(n_steps+1)
        ))

    # task end do for reg
    if args.seq_train_type in REG_TYPE_KEYS:
        regularizer.task_end_do()
    # torch.save(model.state_dict(), os.path.join(model_dir, FINAL_SAVE_NAME))

    return model


if __name__ == '__main__':

    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)

    if not args.z_debug:
        make_dir(args.model_dir_root)

        init_logging(os.path.join(args.model_dir_root, 'log_train.txt'))
        logger.info('args = {}'.format(str(args)))

        model = None
        if args.seq_train_type in ["multitask", "multilm"]:
            model = train(list(range(len(args.tasks))), model)
        else:
            if args.unbound:
                TASK_DICT = lll_unbound_setting(split_size=args.unbound)
            for task_id in range(len(args.tasks)):
                model = train([task_id], model)
    else:
        init_logging(os.path.join(args.model_dir_root, 'log_train_debug.txt'))
        logger.info('args = {}'.format(str(args)))

        model = None
        if args.z_debug_tsk_num >= 1:
            from pytorch_transformers import GPT2LMHeadModel, GPT2Config

            tasks = [args.tasks[args.z_debug_tsk_num - 1]]
            model_dir = get_model_dir(tasks)
            model_config = GPT2Config.from_json_file(os.path.join(model_dir, "config.json"))
            model = GPT2LMHeadModel(model_config).cuda().eval()
            state_dict = torch.load(os.path.join(model_dir, "model-9"), map_location='cuda:0')
            model.load_state_dict(state_dict)

            global TOKENS_WEIGHT
            tsk_cnt = 0
            while tsk_cnt < args.z_debug_tsk_num:
                TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
                gen_token = get_gen_token(args.tasks[tsk_cnt])
                TOKENIZER.add_tokens([gen_token])
                SPECIAL_TOKENS[args.tasks[tsk_cnt]] = gen_token
                SPECIAL_TOKEN_IDS[args.tasks[tsk_cnt]] = TOKENIZER.convert_tokens_to_ids(gen_token)
                tsk_cnt += 1

            model.resize_token_embeddings(len(TOKENIZER))
            if not args.fp32:
                model = FP16_Module(model)
        for task_id in range(args.z_debug_tsk_num, len(args.tasks)):
            model = train([task_id], model)
