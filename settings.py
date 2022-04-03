import os
import json
import argparse
import logging
import datetime
logger = logging.getLogger(__name__)

import GPUtil
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, CONFIG_NAME 
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILL_VAL = -1
# for preseqlen=10 pretraining
# LEN_FACTOR = 1.08
# original setting
# LEN_FACTOR = 1.163
# lamaml setting
LEN_FACTOR = 1.19
MEMORY_FACTOR = {
    "finetune": 0.58,
    "multitask": 0.58,
    "multilm": 0.35,
    "lll": 0.35,
    "ewc": 0.30,
    "mas": 0.18,
    "gem": 0.50,
}
TURING_ARCHS = {'Tesla V100', '2080 Ti'}
MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig),
}
SAVE_NAME = 'model-'
FINAL_SAVE_NAME = 'model-finish'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adam_epsilon", default=1e-4, type=float)

    parser.add_argument("--add_task_tokens", action="store_true")
    parser.add_argument("--use_eos_as_sos", action="store_true")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--decay_style", type=str, default="linear")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--real_sample", action="store_true")
    parser.add_argument("--unbound", type=int, default=0)
    parser.add_argument("--gen_lm_sample_percentage", type=float, default=0.05)
    parser.add_argument("--distil", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=6.25e-5)
    parser.add_argument("--logging_steps", type=int, default=100)

    parser.add_argument("--qa_theta", type=float, default=1.0)
    parser.add_argument("--lm_lambda", type=float, default=0.25)

    parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--max_n_epochs", type=int, default=9)
    parser.add_argument("--min_batch_size", type=int, default=4)
    parser.add_argument("--min_n_steps", type=int, default=1500)
    parser.add_argument("--model_dir_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt2", choices=["gpt2", "openai-gpt"])
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_train_epochs", type=int, default=9)
    parser.add_argument("--dynamic_epochs", action="store_true")
    parser.add_argument("--n_warmup_ratio", type=float, default=0.005)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--use_sep", action="store_true")
    parser.add_argument("--reg_lambda", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_all", action="store_true")
    parser.add_argument("--test_training_set", action="store_true")
    parser.add_argument("--round_robin", action="store_true")
    parser.add_argument("--upsample_data", type=str, default=None)
    parser.add_argument("--seq_distil", action="store_true")
    parser.add_argument("--extra_e2e", action="store_true")
    parser.add_argument("--ref1", action="store_true")
    parser.add_argument("--multitask_specific", action="store_true")
    parser.add_argument("--seq_train_type", type=str, default="lll", choices=["lll","finetune","multitask","mas","ewc","gem","multilm"])
    parser.add_argument("--tasks", nargs='+', default=["squad2"])
    parser.add_argument("--skip_tasks", nargs='+')
    parser.add_argument("--temperature_kd", type=float, default=2.0)
    parser.add_argument("--temperature_lm", type=float, default=1.0)
    parser.add_argument("--temperature_qa", type=float, default=1.0)
    parser.add_argument("--test_batch_size", type=int, default=0)
    parser.add_argument("--tokens_weight", type=float, default=5)
    parser.add_argument("--top_k_lm", type=int, default=20)
    parser.add_argument("--top_k_qa", type=int, default=20)
    parser.add_argument("--top_p_lm", type=float, default=0.)
    parser.add_argument("--top_p_qa", type=float, default=0.)
    parser.add_argument("--train_batch_size", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--qp_margin", type=float, default=0.5)

    # added args
    parser.add_argument("--z_debug", action="store_true")
    parser.add_argument("--z_debug_tsk_num", type=int, default=0)
    parser.add_argument("--z_debug_dir", type=str, default="None")
    parser.add_argument("--z_debug_model", type=str, default="None")
    parser.add_argument("--ppl_thr", type=float, default=100.0)

    parser.add_argument("--z_learning_rate", type=float, default=1e-3)
    parser.add_argument("--z_warmup_step", type=int, default=1000)
    parser.add_argument("--z_step", type=int, default=300)
    parser.add_argument("--preseqlen", type=int, default=10)
    parser.add_argument("--mid_dim", type=int, default=256)

    parser.add_argument("--gradient_block", action="store_true")
    parser.add_argument("--mom", type=float, default=0.1)
    parser.add_argument("--no_repara", action="store_true")
    parser.add_argument("--imp_p", type=float, default=0.9)
    parser.add_argument("--first_only", action="store_true")

    parser.add_argument("--lamaml", action="store_true")

    parser.add_argument("--second_order", action="store_true")
    parser.add_argument("--learn_lr", action="store_true")
    parser.add_argument("--sync_update", action="store_true")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument('--opt_lr', type=float, default=1e-3)
    parser.add_argument('--opt_wt', type=float, default=1e-3)
    parser.add_argument('--alpha_init', type=float, default=1e-3)
    parser.add_argument('--glances', nargs='+', type=int, default=[1, 1, 1, 1, 1])

    parser.add_argument("--pre_learning_rate", type=float, default=1e-4)
    parser.add_argument("--pre_warmup_step", type=int, default=100)
    parser.add_argument("--pre_start_from", type=str, default="None")

    parser.add_argument("--pretrained_prefix", type=str, default="None")
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--z_max_batch_size", type=int, default=32)

    parser.add_argument("--test_skip", type=int, default=0)
    parser.add_argument("--meta_last", action="store_true")
    parser.add_argument("--observe_type", type=int, default=1)
    parser.add_argument("--rep_beta", type=float, default=0.1)
    parser.add_argument("--half_assert", action="store_true")
    parser.add_argument('--thr', nargs='+', type=float, default=[0, 0, 0, 0, 0])
    parser.add_argument("--random_batch", action="store_true")

    parser.add_argument("--replay_first", action="store_true")
    parser.add_argument("--random_first", action="store_true")

    parser.add_argument('--grad_coe', nargs='+', type=float, default=[1.0, 1.0])
    parser.add_argument("--return_m_grads", action="store_true")
    parser.add_argument("--last_half_is_whole", action="store_true")
    parser.add_argument("--first_half_is_old", action="store_true")
    parser.add_argument("--maml_qalm", action="store_true")
    parser.add_argument("--block_first_meta", action="store_true")
    parser.add_argument("--sync_dropout", action="store_true")
    parser.add_argument("--same_pass", action="store_true")

    parser.add_argument('--task_test', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--grad_ob", action="store_true")
    parser.add_argument("--z_debug_noshuff", action="store_true")
    parser.add_argument("--mask_neg", action="store_true")
    parser.add_argument("--mask_pos", action="store_true")
    parser.add_argument("--tanh_trans", action="store_true")
    parser.add_argument("--alpha_temp", type=float, default=1e-8)
    parser.add_argument("--alpha_b", type=float, default=0.0)
    parser.add_argument("--meta_block_emb", action="store_true")
    parser.add_argument("--no_refresh", action="store_true")
    parser.add_argument("--use_momentum", action="store_true")
    parser.add_argument("--learn_lr_with_nomul", action="store_true")

    parser.add_argument("--alpha_dropout", type=float, default=0.0)
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--pretraining", action="store_true")
    parser.add_argument("--prefix_dropout", type=float, default=0.0)
    parser.add_argument('--set_p_drop', nargs='+', type=int, default=[0, 1, 0])
    parser.add_argument("--use_autoM", action="store_true")
    parser.add_argument("--lr_epoch", type=int, default=1)
    parser.add_argument("--single_alpha", action="store_true")
    parser.add_argument("--rev_gradient", action="store_true")
    parser.add_argument("--lr_grad_norm", type=float, default=1.0)
    parser.add_argument('--wt_op', nargs='+', type=bool, default=[True, False, False, True])

    parser.add_argument("--id", type=int, default=1)
    args = parser.parse_args()

    if args.debug:
        args.logging_steps = 1
        torch.manual_seed(0)
        torch.backends.cudnn.deterministric = True

    args.model_dir_root = os.path.join(args.model_dir_root, args.model_name,
            args.seq_train_type, "{}_{}_{}_{}".format(args.id, "_".join(args.tasks),
                args.gen_lm_sample_percentage, args.seed) if "lll" in args.seq_train_type else "_".join(args.tasks)+"_seed_%d"%args.seed)
    logger.info("DIR ROOT CHANGED! {}".format(args.model_dir_root))

    args.device_ids = GPUtil.getAvailable(maxLoad=0.05, maxMemory=0.05, limit=args.n_gpus)
    if len(args.device_ids) == 0:
        logger.error('No available GPUs!')
        raise NotImplementedError("No CPU mode available!")

    if len(args.device_ids) < args.n_gpus:
        logger.warning('Available number of GPU = {} < n_gpus = {}'.format(len(args.device_ids), args.n_gpus))
        args.n_gpus = len(args.device_ids)
        logger.warning('Continue training with {} GPUs'.format(args.n_gpus))

    torch.cuda.set_device(args.device_ids[0])

    gpus = GPUtil.getGPUs()
    gpu_names = [gpus[device_id].name for device_id in args.device_ids]
    if not all(any(turing_arch in gpu_name for turing_arch in TURING_ARCHS) for gpu_name in gpu_names):
        logger.warning('Not all gpus support fp16 training! Will use fp32 instead.')
        args.fp32 = True
    if args.model_name == "openai-gpt":
        logger.warning('openai-gpt is not supported now!')
        exit(0)
    if not args.fp32:
        global MEMORY_FACTOR
        if args.distil:
            MEMORY_FACTOR = dict([k, v*0.7] for k, v in MEMORY_FACTOR.items())
        else:
            MEMORY_FACTOR = dict([k, v*1.4] for k, v in MEMORY_FACTOR.items())

    args.memory_sizes = [gpus[device_id].memoryTotal for device_id in args.device_ids]
    args.memory_sizes[0] = args.memory_sizes[0] * (1 - 0.04 * (args.n_gpus-1))
    for i in range(1, args.n_gpus):
        args.memory_sizes[i] = args.memory_sizes[i] * 1.04
    if args.train_batch_size <= 0:
        args.train_batch_size = [int(memory_size * MEMORY_FACTOR[args.seq_train_type]) for memory_size in args.memory_sizes]
    if args.test_batch_size <= 0:
        args.test_batch_size = [int(memory_size * MEMORY_FACTOR[args.seq_train_type]) for memory_size in args.memory_sizes]

    special_tokens = {"ans_token":'__ans__', "pad_token":'__pad__', "unk_token":'__unk__', "eos_token": '<|endoftext|>'}
    if args.use_sep:
        special_tokens["sep_token"] = '__sep__'

    model_class, tokenizer_class, config_class = MODEL_CLASSES[args.model_name]
    tokenizer = tokenizer_class.from_pretrained('gpt2')
    print("TOKENIZER ORIGIN LEN: {}".format(len(tokenizer)))
    if not args.pretraining:
        tokenizer.add_tokens(list(special_tokens.values()))
    special_token_ids = {k:tokenizer.convert_tokens_to_ids(v) for k,v in special_tokens.items()}
    model_config = config_class.from_pretrained('./PModel')
    model_config.vocab_size = len(tokenizer)
    tokens_weight = torch.ones([model_config.vocab_size], dtype=torch.float).cuda()
    if not args.pretraining:
        tokens_weight[special_token_ids["ans_token"]] = args.tokens_weight
        if args.use_sep:
            tokens_weight[special_token_ids["sep_token"]] = args.tokens_weight

    args.max_len = model_config.n_positions - args.preseqlen

    data_attrs_path = os.path.join(BASE_DIR,"data_attrs.json")
    assert os.path.exists(data_attrs_path)
    with open(data_attrs_path, "r") as f:
        data_attrs = json.load(f)

    if args.seq_train_type in ["multitask", "multilm"]:
        args.n_train_epochs = {'_'.join(args.tasks): args.n_train_epochs}
    elif args.unbound:
        pass
    else:
        if "gem" in args.seq_train_type:
            args.memory_data = []
        if args.dynamic_epochs:
            data_sizes = {task: data_attrs[task]["train"]["data_size"] for task in args.tasks}
            max_total_data_size = max(data_sizes.values()) * args.n_train_epochs
            args.n_train_epochs = {d[0]: min(args.max_n_epochs, max_total_data_size//d[1]) for d in data_sizes.items()}
        else:
            args.n_train_epochs = {task: args.n_train_epochs for task in args.tasks}
    return args, model_config, model_class, tokenizer, config_class, special_token_ids, special_tokens, data_attrs, tokens_weight


class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
            last = self.last
        except AttributeError:
            last = record.relativeCreated

        delta = record.relativeCreated/1000 - last/1000
        record.relative = "{:.1f}".format(delta)
        record.uptime = str(datetime.timedelta(seconds=record.relativeCreated//1000))
        self.last = record.relativeCreated
        return True


def init_logging(filename):
    logging_format = "%(asctime)s - %(uptime)s - %(relative)ss - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(format=logging_format, filename=filename, filemode='a', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())

args, MODEL_CONFIG, MODEL_CLASS, TOKENIZER, CONFIG_CLASS, SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, DATA_ATTRS, TOKENS_WEIGHT = parse_args()


TASK_DICT = {
    "e2enlg2": {
               "train": os.path.join(args.data_dir,"e2enlg2_to_squad-train-v2.0.json"),
               "eval": os.path.join(args.data_dir,"e2enlg2_to_squad-test-v2.0.json"),
               "test": os.path.join(args.data_dir,"e2enlg2_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "e2enlg": {
               "train": os.path.join(args.data_dir,"e2enlg_to_squad-train-v2.0.json"),
               "eval": os.path.join(args.data_dir,"e2enlg_to_squad-test-v2.0.json"),
               "test": os.path.join(args.data_dir,"e2enlg_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "rnnlg.tv": {
               "train": os.path.join(args.data_dir,"rnnlg.tv_to_squad-train-v2.0.json"),
               "eval": os.path.join(args.data_dir,"rnnlg.tv_to_squad-test-v2.0.json"),
               "test": os.path.join(args.data_dir,"rnnlg.tv_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "rnnlg.hotel": {
               "train": os.path.join(args.data_dir,"rnnlg.hotel_to_squad-train-v2.0.json"),
               "eval": os.path.join(args.data_dir,"rnnlg.hotel_to_squad-test-v2.0.json"),
               "test": os.path.join(args.data_dir,"rnnlg.hotel_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "rnnlg.rest": {
               "train": os.path.join(args.data_dir,"rnnlg.rest_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"rnnlg.rest_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"rnnlg.rest_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "rnnlg.laptop": {
               "train":os.path.join(args.data_dir,"rnnlg.laptop_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"rnnlg.laptop_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"rnnlg.laptop_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "squad1": {
               "train":os.path.join(args.data_dir,"squad-train-v1.1.json"),
               "eval":os.path.join(args.data_dir,"squad-dev-v1.1.json"),
               "test":os.path.join(args.data_dir,"squad-dev-v1.1.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "squad2": {
               "train":os.path.join(args.data_dir,"squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"squad-dev-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "iwslt.en.de": {
               "train":os.path.join(args.data_dir,"iwslt.en.de_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"iwslt.en.de_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"iwslt.en.de_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "cnn_dailymail": {
               "train":os.path.join(args.data_dir,"cnn_dailymail_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"cnn_dailymail_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"cnn_dailymail_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "multinli.in.out": {
               "train":os.path.join(args.data_dir,"multinli.in.out_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"multinli.in.out_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"multinli.in.out_to_squad-dev-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "sst": {
               "train":os.path.join(args.data_dir,"sst_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"sst_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"sst_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "srl": {
               "train":os.path.join(args.data_dir,"srl_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"srl_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"srl_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "zre": {
               "train":os.path.join(args.data_dir,"zre_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"zre_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"zre_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "woz.en": {
               "train":os.path.join(args.data_dir,"woz.en_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"woz.en_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"woz.en_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "wikisql": {
               "train":os.path.join(args.data_dir,"wikisql_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"wikisql_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"wikisql_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "schema": {
               "train":os.path.join(args.data_dir,"schema_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"schema_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"schema_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "ag": {
               "train":os.path.join(args.data_dir,"ag_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"ag_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"ag_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "dbpedia": {
               "train":os.path.join(args.data_dir,"dbpedia_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"dbpedia_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"dbpedia_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "yahoo": {
               "train":os.path.join(args.data_dir,"yahoo_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"yahoo_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"yahoo_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "amazon": {
               "train":os.path.join(args.data_dir,"amazon_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"amazon_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"amazon_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "yelp": {
               "train":os.path.join(args.data_dir,"yelp_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"yelp_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"yelp_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
}
