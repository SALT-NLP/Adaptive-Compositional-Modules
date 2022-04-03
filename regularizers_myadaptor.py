import abc
import math
import torch
from torch.optim import Optimizer, SGD
from settings_myadaptor import args, FILL_VAL, TOKENS_WEIGHT
from utils_myadaptor import get_losses, get_model_dir
from parallel import DataParallelCriterion
from torch.nn import CrossEntropyLoss, MSELoss
import pickle as pkl
import os
import logging
logger = logging.getLogger(__name__)


class Regularizer(abc.ABC):
    def __init__(self, model, dataloaders, task, prev_task=None, task_id=1):
        self.model = model
        self.dataloaders = dataloaders
        self.task = task
        self.prev_task = prev_task
        self.task_id = task_id

    @abc.abstractmethod
    def task_start_do(self):
        return NotImplemented

    @abc.abstractmethod
    def task_end_do(self):
        return NotImplemented

    def save_reg_params(self):
        save_params = []
        for param in self.model.parameters():
            if param.requires_grad:
                save_param = self.model.reg_params.get(param)
                save_params.append(save_param)

        model_dir = get_model_dir([self.task])
        reg_params_path = os.path.join(model_dir, "reg_params.pkl")
        with open(reg_params_path, 'wb') as f:
            pkl.dump(save_params, f)

    def load_reg_params(self):
        if self.prev_task:
            model_dir = get_model_dir([self.prev_task])
            reg_params_path = os.path.join(model_dir, "reg_params.pkl")
            with open(reg_params_path, 'rb') as f:
                # self.model.reg_params = pkl.load(f)
                save_params = pkl.load(f)
            cnt = 0
            for param in self.model.parameters():
                if param.requires_grad:
                    del self.model.reg_params[param]
                    self.model.reg_params[param] = save_params[cnt]
                    cnt += 1
                    # logger.info(self.model.reg_params[param])
                    # exit(0)

            # logger.info(self.model.reg_params)
            # input()


class MAS(Regularizer):
    def task_start_do(self):
        # self.load_reg_params()
        task_start_do(self.model)

    def task_end_do(self):
        updater = Omega_update(self.model.parameters(), lr=0.0001, momentum=0.9)
        compute_importance(self.model, updater, self.dataloaders)
        accumulate_reg_params(self.model, self.task_id)
        self.save_reg_params()


class EWC(Regularizer):
    def task_start_do(self):
        # self.load_reg_params()
        logger.info("Using EWC Regularizer!!!")
        task_start_do(self.model)

    def task_end_do(self):
        updater = Omega_update(self.model.parameters(), lr=0.0001, momentum=0.9)
        compute_importance(self.model, updater, self.dataloaders, loss_type="ewc")
        accumulate_reg_params(self.model, self.task_id)
        self.save_reg_params()
        logger.info("Reg para after training...")
        for param in self.model.parameters():
            if param.requires_grad:
                logger.info(param)
                logger.info(self.model.reg_params.get(param))
                break


REG_TYPES = {
    "mas": MAS,
    "ewc": EWC,
    "llewc": EWC,
}
args.REG_TYPE_KEYS = REG_TYPE_KEYS = list(REG_TYPES.keys())


def task_start_do(model):
    if not hasattr(model, "reg_params"):
        initialize_reg_params(model)
    else:
        clean_omega_sum(model)


def initialize_reg_params(model):
    """initialize an omega for each parameter to zero"""
    reg_params = {}
    for name, param in model.named_parameters():
        # if param.requires_grad and not any(nd in name for nd in NO_DECAY):
        if param.requires_grad:
            # print('initializing param',name)
            omega = torch.FloatTensor(param.size()).zero_()
            omega = omega.cuda()
            init_val = param.data.clone().detach()
            init_val = init_val.cuda()
            reg_param = {}
            reg_param['omega'] = omega
            reg_param['omega_sum'] = omega
            # initialize the initial value
            reg_param['init_val'] = init_val
            reg_params[param] = reg_param
    if 'data_count' not in reg_params:
        reg_params['data_count'] = 0
    reg_params['lambda'] = args.reg_lambda
    model.reg_params = reg_params
    logger.info("Reg para after INI...")
    for param in model.parameters():
        if param.requires_grad:
            logger.info(param)
            logger.info(model.reg_params.get(param))
            break
    # logger.info(model.reg_params)


def clean_omega_sum(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            omega = torch.FloatTensor(param.size()).zero_()
            omega = omega.cuda()

            # added by ZYZ
            init_val = param.data.clone().detach()
            init_val = init_val.cuda()

            reg_param = model.reg_params.get(param)
            # reset sum to 0
            reg_param['omega_sum'] = omega

            # added by ZYZ
            # renew the init_val
            del reg_param['init_val']
            reg_param['init_val'] = init_val
            model.reg_params[param] = reg_param

    model.reg_params['data_count'] = 0
    logger.info("Reg para after INI...")
    for param in model.parameters():
        if param.requires_grad:
            logger.info(param)
            logger.info(model.reg_params.get(param))
            break


class Weight_Regularized_AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias)
        super(Weight_Regularized_AdamW, self).__init__(params, defaults)
        """
        logger.info(self.param_groups)
        for group in self.param_groups:
            for p in group['params']:
                logger.info(p)
                logger.info(p.dtype)
                break
            break
        """

    def step(self, reg_params, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        reg_lambda = reg_params.get('lambda')
        # logger.info("One step in Reg Opt......")

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                
                if p in reg_params:

                    reg_param = reg_params.get(p)
                    # get omega for this parameter
                    omega = reg_param.get('omega')
                    # initial value when the training start
                    init_val = reg_param.get('init_val')
                    curr_weight_val = p.data

                    # get the difference
                    weight_dif = curr_weight_val.add(-1, init_val)
                    # compute the reg penalty
                    regulizer = weight_dif.mul(2*reg_lambda*omega)

                    del weight_dif
                    del curr_weight_val
                    del omega
                    del init_val
                    # add the reg regulizer to the gradient
                    grad.add_(regulizer)
                    # grad.add_(1, regulizer)
                    del regulizer

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                
                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                # Regularize PART CODE GOES HERE
                
                # This implementation of Regularization seems to have nan issues
                """
                if p in reg_params:

                    reg_param = reg_params.get(p)
                    # get omega for this parameter
                    omega = reg_param.get('omega')
                    # initial value when the training start
                    init_val = reg_param.get('init_val')
                    curr_weight_val = p.data

                    # get the difference
                    weight_dif = curr_weight_val.add(-1, init_val)
                    # compute the reg penalty
                    regulizer = weight_dif.mul(2*reg_lambda*omega)

                    del weight_dif
                    del curr_weight_val
                    del omega
                    del init_val
                    # add the reg regulizer to the gradient
                    # grad.add_(regulizer)
                    p.data.add_(-group['lr'], regulizer)
                    del regulizer
                # Regularize PART CODE ENDS
                """
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss


# update omega for one task; use in compute_importance
class Omega_update(SGD):
    """
    Update the paramerter importance using the gradient of the function output norm. To be used at deployment time.
    reg_params:parameters omega to be updated
    batch_index,batch_size:used to keep a running average over the seen samples
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):

        super(Omega_update, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super(Omega_update, self).__setstate__(state)

    def step(self, reg_params, batch_size, closure=None):
        """
        Performs a single parameters importance update setp
        """
        # print('************************DOING A STEP************************')
        reg_params['data_count'] += batch_size
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # if the parameter has an omega to be updated
            for p in group['params']:

                # print('************************ONE PARAM************************')

                if p.grad is None:
                    continue

                if p in reg_params:

                    # logger.info(p.grad)

                    # HERE MAS IMPOERANCE UPDATE GOES
                    # get the gradient
                    unreg_dp = p.grad.data.clone().type(torch.float)
                    reg_param = reg_params.get(p)
                    # get parameter omega
                    omega = reg_param.get('omega_sum')
                    if args.seq_train_type == "ewc" or args.seq_train_type == "llewc":
                        omega = omega.add((unreg_dp)**2)
                    else:
                        omega = omega.add(unreg_dp.abs_())
                    reg_param['omega_sum'] = omega
                    reg_params[p] = reg_param
                    # HERE MAS IMPOERANCE UPDATE ENDS
                    # exit(0)

        return loss # HAS NOTHING TO DO


# update omega for one task
def compute_importance(model, updater, dataloaders, loss_type="l2"):
    """Mimic the depoloyment setup where the model is applied on some samples and those are used to update the importance params
       Uses the L2norm of the function output. This is what we MAS uses as default
    """
    # model.eval()  # Set model to training mode so we get the gradient
    # train_loss_fct = DataParallelCriterion(CrossEntropyLoss(ignore_index=FILL_VAL), args.device_ids)
    global TOKENS_WEIGHT
    # softmax = torch.nn.Softmax(dim=-1)
    if loss_type == "l2":
        loss_fct = DataParallelCriterion(torch.nn.MSELoss(reduction='mean'), args.device_ids)
    elif loss_type == "l1":
        loss_fct = DataParallelCriterion(torch.nn.L1Loss(reduction='mean'), args.device_ids)
    elif loss_type == "ewc":
        # """
        while 50265 != TOKENS_WEIGHT.shape[0]:
            TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
            logger.info("Add one dim weight!")
        # """
        CELoss = CrossEntropyLoss(ignore_index=FILL_VAL, reduction='mean', weight=TOKENS_WEIGHT.type(torch.float if args.fp32 else torch.half))
        # loss_fct = DataParallelCriterion(CELoss, args.device_ids)
        loss_fct = CELoss

    # Iterate over data.
    for dataloader in dataloaders:
        # for cq, len_cq, cqa, len_cqa, Y, _, _ in dataloader:
        for (_, _, cqa, _, Y, gen_X, gen_Y, is_extra, idx) in dataloader:
            # get the inputs
            # n_inputs = sum(len(_cq) for _cq in cq)

            n_inputs = cqa[0].shape[0]
            assert n_inputs == 1

            # zero the parameter gradients
            updater.zero_grad()

            # forward
            if loss_type != "ewc":
                raise NotImplementedError("Only support EWC now!")
                """
                logits = model(cq)
                logits = [logit[range(len(logit)), len_cq[i]-1, :] for i, logit in enumerate(logits)]
                # logits = [softmax(logit, dim=-1) for logit in logits]
                target_zeros = [torch.zeros(logit.size()).to(args.device_ids[i]) for i, logit in enumerate(logits)]
                logits = [softmax(logit) for logit in logits]

                if loss_type == "l2":
                    targets = loss_fct(logits, target_zeros)
                elif loss_type == "l1":
                    targets = loss_fct(logits, target_zeros)
                """
            else:
                losses = get_losses(model, cqa[0].cuda(), Y[0].cuda(), gen_X[0].cuda(), gen_Y[0].cuda(), loss_fct)
                if losses[1].item() == 0:
                    loss = losses[0]
                    logger.info("Are you sure to use QA loss only while cal important weight?")
                    exit(0)
                else:
                    loss = sum(losses)

            # targets /= n_inputs

            # compute the gradients
            # targets.backward()
            loss.backward()

            # update the parameters importance
            updater.step(model.reg_params, n_inputs)


# omega of task1 + omega of task2 ...
# new_omega=omega_sum/data_count; omega=new_omega+prev_omega
def accumulate_reg_params(model, task_id):
    """accumelate the newly computed omega with the previously stroed one from the old previous tasks"""

    # Add importance coefficient: online EWC
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param in model.reg_params:
                reg_param = model.reg_params.get(param)
                prev_omega = reg_param.get('omega')
                new_omega = reg_param.get('omega_sum') / model.reg_params["data_count"]

                acc_omega = torch.add(prev_omega, new_omega)

                del reg_param['omega_sum']
                reg_param['omega'] = acc_omega

                model.reg_params[param] = reg_param
                del prev_omega
                del new_omega
                del acc_omega
        else:
            if param in model.reg_params:
                reg_param = model.reg_params.get(param)
                # print('removing unused omega',name)
                del reg_param['omega']
                del model.reg_params[param]


class Weight_Regularized_SGD(SGD):
    r"""Implements SGD training with importance params regulization. IT inherents stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from

    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(Weight_Regularized_SGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super(Weight_Regularized_SGD, self).__setstate__(state)

    def step(self, reg_params, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
        reg_lambda = reg_params.get('lambda')

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # MAS PART CODE GOES HERE
                # if this param has an omega to use for regulization
                if p in reg_params:

                    reg_param = reg_params.get(p)
                    # get omega for this parameter
                    omega = reg_param.get('omega')
                    # initial value when the training start
                    init_val = reg_param.get('init_val')

                    curr_wegiht_val = p.data
                    # move the tensors to cuda
                    init_val = init_val.cuda()
                    omega = omega.cuda()

                    # get the difference
                    weight_dif = curr_wegiht_val.add(-1, init_val)
                    # compute the MAS penalty
                    regulizer = weight_dif.mul(2*reg_lambda*omega)
                    del weight_dif
                    del curr_wegiht_val
                    del omega
                    del init_val
                    # add the MAS regulizer to the gradient
                    d_p.add_(regulizer)
                    del regulizer
                # MAS PARAT CODE ENDS
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data.sign())

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)

        return loss
