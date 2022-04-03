A=e2enlg
B=rnnlg.rest
C=rnnlg.hotel
D=rnnlg.tv
E=rnnlg.laptop
F=woz.en
G=wikisql
H=cnn_dailymail

EXP=CL_GEN

# Example for Finetune baseline

SEED=1
bash train.sh --seq_train_type finetune --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --tasks $A $G $C $F $B > 6ft-log.train.NLG.$SEED 2>&1
sleep 30
bash test.sh --task_test 5 --seq_train_type finetune --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --tasks $A $G $C $F $B > 6ft-log.test.NLG.$SEED 2>&1
sleep 30

SEED=2
bash train.sh --seq_train_type finetune --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --tasks $A $G $C $F $B > 6ft-log.train.NLG.$SEED 2>&1
sleep 30
bash test.sh --task_test 5 --seq_train_type finetune --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --tasks $A $G $C $F $B > 6ft-log.test.NLG.$SEED 2>&1


# Example for (online) EWC baseline, use '--reg_lambda' to tune reg cofficient, 1e6 is selected from {1e4, 1e5, 1e6, 1e7}

SEED=1
bash train.sh --n_train_epochs 9 --reg_lambda 1e6 --seq_train_type ewc --model_name gpt2 --add_task_tokens --seed $SEED --tasks $E $D $C $B $A > 2ewc-log.train.NLG.$SEED 2>&1
sleep 30
bash test.sh --task_test 5 --seq_train_type ewc --model_name gpt2 --add_task_tokens --seed $SEED --tasks $E $D $C $B $A > 2ewc-log.test.ewc.NLG.$SEED 2>&1
sleep 30

SEED=2
bash train.sh --n_train_epochs 9 --reg_lambda 1e6 --seq_train_type ewc --model_name gpt2 --add_task_tokens --seed $SEED --tasks $E $D $C $B $A > 2ewc-log.train.ewc.NLG.$SEED 2>&1
sleep 30
bash test.sh --task_test 5 --seq_train_type ewc --model_name gpt2 --add_task_tokens --seed $SEED --tasks $E $D $C $B $A > 2ewc-log.test.ewc.NLG.$SEED 2>&1


# Example for LAMOL baseline, use '--lamaml' to increase replay frequency

SEED=1
bash train.sh --lamaml --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --tasks $E $D $C $B $A > 2lamol-log.train.lamol.NLG.$SEED 2>&1
sleep 30
bash test.sh --task_test 5 --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --tasks $E $D $C $B $A > 2lamol-log.test.lamol.NLG.$SEED 2>&1
sleep 30

SEED=2
bash train.sh --lamaml --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --tasks $E $D $C $B $A > 2lamol-log.train.lamol.NLG.$SEED 2>&1
sleep 30
bash test.sh --task_test 5 --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --tasks $E $D $C $B $A > 2lamol-log.test.lamol.NLG.$SEED 2>&1