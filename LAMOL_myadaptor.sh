A=e2enlg
B=rnnlg.rest
C=rnnlg.hotel
D=rnnlg.tv
E=rnnlg.laptop
F=woz.en
G=wikisql
H=cnn_dailymail
EXP=CL_GEN


######## NOTE for hyperparameters #########

# How to set epoch and lr for each task:
# Similar sequences:
# ABCDE: 9 1.75e-4

# Dissimilar sequences:
# ABCDE: 12 1.75e-4
# F 12 1.75-4
# G 12 3e-4
# H 9 1.75e-4

# search space for --mix_ini and --entropy_coe:
# {0.12, 0.15, 0.17} {0.01, 0.05} for similar sequences
# {0.03, 0.05, 0.07} {0.01, 0.05} for dissimilar sequences
# Generally speaking these two hyperparameters are not to sentitive to each sequence, you are encouraged to do an exstensive search
# Settings we use to report numbers: (might not be optimal, also could be differnt on difference devices)
# Seq1: 0.15, 0.01
# Seq2: 0.15, 0.01
# Seq3: 0.15, 0.01
# Seq4: 0.12, 0.05
# Seq5: 0.05, 0.01
# Seq6: 0.03, 0.01
# Seq7: 0.05, 0.01
# Seq8: 0.05, 0.01

# The authors also realized that the current setting for --whole_mix_step and --warm_mix_step might not be optimal
# You might want to tune this to understand more about the mixing coefficients

########     END     #########


# Example for our approch

# Similar sequences: for every sequence we try SEED=1 and 2
SEED=1
ID=1
bash train_myadaptor.sh --z_train_epochs 9 9 9 9 9 --z_train_lrs 1.75e-4 1.75e-4 1.75e-4 1.75e-4 1.75e-4 --fit_epoch 0 --random_replay_batch --clear_model --whole_mix_step 6 --warm_mix_step 3 --clear_model --mix_ini 0.15  --lamaml --entropy_coe 0.01 --fp32 --adam_epsilon 1e-6 --learning_rate 1.75e-4 --z_max_batch_size 4 --id $ID --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --tasks $A $B $C $D $E > log.$ID.train.NLG.$SEED 2>&1
sleep 30
bash test_myadaptor.sh --fp32 --id $ID --task_test 5 --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --tasks $A $B $C $D $E > log.$ID.test.NLG.$SEED 2>&1

# Dissimilar sequencess: we use --partial_transfer for all dissimilar sequences

SEED=2
ID=1498
bash train_myadaptor.sh --partial_transfer --z_train_epochs 12 12 12 12 12 --z_train_lrs 1.75e-4 3e-4 1.75e-4 1.75e-4 1.75e-4 --fit_epoch 0 --random_replay_batch --clear_model --whole_mix_step 6 --warm_mix_step 3 --clear_model --mix_ini 0.03  --lamaml --entropy_coe 0.01 --fp32 --adam_epsilon 1e-6 --learning_rate 1.75e-4 --z_max_batch_size 4 --id $ID --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --tasks $A $G $C $F $B > log.$ID.train.NLG.$SEED 2>&1
sleep 30
bash test_myadaptor.sh --fp32 --id $ID --task_test 5 --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --tasks $A $G $C $F $B > log.$ID.test.NLG.$SEED 2>&1

# For adapter+LAMOL
# Just set --mix_ini to a positive number, and set --whole_mix_step and --warm_mix_step to 0, then there is no decision stage but reuse all modules everytime (and replay) by default.

# For adapter+CL
# Just set --mix_ini to a negative number, and set --whole_mix_step and --warm_mix_step to 0, then there is no decision stage but add new modules to every layer everytime by default.

# For ablation study:
# Use args --pseudo_ablation, or set --mix_ini to 0, or set --entropy_coe to zero.



