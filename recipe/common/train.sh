# NOTE: the script is supposed to be used in recipes like:
#     . script.sh
# Please don't try to run the shell directly.

cmd_args="--config-dir $script_dir/conf/train \
    train=$train data=$data model=$model criterions=$criterions optim=$optim \
    train.out_dir=$model_dir train.model_name=$tag train.save_vocab_path=false \
    train.tensorboard_event_path=$tensorboard_dir train.test_log_dir=$in_domain_test_log_dir \
    data.feature_table_key=$feature_table_key data.data_dir=$feature_pack_dir model.vocab_path=$vocab_path"

marine-train $cmd_args
