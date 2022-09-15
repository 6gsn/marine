# NOTE: the script is supposed to be used in recipes like:
#     . script.sh
# Please don't try to run the shell directly.

if [ -z ${exist_target_id_dir} ]; then
    cmd_args="-t $val_test_size"
else
    cmd_args="--target_id_dir $exist_target_id_dir"
fi

marine-pack-corpus $raw_corpus_dir $feature_file_dir $vocab_path $feature_pack_dir -s $accent_status_seq_level -f $feature_table_key $cmd_args
