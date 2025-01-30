#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sp
valid_set=dev
test_sets="test"

train_config=conf/train_delay_asr_small.yaml
inference_config=conf/decode_asr.yaml
ssl_opts="--ssl_choice s3prl --ssl_feature_type wavlm_large --ssl_nlayer 21 --ssl_kmeans_path exp/kmeans_wavlm/wavlm_large_21_2000clusters/km_2000.mdl --ssl_batch_bins 19200000"
bpe_opts="--subword_choice sentencepiece --nbpe 5000"

./speechlm.sh \
    --task "ssl_asr" \
    --data_name tedlium2 \
    --fs 16000 \
    --ngpu 1 \
    --nj 16 \
    --inference_nj 16 \
    --nbest 1 \
    --gpu_inference false \
    --token_list_dir data/token_list/ssl_asr_tedlium2 \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --audio_format "flac.ark" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 100.0 \
    --max_wav_duration 120.0 \
    ${ssl_opts} ${bpe_opts} \
    "$@"

# min_max_wav_duration is only useful for stage 2