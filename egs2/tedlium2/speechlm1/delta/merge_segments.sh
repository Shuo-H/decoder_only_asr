#!/bin/bash

. /projects/bcey/shan1/espnet/tools/activate_python.sh


cd /projects/bcey/shan1/espnet/egs2/tedlium2/speechlm1

python pyscripts/utils/speechlm_concat_examples.py --input_data_json dump/raw_ssl_asr_tedlium2/train_org/data.json --output_dir dump/raw_ssl_asr_tedlium2/train_sp --max_len 120
echo "train"
python pyscripts/utils/speechlm_concat_examples.py --input_data_json dump/raw_ssl_asr_tedlium2/dev_org/data.json   --output_dir dump/raw_ssl_asr_tedlium2/dev      --max_len 120
echo "val"
# python pyscripts/utils/speechlm_concat_examples_segment.py   --input_data_json dump/raw_ssl_asr_tedlium2/dev_org/data.json     --output_dir dump/raw_ssl_asr_tedlium2/test
mkdir -p dump/raw_ssl_asr_tedlium2/test
cp -r dump/raw_ssl_asr_tedlium2/dev dump/raw_ssl_asr_tedlium2/test
echo "test"