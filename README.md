# PreSumm

**This code is forked and refactored from the EMNLP 2019 paper [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)**


Some codes are borrowed from ONMT(https://github.com/OpenNMT/OpenNMT-py

## Data Preparation For Tv2 and DaNewsroom
### Option 1: download the processed data

[Pre-processed data](https://drive.google.com/drive/folders/1VACiGxIYi5cfNqhC4c-zGR0s3_7fHe-8?usp=sharing) TV2

Put all `.pt` files into `bert_data`

### Option 2: process the data yourself

#### Step 1 Download txt files
Download and unzip the `.json` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for the Tv2 articles. Put all  `.json` files in one directory (e.g. `../raw_stories`)

####  Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
export CLASSPATH=/path/to/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-full-2017-06-09` directory. 

####  Step 3. Clean articles
We need to clean the data, removing HTML tags and what not. Change -botxo to True if preprocessing for DaBERT
```
python preprocess.py -mode format_tv2 -raw_path PATH_TO_JSON_FILES -save_path result_path -botxo False
```
Saves txt files to a Corpus folder and also created a mapping folder that contains ids of articles and which train/valid/test split they belong
####  Step 4. Sentence Splitting and Tokenization

```
python preprocess.py -mode tokenize -raw_path RAW_PATH -save_path TOKENIZED_PATH
```

* `RAW_PATH` is the directory containing new txt files just generated.


####  Step 5. Format to Simpler Json Files
 
```
python preprocess.py -mode format_to_lines -raw_path RAW_PATH -save_path JSON_PATH -n_cpus 1 -use_bert_basic_tokenizer false -map_path MAP_PATH
```

* `RAW_PATH` is the directory containing tokenized files (`../merged_stories_tokenized`), `JSON_PATH` is the target directory to save the generated json files (`../json_data/cnndm`), `MAP_PATH` is the directory for the mapping folder that contains ids of articles and which train/valid/test split they belong.

####  Step 6. Format to PyTorch Files
```
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH  -lower -n_cpus 1 -log_file ../logs/preprocess.log
```

* `JSON_PATH` is the directory containing json files (`../json_data`), `BERT_DATA_PATH` is the target directory to save the generated binary files (`../bert_data`)

## Model Training

**First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

### Extractive Setting

```
python train.py -task ext -mode train -bert_data_path BERT_DATA_PATH -ext_dropout 0.1 -model_path MODEL_PATH -lr 2e-3 -visible_gpus 0 -report_every 50 -save_checkpoint_steps 1000 -batch_size 2000 -train_steps 50000 -accum_count 8 -log_file ../logs/ext_bert_tv2 -use_interval true -warmup_steps 10000 -max_pos 512
```

### Abstractive Setting

#### mBertAbs
```
python train.py  -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 10 -train_steps 200000 -report_every 50 -accum_count 30 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3  -log_file ../logs/abs_mbert_tv2
```
#### mBertMix
```
python train.py  -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 10 -train_steps 200000 -report_every 50 -accum_count 30 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3 -log_file ../logs/mix_mbert_tv2  -load_from_extractive EXT_CKPT   
```
* `EXT_CKPT` is the saved `.pt` checkpoint of the extractive model.

#### DaBERTAbs
Use -bert_model PATH_TO_DaBERT_FOLDER if using DaBERT.

```
python train.py  -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.004 -lr_dec 0.1982 -save_checkpoint_steps 2000 -batch_size 10 -train_steps 200000 -report_every 50 -accum_count 89 -beta1 0.913 -beta2 0.981 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3  -log_file ../logs/abs_Dabert_tv2 -bert_model PATH_TO_DaBERT_FOLDER
```
#### DaBERTMix
```
python train.py  -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.004 -lr_dec 0.1982 -save_checkpoint_steps 2000 -batch_size 10 -train_steps 200000 -report_every 50 -accum_count 89 -beta1 0.913 -beta2 0.981 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3 -log_file ../logs/mix_bert_tv2  -load_from_extractive EXT_CKPT -bert_model PATH_TO_DaBERT_FOLDER 
```
* `EXT_CKPT` is the saved `.pt` checkpoint of the extractive model.


## Model Evaluation
### Tv2
Change -task to "ext" if evaluating extractive models
```
python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path BERT_DATA_PATH -log_file ../logs/val_abs_mbert_tv2 -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_tv2 
```
