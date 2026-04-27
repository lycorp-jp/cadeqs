# Baselines

This directory contains baseline implementations used in our experiments: **FT-T5**, **FT-LLM**, and **MONR**.  
Note that BM25 and SPLADE are training-free methods and can be reproduced with [bm25s](https://github.com/xhluca/bm25s) and [pyserini](https://github.com/castorini/pyserini), respectively.  
BGE-M3 can be reproduced using [cadeqs/infer.py](../cadeqs/infer.py) by setting the encoder path to a BGE-M3 checkpoint.  
HRED can be reproduced using the authors' [public implementation](https://github.com/sordonia/hred-qs).

# Usage

Below are minimal train / infer command examples.


## FT-T5

```bash
# train
python -m baselines.t5.train \
  --model_name_or_path google/t5-efficient-tiny \
  --train_file /tmp/qs_aol/train.jsonl \
  --output_dir /tmp/qs_aol/t5_model

# infer
python -m baselines.t5.infer \
  --model_path /tmp/qs_aol/t5_model \
  --test_file /tmp/qs_aol/test.jsonl \
  --output_file /tmp/qs_aol/t5_predictions.jsonl
```

## FT-LLM

```bash
# train
python -m baselines.llm.train \
  --base_model meta-llama/Llama-3.2-1B \
  --train_file /tmp/qs_aol/train.jsonl \
  --output_dir /tmp/qs_aol/llm_model

# infer (requires vLLM)
python -m baselines.llm.infer \
  --model_path /tmp/qs_aol/llm_model \
  --test_file /tmp/qs_aol/test.jsonl \
  --output_file /tmp/qs_aol/llm_predictions.jsonl
```


## MONR

```bash
# train
python -m baselines.monr.train \
  --model_name_or_path MiniLMv2-L6-H768-distilled-from-BERT-Base \
  --train_data /tmp/qs_aol/train.jsonl \
  --output_dir /tmp/qs_aol/monr_model

# infer
python -m baselines.monr.infer \
  --model_path /tmp/qs_aol/monr_model \
  --corpus_path /tmp/qs_aol/inventory.jsonl \
  --test_file /tmp/qs_aol/test.jsonl \
  --output_file /tmp/qs_aol/monr_predictions.jsonl \
  --index_path /tmp/qs_aol/monr_faiss.index
```
