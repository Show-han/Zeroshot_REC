# Zero-shot Referring Expression Comprehension via Structural Similarity Between Images and Captions
This repository contains the code for the paper [Zero-shot Referring Expression Comprehension via Structural Similarity Between Images and Captions](https://arxiv.org/pdf/2311.17048.pdf)
(CVPR 2024).

## Setup
1. Create a new conda environment:
```
conda create --name vg python=3.8
conda activate vg
```
2. Install pytorch following the [official website](https://pytorch.org/get-started/locally/). We have successfully tested our code on pytorch 2.2.1 (`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`), but lower version should also be feasible.

3. Install pip requirements via:
```
pip install -r requirements.txt
```

## Data Download
### RefCOCO/g/+
We use the same pre-processed data for RefCOCO/g/+ as [ReCLIP](https://github.com/allenai/reclip) for a fair comparison.
Download pre-processed data files via `gsutil cp gs://reclip-sanjays/reclip_data.tar.gz`, and extract the data using `tar -xvzf reclip_data.tar.gz`. This data does not include images, therefore also download the images for RefCOCO/g/+ from [http://images.cocodataset.org/zips/train2014.zip](http://images.cocodataset.org/zips/train2014.zip). 

### Caption Triplets
We release the triplets files in [Huggingface](https://huggingface.co/datasets/CresCat01/RefCOCO-Triplets)

### LoRA Checkpoints
They are also in [Huggingface](https://huggingface.co/datasets/CresCat01/RefCOCO-Triplets).

### VLA Fine-tuning Data
For CLIP/FLAVA fine-tuning, you need to download the following datasets:
1. [HICO-det](https://websites.umich.edu/~ywchao/hico/)
2. [SWiG](https://github.com/allenai/swig?tab=readme-ov-file)
3. [VG](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)

Please follow the instructions provided on each website to download the raw data.

After downloading the datasets, modify the configuration file `VLA_finetune/training/data_path.ini` according to the root directory of your downloaded data.

For the `triplets_file` required for SWiG, it can be found in [Huggingface](https://huggingface.co/datasets/CresCat01/RefCOCO-Triplets) (`SWiG_triplets.json`)


## Results with CLIP/FLAVA on RefCOCO/g/+
The following format can be used to run experiments:
```
python eval_refcoco/main.py --input_file [INPUT_FILE] --image_root [IMAGE_ROOT] --method [parse/baseline/random/matching] --clip_model ViT-B/32 --triplets_file [TRIPLETS_FILE] --detector_file [DETECTION_FILE] {--rule_filter} {--enable_lora} {--lora_path [LORA_PATH]}
```

(`/` is used above to denote different options for a given argument. Content enclosed in brackets `[]` should be replaced with the actual content, while braces `{}` denote optional arguments.)

`--input_file`: should be in `.jsonl` format, e.g., reclip_preprocess/refcocog_val.jsonl (we provide these files for the datasets same as ReCLIP; see the Data Download information above).

`--image_root`: the top-level directory containing all images in the dataset, e.g., COCO2014/train2014/

`--detector_file`: if not specified, ground-truth proposals are used. The detection files are in `reclip_data.tar.gz` and have the format `{refcoco/refcocog/refcoco+}_dets_dict.json`, e.g., reclip_preprocess/refcocog_dets_dict.json 

`--triplets_file`: caption triplets generated using ChatGPT, e.g., triplets/gpt_refcocog_val.jsonl

`--rule_filter`: Whether to filter out redundant triplets based on heuristic rules. Should be activated on default.

`--enable_lora`: Whether to load pre-trained LoRA modules for CLIP and FLAVA. 

`--lora_path`: If enable_lora, you should specify the path to the pre-trained LoRA module, e.g., pre_trained/epoch_latest.pt

Choices for `method`: "matching" stands for our proposed triplet-to-instance visual grounding pipeline. "parse", "baseline" are two grounding methods used in the baseline paper [ReCLIP](https://github.com/allenai/reclip). "random" selects one of the proposals uniformly at random.

Choices for `clip_model`: We only use ViT-B/32 for entire experiments. 

To see explanations of other arguments see the `eval_refcoco/main.py` file.

## How to Generate Caption Triplets
1. Install [Gentopia](https://github.com/Gentopia-AI/Gentopia) by running the following command:
```
pip install gentopia
```
2. Modify the `ROOT_PATH` and `DATA_BASE_PATH` variables in the `triplets_chatgpt/scripts/run.sh` file. Ensure that the paths point to the correct directories. (For `reclip_preprocess`, see the Data Download information above).

3. Make sure to add your OpenAI key in the `triplets_chatgpt/.env` file for authentication.

4. After making the necessary modifications, run the bash file `run.sh` to generate the caption triplets.


## How to Fine-tune CLIP/FLAVA
Make sure the data is ready for VLA fine-tuning. 

The following format can be used to fine-tune CLIP:
```
torchrun --nproc_per_node 8 --master_port 23450 -m VLA_finetune.training.main --name CLIP_finetune --lora 4 --pretrained openai --epochs 20 --warmup 150 --workers 48 --lr 0.000005 --save-frequency 5 --batch-size 128 --model ViT-B/32
```

The following format can be used to fine-tune FLAVA:
```
torchrun --nproc_per_node 8 --master_port 23452 -m VLA_finetune.training.main --name FLAVA_finetune --lora 16 --epochs 20 --warmup 150 --workers 48 --lr 0.000005 --save-frequency 5 --batch-size 128 --flava
```

## TODO
- Upload code for Who's Waldo dataset

## Acknowledgements
The code in the `eval_refcoco` directory is adapted from the baseline method [ReCLIP](https://github.com/allenai/reclip) to ensure a fair comparison. We have removed code pertaining to other comparison methods originally present in [ReCLIP repo](https://github.com/allenai/reclip) to enhance readability. The code in the `VLA_finetune` directory is adapted from code in [TSVLC](https://github.com/SivanDoveh/TSVLC).


## Citation
If you find this repository useful, please cite our paper:
```
@article{han2023zero,
  title={Zero-shot Referring Expression Comprehension via Structural Similarity Between Images and Captions},
  author={Han, Zeyu and Zhu, Fangrui and Lao, Qianru and Jiang, Huaizu},
  journal={arXiv preprint arXiv:2311.17048},
  year={2023}
}
```
