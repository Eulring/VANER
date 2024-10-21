
# VANER: Biomedical Named Entity Recognition by LLM


Note: the vaner is based on LLama-2-7b-hf, you need to download this base model.

## Train:

Mix multiple datasets for Instruction-Tuning with DBR

CUDA_VISIBLE_DEVICES=0 python unllama_train_vaner.py unidev_kgmix2 128 mt vaner

Single datasets Instruction-Tuning with DBR

CUDA_VISIBLE_DEVICES=0 python unllama_train_vaner.py ncbi 128 mt vaner



If you lack computational resources for training, you can also download our pre-trained models. We provide two models:

[Download Links](https://drive.google.com/drive/folders/1D1oasrS9pJ3gxz38t1wWtSYt-UaDvynm?usp=drive_link)

__bjy_unidev_kgmmix2:__

This model has been fine-tuned with a mix of 8 datasets and has undergone DBR.

__bjy_unidev_kgmmix2_nodataname:__

This model has also been fine-tuned with a mix of 8 datasets, but does not specify the name of the dataset in the construction of the prompt, corresponding to the VANER_adapt method in the paper.




## Evaluation:

Change 'model_id' in unllama_eval_vaner.py as the path of LLama-2-7b-hf.

__Evaluation script__

CUDA_VISIBLE_DEVICES=0 python unllama_eval_vaner.py ncbi vaner_unidev_kgmix2_checkpoint-number
