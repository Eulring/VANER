
# VANER: Biomedical Named Entity Recognition by LLM



# Train:

Mix multiple datasets for Instruction-Tuning with DBR

CUDA_VISIBLE_DEVICES=0 python unllama_train_vaner.py unidev_kgmix2 128 mt vaner

Single datasets Instruction-Tuning with DBR

CUDA_VISIBLE_DEVICES=0 python unllama_train_vaner.py ncbi 128 mt vaner




# Eva:

Evaluation for multiple datasets model

CUDA_VISIBLE_DEVICES=0 python unllama_eval_vaner.py ncbi vaner_unidev_kgmix2/checkpoint-number

Evaluation for single dataset model

CUDA_VISIBLE_DEVICES=0 python unllama_eval_vaner.py ncbi vaner_ncbi/checkpoint-number
