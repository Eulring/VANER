
train:

unikg

CUDA_VISIBLE_DEVICES=0 python unllama_train_vaner.py unidev_kgmix2 128 mt vaner

singlekg

CUDA_VISIBLE_DEVICES=0 python unllama_train_vaner.py ncbi 128 mt vaner




eva:

unikg

CUDA_VISIBLE_DEVICES=0 python unllama_eval_vaner.py ncbi vaner_unidev_kgmix2/checkpoint-1428820

singlekg

CUDA_VISIBLE_DEVICES=0 python unllama_eval_vaner.py ncbi vaner_ncbi/checkpoint-1428820

