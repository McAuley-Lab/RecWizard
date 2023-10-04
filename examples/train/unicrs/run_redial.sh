# ReDIAL
CUDA_VISIBLE_DEVICES=0 python examples/train/unicrs/pretrain.py --num_train_epochs 5 --num_workers 2 --learning_rate 5e-4
CUDA_VISIBLE_DEVICES=0 python examples/train/unicrs/train_conv.py  --run_infer --num_train_epochs 10 --num_workers 2
CUDA_VISIBLE_DEVICES=0 python examples/train/unicrs/train_rec.py --num_workers 2
