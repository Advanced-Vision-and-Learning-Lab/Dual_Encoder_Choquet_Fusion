#!/usr/bin/env bash
set -e


python demo.py --dataset deepship --deepship_root datasets/DeepShip --deepship_split_source random --encoder_mode dual --dual_branch_loss --model_name_wave custom_weights/aves_hf_base_nonbio --model_name_tf MIT/ast-finetuned-audioset-10-10-0.4593 --freeze_fe --peft_method adapter --adapter_dim 16 --adapter_insertion mhsa --batch_size 64 --num_workers 8 --lr 5e-4 --max_epochs 1 --gpus 1 --runs 1 --no_augment
python demo.py --dataset deepship --deepship_root datasets/DeepShip --deepship_split_source random --encoder_mode dual --dual_branch_loss --model_name_wave custom_weights/aves_hf_base_nonbio --model_name_tf MIT/ast-finetuned-audioset-10-10-0.4593 --freeze_fe --peft_method ia3 --ia3_target kv_ffn --batch_size 64 --num_workers 8 --lr 5e-4 --max_epochs 1 --gpus 1 --runs 1 --no_augment
python demo.py --dataset deepship --deepship_root datasets/DeepShip --deepship_split_source random --encoder_mode dual --dual_branch_loss --model_name_wave custom_weights/aves_hf_base_nonbio --model_name_tf MIT/ast-finetuned-audioset-10-10-0.4593 --freeze_fe --peft_method hpt --hpt_shared --hpt_bins 4 --hpt_location mhsa --batch_size 64 --num_workers 8 --lr 5e-4 --max_epochs 1 --gpus 1 --runs 1 --no_augment

python demo.py --dataset shipsear --shipsear_root datasets/ShipsEar --shipsear_split_strategy random --encoder_mode single --model_name MIT/ast-finetuned-audioset-10-10-0.4593 --freeze_fe --peft_method none --batch_size 64 --num_workers 8 --lr 1e-5 --max_epochs 1 --gpus 1 --runs 1 --no_augment
python demo.py --dataset shipsear --shipsear_root datasets/ShipsEar --shipsear_split_strategy random --encoder_mode single --model_name MIT/ast-finetuned-audioset-10-10-0.4593 --freeze_fe --peft_method linear --batch_size 64 --num_workers 8 --lr 1e-3 --max_epochs 1 --gpus 1 --runs 1 --no_augment
python demo.py --dataset shipsear --shipsear_root datasets/ShipsEar --shipsear_split_strategy random --encoder_mode single --model_name MIT/ast-finetuned-audioset-10-10-0.4593 --freeze_fe --peft_method lora --lora_r 8 --lora_target query --batch_size 64 --num_workers 8 --lr 5e-4 --max_epochs 1 --gpus 1 --runs 1 --no_augment
