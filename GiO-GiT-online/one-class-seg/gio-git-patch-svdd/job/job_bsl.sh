cd ..
export CUDA_VISIBLE_DEVICES=0
obj=bottle
python main_train.py --obj=$obj --epochs 1201 --log_epoch 10
