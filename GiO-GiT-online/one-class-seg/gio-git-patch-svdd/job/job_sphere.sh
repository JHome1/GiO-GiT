cd ..
export CUDA_VISIBLE_DEVICES=0
obj=bottle
python main_train.py --obj=$obj --lr 1e-4 --epochs 1201 --log_epoch 10 --use_geo use_sphere --curvature_s 1.0
