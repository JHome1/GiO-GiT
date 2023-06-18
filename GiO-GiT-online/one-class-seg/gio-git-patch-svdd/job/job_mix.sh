cd ..
export CUDA_VISIBLE_DEVICES=0
obj=bottle
python main_train.py --obj=$obj --lr 1e-4 --epochs 1201 --log_epoch 10 --use_geo use_mix --curvature_s 1.0 --curvature_h 1.0
