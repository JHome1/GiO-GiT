cd ..
export CUDA_VISIBLE_DEVICES=0
class=0
python trainer.py --REAL_CLASS=$class --USE_Hyper --CURVATURE_Hyper 0.005
