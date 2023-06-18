cd ..
export CUDA_VISIBLE_DEVICES=0
class=0
python trainer.py --REAL_CLASS=$class --USE_Mix --CURVATURE_Sphere 1.0 --CURVATURE_Hyper 0.005
