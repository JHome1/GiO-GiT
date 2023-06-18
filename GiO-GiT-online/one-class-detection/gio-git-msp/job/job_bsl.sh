cd ..
export CUDA_VISIBLE_DEVICES=0
class=0
python trainer.py --REAL_CLASS=$class --USE_Baseline
