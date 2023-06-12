cd ..
# HiT
python train.py --cfg ./config/streethazards-resnet50-upernet-hyper.yaml

# SiT
python train.py --cfg ./config/streethazards-resnet50-upernet-sphere.yaml

# MiT
python train.py --cfg ./config/streethazards-resnet50-upernet-mix.yaml
