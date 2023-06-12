cd ..
# HiT
python eval_ood.py --cfg ./config/streethazards-resnet50-upernet-hyper.yaml DATASET.list_val ./data/streethazards/test.odgt

# SiT
python eval_ood.py --cfg ./config/streethazards-resnet50-upernet-sphere.yaml DATASET.list_val ./data/streethazards/test.odgt

# MiT
python eval_ood.py --cfg ./config/streethazards-resnet50-upernet-mix.yaml DATASET.list_val ./data/streethazards/test.odgt
