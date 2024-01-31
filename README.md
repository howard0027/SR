## SR

The final project of Advanced Computer Vision.

## Environment Installation
(~3.8G)
```
python3 -m pip install -r requirements.txt 
```

## Datasets
Training Data: DIV2K (~4.8G)
Testing Data: Set5, Set14, B100, Urban100 (~2.1G)
```
bash get_datasets.sh
```

## Training
```
cd src
python3 train.py --config ./configs/x2.yml
```

## Note
The implementation of this project is based on the following codes:
https://github.com/clovaai/cutblur
https://github.com/xindongzhang/ELAN

