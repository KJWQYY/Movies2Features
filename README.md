# Movies2Features
## Environment

```bash
pip install -r requirements.txt
```
## Datasets
We performed our experiments on two datasets, MovieNet and MM-Douban. 
### MovieNet
MovieNet: https://movienet.github.io
### MM-Douban
MM-Douban: https://pan.baidu.com/s/1rFMebQoZgsFSqEd9lwToqg Code:zydh
## Data processing
MovieNet as an example:
### MovieNet
```bash
/data/MovieNet/prepare_data.sh
```
```bash
/data/MovieNet/features/prepare_features.sh
```

## Train and Evaluate
```bash
  python stage_one.py && python stage_two.py
```
