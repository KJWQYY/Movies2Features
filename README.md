# Movies2Features
## Environment

```bash
pip install -r requirements.txt
```
## Datasets
We performed our experiments on two datasets, MovieNet(https://movienet.github.io/) and MM-Douban. 
## Data processing

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
