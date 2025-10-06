# Movies2Features
## Environment

```bash
pip install -r requirements.txt
```
## Datasets
We performed our experiments on two datasets, MovieNet and MM-Douban. 
[MovieNet]: https://movienet.github.io
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
