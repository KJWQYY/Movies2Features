# Movies2Features
## Environment

```bash
pip install -r requirements.txt
```

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
