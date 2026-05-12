# CondTSC / DCDDM Time-Series Condensation

This project creates a small synthetic time-series dataset from the original
data, then checks how well models trained on that synthetic data perform.

Put your `.npz` files in:

```text
data/
```

Each `.npz` file is treated as one class. If the raw data is complex-valued, the
default setting uses magnitude values for training and synthetic generation.

## 1. Setup

Run this once before training:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This creates a Python environment and installs the packages needed for PyTorch,
data loading, training, plotting, and evaluation.

## 2. Create Synthetic Data

Run:

```bash
bash run_dcdmm_pipeline.sh
```

This starts the full condensation pipeline. First, it trains teacher/expert
models on the original time-series data and stores their learning behavior in
replay buffers. Then it uses those buffers to learn a much smaller synthetic
dataset that tries to represent the important information from the original
data.

The main synthetic dataset is saved here:

```text
outputs_dcdmm/condensed_dataset.npz
```

Inside that file:

```text
x = synthetic time-series samples
y = class labels
class_names = class names
```

## 3. Compare Original vs Synthetic Accuracy

After the synthetic dataset is created, run:

```bash
python3 compare_original_synthetic.py \
  --output-dir outputs_dcdmm \
  --epoch_eval_train 50 \
  --num_eval 3 \
  --device cuda
```

This trains models in two ways. One model is trained on the full original
training data. Another fresh model is trained only on the synthetic data. Both
are tested on the same real test set, so the comparison is fair and easy to
read.

By default, it evaluates:

```text
MLP, CNNBN, CNNIN, TCN
```

The results are saved here:

```text
outputs_dcdmm/accuracy_table.xlsx
outputs_dcdmm/accuracy_table.csv
outputs_dcdmm/accuracy_comparison.json
outputs_dcdmm/confusion matrix/
outputs_dcdmm/amplitude/
```

Use `accuracy_table.xlsx` for the final accuracy table. The `confusion matrix`
folder contains confusion-matrix images for every model/run, and the `amplitude`
folder contains waveform comparisons between original and synthetic samples.
