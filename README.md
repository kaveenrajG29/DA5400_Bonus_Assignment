## Setup and Execution

### 1. Create the `data` Folder
Inside your project directory, create a folder named `data` and store all dataset files there.

```bash
mkdir data
```

### 2. Install Requirements
Install all required dependencies to run the pipeline.



### 3. Run a Small Verification Pipeline
Runs a small and fast version to verify that the pipeline works correctly.

```bash
python3 condtsc_pipeline.py --data-dir data --epochs 2 --spc 2 --max-train-per-class 200 --eval-epochs 2 --batch-size 32
```

### 4. Recommended Initial Run
Start with this configuration before scaling up experiments.

```bash
python condtsc_pipeline.py --data-dir data --epochs 200 --spc 10 --max-train-per-class 2000 --eval-epochs 100 --batch-size 128 --run-full-baseline
```

### 5. Heavier Paper-like Run
For a more extensive experiment similar to paper-level settings:

- Increase `--epochs`
- Increase `--m-real`
- Remove or increase `--max-train-per-class`

Example:

```bash
python condtsc_pipeline.py --data-dir data --epochs 500 --spc 20 --eval-epochs 200 --batch-size 128 --run-full-baseline
```

### 6. Outputs
All generated outputs are written to the `outputs/` directory as `.npz` file.


