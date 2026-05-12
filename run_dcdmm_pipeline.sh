#!/usr/bin/env bash
set -euo pipefail

# End-to-end DCDDM runner for condtsc_pipeline.py.
# Stable default run for the local complex .npz time-series data.
# Override any variable from the shell, e.g.:
#   ITERATION=1000 NUM_EXPERTS=20 DEVICE=cuda ./run_dcdmm_pipeline.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt

DATASET="${DATASET:-local_npz}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_dcdmm}"
BUFFER_PATH="${BUFFER_PATH:-buffers}"
CONFIG_FILE="${CONFIG_FILE:-TimeSeriesCond-master/config.yml}"
DEVICE="${DEVICE:-cuda}"
MODEL="${MODEL:-CNNIN}"
IPC="${IPC:-5}"
INPUTAUG="${INPUTAUG:-raw}"
AUG="${AUG:-None}"
PIX_INIT="${PIX_INIT:-real}"

NUM_EXPERTS="${NUM_EXPERTS:-5}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
BATCH_TRAIN="${BATCH_TRAIN:-256}"
LR_TEACHER_BUFFER="${LR_TEACHER_BUFFER:-0.0001}"

ITERATION="${ITERATION:-200}"
EVAL_IT="${EVAL_IT:-50}"
NUM_EVAL="${NUM_EVAL:-2}"
EPOCH_EVAL_TRAIN="${EPOCH_EVAL_TRAIN:-50}"
LR_TEACHER="${LR_TEACHER:-0.0001}"
LR_FEAT="${LR_FEAT:-0.01}"
LR_LR="${LR_LR:-0.00000001}"
LAMBDA_DM="${LAMBDA_DM:-1.0}"
MAX_START_EPOCH="${MAX_START_EPOCH:-10}"
EXPERT_EPOCHS="${EXPERT_EPOCHS:-10}"
SYN_STEPS="${SYN_STEPS:-10}"

COMMON_ARGS=(
  --config_filename "${CONFIG_FILE}"
  --dataset "${DATASET}"
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --buffer_path "${BUFFER_PATH}"
  --framework DCDDM
  --model "${MODEL}"
  --batch_train "${BATCH_TRAIN}"
  --inputaug "${INPUTAUG}"
  --aug "${AUG}"
  --device "${DEVICE}"
)

"${VENV_DIR}/bin/python" condtsc_pipeline.py \
  --mode buffer \
  "${COMMON_ARGS[@]}" \
  --num_experts "${NUM_EXPERTS}" \
  --train_epochs "${TRAIN_EPOCHS}" \
  --save_interval "${SAVE_INTERVAL}" \
  --lr_teacher "${LR_TEACHER_BUFFER}"

"${VENV_DIR}/bin/python" condtsc_pipeline.py \
  --mode distill \
  "${COMMON_ARGS[@]}" \
  --ipc "${IPC}" \
  --Iteration "${ITERATION}" \
  --eval_it "${EVAL_IT}" \
  --num_eval "${NUM_EVAL}" \
  --epoch_eval_train "${EPOCH_EVAL_TRAIN}" \
  --pix_init "${PIX_INIT}" \
  --max_start_epoch "${MAX_START_EPOCH}" \
  --expert_epochs "${EXPERT_EPOCHS}" \
  --syn_steps "${SYN_STEPS}" \
  --lr_teacher "${LR_TEACHER}" \
  --lr_feat "${LR_FEAT}" \
  --lr_lr "${LR_LR}" \
  --lambda_DM "${LAMBDA_DM}"
