# run_pretrain.ps1 — 预训练全流程（Windows）
param(
    [string]$Config = "configs/pretrain.yaml"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\Activate.ps1

python -m src.tokenizer.train_bpe --config $Config
python -m src.data.pretrain --config $Config --prepare
python -m src.train.pretrain --config $Config @args
