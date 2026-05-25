# run_dpo.ps1 — DPO 训练 (Windows)
param([string]$Config = "configs/dpo.yaml")
$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\Activate.ps1
python -m src.train.dpo --config $Config @args
