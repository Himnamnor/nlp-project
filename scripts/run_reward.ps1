# run_reward.ps1 — 奖励模型训练 (Windows)
param([string]$Config = "configs/reward.yaml")
$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\Activate.ps1
python -m src.train.reward --config $Config @args
