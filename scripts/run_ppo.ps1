# run_ppo.ps1 — PPO RLHF (Windows)
param([string]$Config = "configs/ppo.yaml")
$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\Activate.ps1
python -m src.train.ppo --config $Config @args
