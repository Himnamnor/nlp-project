# run_ppo_conservative.ps1 — 保守 PPO 消融 (Windows)
param([string]$Config = "configs/ppo_conservative.yaml")
$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\Activate.ps1
python -m src.train.ppo --config $Config @args
