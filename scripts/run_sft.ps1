# run_sft.ps1
param([string]$Config = "configs/sft.yaml")
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\Activate.ps1
python -m src.train.sft --config $Config
