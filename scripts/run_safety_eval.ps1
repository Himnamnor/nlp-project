# run_safety_eval.ps1 — RLHF 安全评测 (Windows)
param(
    [string]$Config = "configs/ppo.yaml",
    [string]$Ckpt = "",
    [string]$Label = "model",
    [string]$Output = ""
)
$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\Activate.ps1

$cli = @("-m", "src.eval.safety_eval", "--config", $Config, "--label", $Label)
if ($Ckpt -ne "")   { $cli += @("--ckpt", $Ckpt) }
if ($Output -ne "") { $cli += @("--output", $Output) }
python @cli @args
