# setup_cloud.ps1 — 本地 Windows 环境准备
# 用法: .\scripts\setup_cloud.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "Project root: $(Get-Location)"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host @"

Done. Quick start:
  python -m src.tokenizer.train_bpe --config configs/pretrain.yaml
  python -m src.data.pretrain --config configs/pretrain.yaml --prepare
  python -m src.train.pretrain --config configs/pretrain.yaml --max_steps 2

"@
