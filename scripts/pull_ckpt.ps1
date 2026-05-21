# pull_ckpt.ps1 — 从 AutoDL 拉回 checkpoint 到本机
# 用法: 修改下方变量后运行 .\scripts\pull_ckpt.ps1

$RemoteHost = "connect.autodl.com"
$RemotePort = 12345          # AutoDL 实例页显示的 SSH 端口
$RemoteUser = "root"
$RemotePath = "/root/autodl-tmp/Project/checkpoints"
$LocalPath  = "G:\我的云端硬盘\ZuxiChen\大三下\NLP\Project\checkpoints"

New-Item -ItemType Directory -Force -Path $LocalPath | Out-Null

Write-Host "Pulling $RemotePath -> $LocalPath"
scp -P $RemotePort -r "${RemoteUser}@${RemoteHost}:${RemotePath}/*" $LocalPath

Write-Host "Done."
