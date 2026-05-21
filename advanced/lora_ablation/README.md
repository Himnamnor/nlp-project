# LoRA 消融实验

## 扫描维度

| 变量 | 取值 |
|------|------|
| rank r | 4, 8, 16 |
| target_modules | `q_proj,v_proj` / `q,k,v,o` / `q,k,v,o,gate,up,down` |

## 记录指标

- 可训练参数量
- PPO/DPO 峰值显存（`torch.cuda.max_memory_allocated`）
- 安全回答率（`safety_eval.py`）
- 每 step 耗时

## 运行模板

```bash
# 修改 configs/ppo.yaml 中 lora.r 和 target_modules 后重复：
python -m src.train.ppo --config configs/ppo.yaml
python -m src.eval.safety_eval --config configs/ppo.yaml --label lora_r8_qv
```

## TODO

- [ ] `run_ablation.py` 自动 sweep 并汇总 csv
- [ ] 结果表写入 report
