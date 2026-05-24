import torch

from src.model.llama import LlamaConfig, LlamaForCausalLM
from src.model.reward import RewardModel
from src.train.ppo import PolicyWithValue, collect_rollout, ppo_update


def test_handwritten_ppo_rollout_and_update_smoke():
    cfg = {
        "ppo": {
            "max_new_tokens": 3,
            "temperature": 0.0,
            "top_p": 1.0,
            "init_kl_coef": 0.1,
            "gamma": 1.0,
            "lam": 0.95,
            "mini_batch_size": 1,
            "ppo_epochs": 1,
            "cliprange": 0.2,
            "cliprange_value": 0.2,
            "vf_coef": 0.5,
            "entropy_coef": 0.0,
            "grad_clip": 1.0,
        }
    }
    llm_cfg = LlamaConfig(
        vocab_size=64,
        n_layer=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        context_length=32,
    )
    policy = PolicyWithValue(LlamaForCausalLM(llm_cfg))
    ref = LlamaForCausalLM(llm_cfg)
    ref.load_state_dict(policy.policy.state_dict(), strict=True)
    reward = RewardModel(llm_cfg)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    batch = {
        "input_ids": torch.tensor([[1, 5, 6, 7], [1, 8, 9, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long),
    }

    rollout = collect_rollout(
        policy,
        ref,
        reward,
        batch,
        cfg,
        pad_id=0,
        eos_id=2,
        device=torch.device("cpu"),
        dtype=torch.float16,
        use_amp=False,
    )
    assert rollout.input_ids.shape[0] == 2
    assert rollout.response_mask.sum().item() > 0

    metrics = ppo_update(
        policy,
        optimizer,
        scaler,
        rollout,
        cfg,
        device=torch.device("cpu"),
        dtype=torch.float16,
        use_amp=False,
    )
    assert set(metrics) == {"policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"}
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
