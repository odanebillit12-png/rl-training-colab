"""
PPO (Proximal Policy Optimization) agent — shared by both drawing and NPC RL.

Uses PyTorch. Exports to ONNX for direct loading in Godot.

Architecture: shared MLP backbone → policy head (actor) + value head (critic)
This is the same algorithm family that DeepMind used for game-playing agents.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ── Network ───────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """Shared MLP backbone with separate actor/critic heads."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, action_dim)   # logits
        self.critic = nn.Linear(hidden, 1)             # state value

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        return self.actor(features), self.critic(features).squeeze(-1)

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value


# ── Rollout Buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores one rollout worth of transitions for PPO update."""

    def __init__(self, size: int, obs_dim: int):
        self.size    = size
        self.obs     = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values  = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones   = np.zeros(size, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, value, log_prob, done):
        i = self.ptr % self.size
        self.obs[i]       = obs
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.values[i]    = value
        self.log_probs[i] = log_prob
        self.dones[i]     = float(done)
        self.ptr += 1

    def compute_returns(self, gamma: float = 0.99, lam: float = 0.95) -> np.ndarray:
        """GAE-Lambda advantage estimation."""
        n = min(self.ptr, self.size)
        advantages = np.zeros(n, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(n)):
            next_val   = self.values[t + 1] if t + 1 < n else 0.0
            delta      = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            last_adv   = delta + gamma * lam * (1 - self.dones[t]) * last_adv
            advantages[t] = last_adv
        return advantages

    def get(self) -> dict:
        n = min(self.ptr, self.size)
        advantages = self.compute_returns()
        returns    = advantages + self.values[:n]
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return {
            "obs":        torch.tensor(self.obs[:n]),
            "actions":    torch.tensor(self.actions[:n]),
            "returns":    torch.tensor(returns),
            "advantages": torch.tensor(advantages),
            "log_probs":  torch.tensor(self.log_probs[:n]),
        }

    def reset(self):
        self.ptr = 0


# ── PPO Agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    PPO agent with:
    - Clipped surrogate objective
    - Value function loss
    - Entropy bonus (encourages exploration)
    - ONNX export for Godot loading
    """

    def __init__(
        self,
        obs_dim:     int,
        action_dim:  int,
        hidden:      int   = 128,
        lr:          float = 3e-4,
        clip_eps:    float = 0.2,
        value_coef:  float = 0.5,
        entropy_coef:float = 0.01,
        n_epochs:    int   = 4,
        batch_size:  int   = 64,
        rollout_size:int   = 512,
        gamma:       float = 0.99,
        lam:         float = 0.95,
        device:      str   = "cpu",
    ):
        self.obs_dim      = obs_dim
        self.action_dim   = action_dim
        self.clip_eps     = clip_eps
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.gamma        = gamma
        self.lam          = lam
        self.device       = torch.device(device)

        self.net    = ActorCritic(obs_dim, action_dim, hidden).to(self.device)
        self.optim  = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer(rollout_size, obs_dim)

        self.total_updates = 0
        self.total_steps   = 0

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """Returns (action_idx, log_prob, value)."""
        with torch.no_grad():
            obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            action, log_prob, value = self.net.get_action(obs_t)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def store(self, obs, action, reward, value, log_prob, done):
        self.buffer.add(obs, action, reward, value, log_prob, done)
        self.total_steps += 1

    def update(self) -> dict:
        """Run PPO update on buffered rollout. Returns loss stats."""
        data  = {k: v.to(self.device) for k, v in self.buffer.get().items()}
        n     = data["obs"].shape[0]
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}

        for _ in range(self.n_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                obs_b    = data["obs"][idx]
                act_b    = data["actions"][idx]
                ret_b    = data["returns"][idx]
                adv_b    = data["advantages"][idx]
                old_lp_b = data["log_probs"][idx]

                logits, values = self.net(obs_b)
                dist    = Categorical(logits=logits)
                new_lp  = dist.log_prob(act_b)
                entropy = dist.entropy().mean()

                ratio       = (new_lp - old_lp_b).exp()
                surr1       = ratio * adv_b
                surr2       = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = nn.functional.mse_loss(values, ret_b)
                loss        = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optim.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"]  += float(value_loss.item())
                stats["entropy"]     += float(entropy.item())
                stats["updates"]     += 1

        self.buffer.reset()
        self.total_updates += 1
        n_b = max(1, stats["updates"])
        return {k: round(v / n_b, 4) if isinstance(v, float) else v for k, v in stats.items()}

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "net":           self.net.state_dict(),
            "optim":         self.optim.state_dict(),
            "obs_dim":       self.obs_dim,
            "action_dim":    self.action_dim,
            "total_updates": self.total_updates,
            "total_steps":   self.total_steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optim.load_state_dict(ckpt["optim"])
        self.total_updates = ckpt.get("total_updates", 0)
        self.total_steps   = ckpt.get("total_steps", 0)

    def export_onnx(self, path: str, agent_name: str = "agent"):
        """
        Export the actor to ONNX so Godot can load it via GDNative/ONNX runtime.
        Only the actor (policy) is exported — Godot just needs action probabilities.
        """
        dummy  = torch.zeros(1, self.obs_dim)
        actor  = _ActorOnly(self.net).eval()
        torch.onnx.export(
            actor, dummy, path,
            input_names=["obs"],
            output_names=["action_probs"],
            dynamic_axes={"obs": {0: "batch"}},
            opset_version=11,
        )
        print(f"  📦 ONNX exported → {path}  ({agent_name})")
        return path


class _ActorOnly(nn.Module):
    """Wraps ActorCritic to export only the softmax policy (for Godot)."""
    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.backbone = ac.backbone
        self.actor    = ac.actor

    def forward(self, obs):
        return torch.softmax(self.actor(self.backbone(obs)), dim=-1)
