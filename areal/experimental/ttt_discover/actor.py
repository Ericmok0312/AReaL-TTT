# SPDX-License-Identifier: Apache-2.0

# areal/experimental/ttt_discover/actor.py
import math
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from areal.engine.fsdp_engine import FSDPEngine
from areal.experimental.ttt_discover.sampler import StateSampler
from areal.utils import logging
from areal.utils.functional import reward_overlong_penalty
from areal.utils.perf_tracer import trace_perf

logger = logging.getLogger("TTTDActor")

if TYPE_CHECKING:
    from areal.api.scheduler_api import Scheduler

    from .config import TTTDPPOActorConfig


class TTTDActor(FSDPEngine):
    """PPO Actor with TTT-Discover Entropic Objective support.

    Key difference from standard PPO:
    - Uses entropic objective for advantage computation: w_beta(a) - 1
    - Applies KL penalty at token level: A = (w_beta - 1) - lambda * KL
    - Operates on sequence-level rewards but produces token-level advantages
    """

    def __init__(self, config: "TTTDPPOActorConfig"):
        from areal.trainer.ppo.actor import PPOActor

        super().__init__(config)
        self.actor = PPOActor(config, self)

        # Validate configuration type. is_tttd_config lives on TTTDPPOConfig (top-level),
        # so check the actor config class directly instead.
        from areal.experimental.ttt_discover.config import TTTDPPOActorConfig

        adv_estimator = getattr(config, "adv_estimator", "<missing>")
        logger.info(
            f"[TTTDActor] config type: {type(config).__name__}, "
            f"is_tttd_actor_config: {isinstance(config, TTTDPPOActorConfig)}, "
            f"adv_estimator: {adv_estimator}"
        )

        if not isinstance(config, TTTDPPOActorConfig):
            import warnings

            warnings.warn(
                "Using standard PPOActorConfig with TTTDPPOActor. "
                "Entropic advantages will use default parameters (adv_estimator='gae'). "
                "Consider using TTTDPPOActorConfig for full functionality.",
                UserWarning,
            )

        self.config = config
        self.sampler = None

    @trace_perf("tttd_ppo_actor.compute_logp", category="compute")
    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @trace_perf("tttd_ppo_actor.compute_advantages", category="compute")
    @torch.no_grad()
    def compute_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Compute advantages using TTT-Discover entropic objective.

        Formula: A(a;s) = w_{beta(s)}(a) - 1 - lambda * log(pi_theta(a|s) / pi_ref(a|s))

        where:
        - w_{beta(s)}(a) = exp(beta * R(a)) / sum(exp(beta * R(a')))
        - R(a) is the external reward (sequence-level)
        - KL penalty is applied at token level
        """
        if getattr(self.config, "use_standard_grpo", False):
            return self._compute_standard_grpo_advantages(data)
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        # Reward Penalty on length
        if self.config.overlong_reward_penalty:
            overlong_tokens = self.config.overlong_tokens
            overlong_penalty_factor = self.config.overlong_penalty_factor

            assert overlong_tokens is not None
            assert overlong_penalty_factor is not None
            data = reward_overlong_penalty(
                data,
                overlong_tokens=overlong_tokens,
                overlong_penalty_factor=overlong_penalty_factor,
                max_response_length=self.config.max_new_tokens,
            )

        # Reward Scaling
        reward_score = data["rewards"]
        # Ensure reward_score is 1D [bs] for sequence-level reward processing
        if reward_score.dim() > 1:
            reward_score = reward_score.squeeze(-1)
        reward_score = (
            reward_score + self.actor.reward_bias
        ) * self.actor.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.actor.reward_clip, min=-self.actor.reward_clip
        )
        if self.actor.reward_norm:
            reward_score = self.actor.reward_norm(reward_score)

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)

        # Apply the mask to log probabilities.
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            prox_logp_value = data["prox_logp"]
            if prox_logp_value is None:
                raise ValueError(
                    "prox_logp is None but recompute_logprob=True. "
                    "This indicates compute_logp() was skipped incorrectly."
                )
            old_logp = data["logprobs"] = prox_logp_value
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                # prox logp not available, use inferenced logp
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp")
        if ref_logp is None:
            ref_logp = torch.zeros_like(old_logp)
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL divergence.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()

        kl_div = self.actor.kl_estimator(old_logp, ref_logp)
        kl_penalty = self.actor.kl_ctl * kl_div  # [bs, seqlen]
        kl_rewards = -kl_penalty
        kl_rewards[batch_indices, seqlens - 1] = 0

        # Compute Entropic Advantages.
        # TTT-Discover uses sequence-level external reward only (no KL aggregation)
        # to compute w_{beta(s)}(a), then subtracts token-level KL penalty.
        # Use globally-consistent group keys so that DP ranks can synchronize
        # the entropic group (one parent = one group).
        group_keys = self._extract_group_keys(data, bs)

        # Compute sequence-level entropic advantages: w_beta - 1
        logger.info(
            f"[TTTDActor.compute_advantages] adv_estimator={self.config.adv_estimator}"
        )
        entropic_adv_seq = self._compute_entropic_advantages(
            reward_score, group_keys, self.config.adv_estimator
        )  # [bs]

        # Broadcast to token level: [bs] -> [bs, seqlen]
        entropic_adv_token = entropic_adv_seq.unsqueeze(-1).expand(-1, max_seqlen)

        # Apply TTT-Discover formula: A = (w_beta - 1) - lambda * KL
        advantages = entropic_adv_token - kl_penalty
        advantages = advantages * loss_mask

        # Returns equal advantages since TTT-Discover has no value function baseline
        data["returns"] = advantages

        # Optionally perform advantage normalization.
        if self.actor.adv_norm is not None:
            advantages = self.actor.adv_norm(advantages, loss_mask)

        # Compute token-level total rewards (KL-regularized) for logging
        # TTT-Discover: external reward at EOS position, KL elsewhere
        tot_rewards = kl_rewards.clone()
        indices = torch.clip(seqlens - 2, min=0)
        tot_rewards[batch_indices, indices] += reward_score

        # Store data in the dict.
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards * loss_mask
        data["tot_rewards"] = tot_rewards
        data["loss_mask"] = loss_mask
        data["logprobs"] = old_logp

        return data

    def _extract_group_ids(self, data: dict, batch_size: int) -> torch.Tensor:
        """Extract group ids from data or auto-generate based on group_size config."""
        device = data["input_ids"].device

        # Use explicit group_ids if provided
        if "group_ids" in data:
            return data["group_ids"]

        # Auto-generate from group_size config
        group_size = getattr(self.config, "group_size", None)
        if group_size is not None and group_size > 0:
            group_ids = torch.arange(batch_size, device=device) // group_size
            return group_ids

        # Default: entire batch as one group (standard TTT-Discover behavior)
        return torch.zeros(batch_size, device=device, dtype=torch.long)

    def _extract_group_keys(self, data: dict, batch_size: int) -> list[str]:
        """Extract per-parent group keys for entropic advantage grouping.

        Each parent state defines one entropic group. The rollouts of a single
        parent are kept together by ``GroupedRolloutWorkflow``, so grouping by
        parent id here is sufficient and can be done locally on each DP rank.
        """
        parents = data.get("_breakthrough_parent")
        if parents is not None and len(parents) == batch_size:
            keys = []
            for i, state in enumerate(parents):
                sid = getattr(state, "id", None)
                if sid is None:
                    sid = f"unknown_{self.data_parallel_rank}_{i}"
                keys.append(str(sid))
            return keys

        # Fallback: per-rank integer groups (may mix parents if group_size
        # is not configured; rely on config.group_size to split correctly).
        if dist.is_initialized() and self.data_parallel_world_size > 1:
            logger.warning(
                "_breakthrough_parent metadata not found in batch; "
                "falling back to config.group_size for local grouping."
            )
        group_ids = self._extract_group_ids(data, batch_size)
        rank = self.data_parallel_rank if dist.is_initialized() else 0
        return [f"rank{rank}_group{int(gid)}" for gid in group_ids.tolist()]

    def _compute_entropic_advantages(
        self, rewards: torch.Tensor, group_keys: list[str], method: str
    ) -> torch.Tensor:
        """Compute sequence-level entropic advantages (w_beta - 1) for each group.

        Grouping is done per parent state. Because ``GroupedRolloutWorkflow``
        concatenates all rollouts of a parent into a single trajectory dict and
        ``DistRolloutCoordinator`` redistributes trajectory dicts atomically, each
        parent group is wholly present on one DP rank. Therefore beta can be solved
        locally without cross-rank communication.
        """
        advantages = torch.zeros_like(rewards)

        # Build local key -> index mapping.
        key_to_indices: dict[str, list[int]] = {}
        for i, key in enumerate(group_keys):
            key_to_indices.setdefault(key, []).append(i)

        if method == "mean_baseline":
            for idxs in key_to_indices.values():
                group_rewards = rewards[idxs]
                advantages[idxs] = group_rewards - group_rewards.mean()
        elif method == "entropic":
            beta = self.config.adv_estimator_beta
            for idxs in key_to_indices.values():
                advantages[idxs] = self._entropic_weight_minus_one(rewards[idxs], beta)
        elif method == "entropic_adaptive_beta":
            delta = getattr(self.config, "adv_estimator_target_kl", 0.693)
            beta_max = getattr(self.config, "adv_estimator_beta_max", 1e6)
            iters = getattr(self.config, "adv_estimator_beta_iters", 60)

            for idxs in key_to_indices.values():
                group_rewards = rewards[idxs]
                beta = self._solve_adaptive_beta(group_rewards, delta, beta_max, iters)
                logger.info(
                    f"[TTTDActor._compute_entropic_advantages] "
                    f"method={method}, group_size={group_rewards.shape[0]}, "
                    f"solved_beta={beta.item():.6f}"
                )
                advantages[idxs] = self._entropic_weight_minus_one(group_rewards, beta)
        else:
            raise ValueError(f"Unknown advantage estimator: {method}")

        return advantages

    def _entropic_weight_minus_one(
        self, rewards_G: torch.Tensor, beta: float
    ) -> torch.Tensor:
        """
        Compute w_beta - 1 where w_beta = exp(beta * R) / E[exp(beta * R)].

        Uses leave-one-out (LOO) estimation for Z to reduce variance:
        Z = (sum(exp(beta * R)) - exp(beta * R_i)) / (k - 1)
        """
        if beta == 0:
            return torch.zeros_like(rewards_G)

        # Numerical stability: subtract max before exp
        s_safe = rewards_G - rewards_G.max()
        exp_beta_r = torch.exp(beta * s_safe)
        k = exp_beta_r.shape[0]

        if k == 1:
            # Single sample: uniform weight
            Z = exp_beta_r
        else:
            # Leave-one-out estimation of partition function
            sum_exp = exp_beta_r.sum()
            Z = (sum_exp - exp_beta_r) / (k - 1)

        # w_beta = exp(beta * R) / Z
        w_beta = exp_beta_r / (Z + 1e-12)

        # Return w_beta - 1 (so that E[w_beta - 1] = 0)
        return w_beta - 1.0

    def _solve_adaptive_beta(
        self, rewards_G: torch.Tensor, delta: float, beta_max: float, iters: int
    ) -> torch.Tensor:
        """Solve for beta such that KL(q_beta || uniform) = delta."""

        r = rewards_G.float()
        k = r.shape[0]

        if k < 2:
            return r.new_tensor(0.0)

        logK = math.log(k)

        def kl_hat(beta_scalar: float) -> float:
            """Compute KL(q_beta || uniform) for given beta."""
            b = r.new_tensor(beta_scalar)
            logits = b * (r - r.max())
            logq = logits - torch.logsumexp(logits, dim=0)
            q = torch.exp(logq)
            kl = (q * (logq + logK)).sum()
            return float(kl.item())

        # Binary search for beta
        lo, hi = 0.0, 1.0

        # Expand search range if needed
        if kl_hat(hi) < delta:
            while hi < beta_max and kl_hat(hi) < delta:
                hi *= 2.0
            if kl_hat(hi) < delta:
                return r.new_tensor(hi)

        # Binary search
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if kl_hat(mid) < delta:
                lo = mid
            else:
                hi = mid

        return r.new_tensor(hi)

    def ppo_update(self, *args, **kwargs) -> None:
        self.actor.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config: "TTTDPPOActorConfig", scheduler: "Scheduler"):
        from areal.trainer.ppo.actor import PPOActorController

        return PPOActorController(train_engine=cls, config=config, scheduler=scheduler)

    def connect_sampler(self, sampler: StateSampler):
        """
        Connects State Sampler to actor, allows actor to control sampler synchronization.
        """
        if self.sampler is not None:
            self.logger.warning(
                "Sampler is already connected to actor. Overwriting existing sampler connection."
            )

        self.sampler = sampler
        # Use sampler.budget_s as the lightweight checkpoint save frequency.
        # create_sampler_from_config passes save_freq as budget_s.
        self._sampler_save_freq = max(1, int(getattr(sampler, "budget_s", 1) or 1))
        dist.barrier(
            group=self.cpu_group
        )  # Ensure all actors have connected their samplers before proceeding()

    def sync_sampler(
        self, local_children=None, local_parents=None, local_failed=None, step=None
    ):
        """
        Synchronize sampler state across all data parallel ranks.

        Three-phase pipeline: Gather -> Distribute (rank0 compute) -> Synchronize.
        """
        if self.sampler is None:
            raise RuntimeError("No sampler connected. Call connect_sampler() first.")

        # Short-circuit for single-node
        if not dist.is_initialized() or self.data_parallel_world_size <= 1:
            self._apply_updates_locally(
                local_children, local_parents, local_failed, step
            )
            return

        self.logger.info(
            f"[Rank {self.dp_rank}][Step {step}] sync_sampler START: "
            f"local_children={len(local_children or [])}, "
            f"local_parents={len(local_parents or [])}, "
            f"local_failed={len(local_failed or [])}"
        )

        # Phase 1: Gather updates from all ranks to rank 0
        self.logger.info(
            f"[Rank {self.dp_rank}][Step {step}] Phase 1: Gathering updates..."
        )
        gathered = self._gather_updates(
            local_children, local_parents, local_failed, step
        )
        self.logger.info(f"[Rank {self.dp_rank}][Step {step}] Phase 1: Gather complete")

        # Phase 2: Rank 0 applies updates and prepares state package
        self.logger.info(
            f"[Rank {self.dp_rank}][Step {step}] Phase 2: Applying updates..."
        )
        state_package = self._apply_updates(gathered, step)
        self.logger.info(f"[Rank {self.dp_rank}][Step {step}] Phase 2: Apply complete")

        # Phase 3: Broadcast and apply synchronized state to all ranks
        self.logger.info(
            f"[Rank {self.dp_rank}][Step {step}] Phase 3: Synchronizing state..."
        )
        self._synchronize_state(state_package, step)
        self.logger.info(f"[Rank {self.dp_rank}][Step {step}] Phase 3: Sync complete")

        # Phase 4: Save version snapshot for lazy PUCT sampling
        # Step N 完成后，sampler 状态已更新为 version N+1，供下一轮采样使用
        if hasattr(self.sampler, "save_version_snapshot") and step is not None:
            next_version = step + 1
            self.sampler.save_version_snapshot(next_version)
            self.logger.info(
                f"[Rank {self.dp_rank}][Step {step}] Phase 4: Saved PUCT snapshot version={next_version}"
            )

        self.logger.info(f"[Rank {self.dp_rank}][Step {step}] sync_sampler END")

    def _apply_updates_locally(self, children, parents, failed, step):
        """Apply updates without distributed communication (single-node shortcut).

        NOTE: Do NOT acquire sampler._lock here!
        sampler methods have their own locking to avoid reentrancy deadlock.
        """
        for f in failed or []:
            self.sampler.record_failed_rollout(f)
        if children:
            self.sampler.update_states(children, parents, save=False)
        if step is not None:
            self.sampler._current_step = step
            # Save lightweight sampler checkpoint every save_freq steps.
            if step % self._sampler_save_freq == 0:
                self.sampler._save(step)

        # Save version snapshot for lazy PUCT sampling (single-node case)
        # Step N 完成后，sampler 状态已更新为 version N+1，供下一轮采样使用
        if hasattr(self.sampler, "save_version_snapshot") and step is not None:
            next_version = step + 1
            self.sampler.save_version_snapshot(next_version)

    def _gather_updates(self, local_children, local_parents, local_failed, step):
        """
        Phase 1: Gather all local updates to rank 0.

        Uses gather_object for lower communication overhead (only rank 0 receives).
        Also gathers batch version mappings from all ranks for distributed consistency.

        Returns:
            Tuple (all_children, all_parents, all_failed, all_batch_mappings) on rank 0,
            or ([], [], [], {}) on other ranks.
        """
        from areal.experimental.ttt_discover.state import state_from_dict

        # Serialize
        c_dicts = [s.to_dict() for s in (local_children or [])]
        p_dicts = [s.to_dict() for s in (local_parents or [])]
        f_dicts = [s.to_dict() for s in (local_failed or [])]

        # Get batch version mappings (for lazy PUCT consistency)
        batch_mappings = (
            dict(self.sampler._batch_version_mappings) if self.sampler else {}
        )

        world_size = self.data_parallel_world_size
        is_rank0 = self.dp_rank == 0

        # Prepare containers (only rank 0 needs them)
        all_c = [None] * world_size if is_rank0 else None
        all_p = [None] * world_size if is_rank0 else None
        all_f = [None] * world_size if is_rank0 else None
        all_mappings = [None] * world_size if is_rank0 else None

        try:
            # Use gather_object for lower communication overhead
            dist.gather_object(c_dicts, all_c, dst=0, group=self.data_parallel_group)
            dist.gather_object(p_dicts, all_p, dst=0, group=self.data_parallel_group)
            dist.gather_object(f_dicts, all_f, dst=0, group=self.data_parallel_group)
            dist.gather_object(
                batch_mappings, all_mappings, dst=0, group=self.data_parallel_group
            )
        except Exception as e:
            self.logger.error(
                f"[Rank {self.dp_rank}] Gather failed at step {step}: {e}"
            )
            raise

        if not is_rank0:
            return [], [], [], {}

        # Deserialize on rank 0
        children = [state_from_dict(d) for lst in all_c if lst for d in lst]
        parents = [state_from_dict(d) for lst in all_p if lst for d in lst]
        failed = [state_from_dict(d) for lst in all_f if lst for d in lst]

        # Merge batch mappings from all ranks with conflict detection
        merged_mappings = {}
        conflicts = []
        for rank_idx, rank_mappings in enumerate(all_mappings):
            if not rank_mappings:
                continue
            for batch_id, version in rank_mappings.items():
                if batch_id in merged_mappings:
                    if merged_mappings[batch_id] != version:
                        # Conflict detected: same batch_id, different version
                        conflicts.append(
                            {
                                "batch_id": batch_id,
                                "existing_version": merged_mappings[batch_id],
                                "conflict_version": version,
                                "conflict_rank": rank_idx,
                            }
                        )
                else:
                    merged_mappings[batch_id] = version

        if conflicts:
            error_msg = (
                f"[Step {step}] BATCH_VERSION_CONFLICT detected: {len(conflicts)} conflicts! "
                f"Details: {conflicts[:5]}..."  # Show first 5
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        if step and merged_mappings:
            self.logger.info(
                f"[Step {step}] Merged batch mappings: {len(merged_mappings)} entries"
            )

        if step:
            self.logger.info(
                f"[Step {step}] Gathered from {world_size} ranks: "
                f"children={len(children)}, parents={len(set(p.id for p in parents))}, failed={len(failed)}"
            )
        return children, parents, failed, merged_mappings

    def _apply_updates(self, gathered_data, step):
        """
        Phase 2: Rank 0 applies updates and prepares state for broadcast.

        NOTE: Do NOT acquire sampler._lock here!
        sampler.update_states() and record_failed_rollout() have their own locking.
        Using lock here would cause reentrancy deadlock.

        Returns:
            State package (dict) if rank 0, else None.
        """
        if self.dp_rank != 0:
            return None

        children, parents, failed, merged_batch_mappings = gathered_data
        self.logger.info(
            f"[Step {step}] _apply_updates: children={len(children)}, parents={len(parents)}, failed={len(failed)}"
        )

        # Update rank 0's batch version mappings with merged result
        # (other ranks will get this via deserialize_full_state)
        if merged_batch_mappings:
            self.sampler._batch_version_mappings.update(merged_batch_mappings)
            self.logger.info(
                f"[Step {step}] Updated batch mappings: {len(merged_batch_mappings)} entries"
            )

        # Record failures (record_failed_rollout has its own lock)
        if failed:
            _T_before = self.sampler._T
            for f in failed:
                self.sampler.record_failed_rollout(f)
            if step:
                self.logger.info(
                    f"[Step {step}] Recorded {len(failed)} failures, _T: {_T_before} -> {self.sampler._T}"
                )

        # Record successes (update_states has its own lock)
        if children:
            _T_before = self.sampler._T
            self.sampler.update_states(children, parents, save=False)
            if step:
                self.logger.info(
                    f"[Step {step}] Updated states: _T={self.sampler._T} (+{self.sampler._T - _T_before}), "
                    f"total_states={len(self.sampler._states)}"
                )

        # Save and package (flush has its own lock).
        # Only dump a lightweight checkpoint every save_freq steps; the final
        # full sampler state is saved separately at the end of training.
        if step:
            if step % self._sampler_save_freq == 0:
                self.sampler.flush(step)
            else:
                # Still trim the buffer to avoid unbounded growth.
                self.sampler._finalize_and_save(step=None)

        return {
            "states": [s.to_dict() for s in self.sampler._states],
            "initial_states": [s.to_dict() for s in self.sampler._initial_states],
            "T": self.sampler._T,
            "n": self.sampler._n,
            "m": self.sampler._m,
            "current_step": self.sampler._current_step,
            "last_sampled_states": [
                s.to_dict() for s in self.sampler._last_sampled_states
            ],
            "last_sampled_indices": self.sampler._last_sampled_indices,
            "last_puct_stats": self.sampler._last_puct_stats,
            "last_scale": self.sampler._last_scale,
            # Include merged batch version mappings from all ranks
            "batch_version_mappings": merged_batch_mappings,
        }

    def _synchronize_state(self, state_package, step):
        """
        Phase 3: Broadcast state from rank 0 and apply to all ranks.
        """

        # Broadcast
        package = [state_package] if self.dp_rank == 0 else [None]
        try:
            dist.broadcast_object_list(package, src=0, group=self.data_parallel_group)
        except Exception as e:
            self.logger.error(
                f"[Rank {self.dp_rank}] Broadcast failed at step {step}: {e}"
            )
            raise

        # Apply (non-rank-0 only, rank 0 already has the state)
        if self.dp_rank != 0:
            data = package[0]
            self.sampler.deserialize_full_state(data)

            if step:
                self.logger.info(
                    f"[Rank {self.dp_rank}][Step {step}] Synchronized: "
                    f"T={self.sampler._T}, states={len(self.sampler._states)}"
                )

        # All ranks wait here to ensure synchronization is complete
        if dist.is_initialized():
            dist.barrier(group=self.data_parallel_group)

    def _compute_standard_grpo_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        """Standard GRPO advantage computation (DeepSeekMath-style).

        Advantage = (reward - group_mean) / (group_std + eps) - lambda * KL

        Optionally adds a small anchor bonus for rollouts close to the sampler's
        best known reward (best_reward_anchor=True).
        """
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        device = data["input_ids"].device
        batch_indices = torch.arange(bs, device=device, dtype=torch.long)

        # Reward Penalty on length
        if self.config.overlong_reward_penalty:
            data = reward_overlong_penalty(
                data,
                overlong_tokens=self.config.overlong_tokens,
                overlong_penalty_factor=self.config.overlong_penalty_factor,
                max_response_length=self.config.max_new_tokens,
            )

        # Reward Scaling
        reward_score = data["rewards"]
        if reward_score.dim() > 1:
            reward_score = reward_score.squeeze(-1)
        reward_score = (
            reward_score + self.actor.reward_bias
        ) * self.actor.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.actor.reward_clip, min=-self.actor.reward_clip
        )
        if self.actor.reward_norm:
            reward_score = self.actor.reward_norm(reward_score)

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)

        # Apply the mask to log probabilities (same as standard PPO)
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            prox_logp_value = data["prox_logp"]
            if prox_logp_value is None:
                raise ValueError(
                    "prox_logp is None but recompute_logprob=True. "
                    "This indicates compute_logp() was skipped incorrectly."
                )
            old_logp = data["logprobs"] = prox_logp_value
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp")
        if ref_logp is None:
            ref_logp = torch.zeros_like(old_logp)
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL divergence.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        kl_div = self.actor.kl_estimator(old_logp, ref_logp)
        kl_penalty = self.actor.kl_ctl * kl_div
        kl_rewards = -kl_penalty
        kl_rewards[batch_indices, seqlens - 1] = 0

        # Extract group ids
        group_ids = self._extract_group_ids(data, bs)

        # Compute standard GRPO advantages per group
        grpo_adv_seq = torch.zeros_like(reward_score)
        unique_groups = torch.unique(group_ids)
        for gid in unique_groups:
            mask = group_ids == gid
            group_rewards = reward_score[mask]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            grpo_adv_seq[mask] = (group_rewards - group_mean) / group_std

        # Optional: best reward anchor for small-group stability
        if (
            getattr(self.config, "best_reward_anchor", False)
            and self.sampler is not None
        ):
            try:
                best_solution = self.sampler.get_best_solution()
                if best_solution is not None:
                    best_value = best_solution.get("value")
                    if best_value is not None:
                        best_reward = (
                            best_value + self.actor.reward_bias
                        ) * self.actor.reward_scaling
                        # Small Gaussian bonus for rollouts close to best
                        for gid in unique_groups:
                            mask = group_ids == gid
                            group_rewards = reward_score[mask]
                            group_std = group_rewards.std() + 1e-8
                            closeness = torch.exp(
                                -((group_rewards - best_reward) ** 2)
                                / (2 * group_std**2)
                            )
                            grpo_adv_seq[mask] = grpo_adv_seq[mask] + 0.1 * closeness
            except Exception:
                pass  # Silently skip if sampler doesn't support get_best_solution

        # Broadcast to token level
        grpo_adv_token = grpo_adv_seq.unsqueeze(-1).expand(-1, max_seqlen)
        advantages = grpo_adv_token - kl_penalty
        advantages = advantages * loss_mask

        data["returns"] = advantages
        if self.actor.adv_norm is not None:
            advantages = self.actor.adv_norm(advantages, loss_mask)

        # Compute total rewards for logging
        tot_rewards = kl_rewards.clone()
        indices = torch.clip(seqlens - 2, min=0)
        seq_no_eos_mask = seqlens == attn_mask.shape[1]
        if getattr(self.actor, "mask_no_eos_with_zero", False):
            tot_rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            tot_rewards[batch_indices, indices] += reward_score

        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards * loss_mask
        data["tot_rewards"] = tot_rewards
        data["loss_mask"] = loss_mask
        data["logprobs"] = old_logp

        return data
