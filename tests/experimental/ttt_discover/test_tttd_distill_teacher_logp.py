# SPDX-License-Identifier: Apache-2.0
"""Unit tests for teacher logp alignment in TTT-Discover distillation.

These tests target the pure alignment helper and the privileged-teacher-logp
method of ``TTTDDistillTrainer``.  They avoid heavy engine initialization by
mocking the trainer dependencies.
"""

from unittest.mock import MagicMock

import pytest
import torch

from areal.experimental.ttt_discover.examples.train_tttd_distill import (
    TTTDDistillTrainer,
    align_teacher_completion_logps,
)


def _make_state(code: str = "int main() {}", value: float = 0.5, state_id: str = "abc"):
    """Return a minimal mock PUCT state."""
    state = MagicMock()
    state.code = code
    state.value = value
    state.id = state_id
    state.raw_score = None
    return state


class TestAlignTeacherCompletionLogps:
    """Tests for the pure alignment helper."""

    def test_align_teacher_completion_logps_basic(self):
        """Single item: completion logps are shifted by prompt length difference."""
        # Teacher prompt len = 3, student prompt len = 2, completion len = 2.
        # Teacher seq len = 3 + 2 = 5; teacher_logps_full[i, p] = p.
        teacher_logps_full = torch.arange(5, dtype=torch.float32).unsqueeze(0)
        aligned = align_teacher_completion_logps(
            teacher_logps_full,
            teacher_prompt_lens=[3],
            student_prompt_lens=[2],
            comp_lens=[2],
            student_seqlen=4,
        )
        expected = torch.zeros(1, 4, dtype=torch.float32)
        # Completion logps are at teacher positions 2 and 3 -> values 2 and 3.
        # They map to student positions 1 and 2.
        expected[0, 1:3] = torch.tensor([2.0, 3.0])
        torch.testing.assert_close(aligned, expected)

    def test_align_teacher_completion_logps_zero_completion(self):
        """Zero-length completion leaves the output zero-filled."""
        teacher_logps_full = torch.arange(5, dtype=torch.float32).unsqueeze(0)
        aligned = align_teacher_completion_logps(
            teacher_logps_full,
            teacher_prompt_lens=[3],
            student_prompt_lens=[2],
            comp_lens=[0],
            student_seqlen=4,
        )
        expected = torch.zeros(1, 4, dtype=torch.float32)
        torch.testing.assert_close(aligned, expected)

    def test_align_teacher_completion_logps_batch_different_prompts(self):
        """Batch items with different teacher/student prompt lengths."""
        # Item 0: T=2, S=3, C=2 -> teacher positions 1,2 -> student positions 2,3.
        # Item 1: T=4, S=2, C=3 -> teacher positions 3,4,5 -> student positions 1,2,3.
        teacher_logps_full = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0, 0.0, 0.0],  # padded to teacher seq len 6
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            ],
            dtype=torch.float32,
        )
        aligned = align_teacher_completion_logps(
            teacher_logps_full,
            teacher_prompt_lens=[2, 4],
            student_prompt_lens=[3, 2],
            comp_lens=[2, 3],
            student_seqlen=5,
        )
        expected = torch.zeros(2, 5, dtype=torch.float32)
        expected[0, 2:4] = torch.tensor([1.0, 2.0])
        expected[1, 1:4] = torch.tensor([3.0, 4.0, 5.0])
        torch.testing.assert_close(aligned, expected)


class TestComputePrivilegedTeacherLogp:
    """Tests for ``TTTDDistillTrainer._compute_privileged_teacher_logp``."""

    @pytest.fixture
    def trainer(self):
        """Return a trainer instance with mocked dependencies (no __init__)."""
        trainer = object.__new__(TTTDDistillTrainer)
        trainer.is_multi_teacher = False

        config = MagicMock()
        config.gconfig.n_samples = 2
        config.use_breakthrough_opd = False
        config.privileged_prompt_mode = "hint"
        config.enable_thinking = False
        trainer.config = config

        actor = MagicMock()
        actor.data_parallel_world_size = 1
        actor.data_parallel_rank = 0
        trainer.actor = actor

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        def _apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        ):
            # Deterministic tokenization: one id per character of prompt content.
            content = messages[0]["content"]
            return list(range(1, len(content) + 2))

        tokenizer.apply_chat_template = _apply_chat_template
        trainer.tokenizer = tokenizer

        teacher_sampler = MagicMock()
        teacher_sampler.sample_states = lambda n: [_make_state()] * n
        trainer.teacher_sampler = teacher_sampler

        env = MagicMock()
        env.get_prompt = lambda state: "privileged prompt"
        trainer.env = env

        trainer._build_hint = lambda state: "\n\n[Hint]\n"
        trainer._milestone_hints_data = None

        def _compute_logp(batches):
            batch = batches[0]
            batch_size, seq_len = batch["input_ids"].shape
            # Deterministic logps: position index as the logp value.
            logps = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).repeat(
                batch_size, 1
            )
            return [logps]

        teacher = MagicMock()
        teacher.compute_logp = _compute_logp
        trainer.teacher = teacher

        return trainer

    def test_single_teacher_hint_mode_aligns_logps(self, trainer):
        """In hint mode, teacher logps align to student completion positions."""
        # Student prompt text length = 14 -> token ids [1..15] -> prompt_len = 15.
        # Student completion tokens [20, 21] -> comp_len = 2.
        # Teacher prompt = student prompt + hint.  Hint "\n\n[Hint]\n" is 9 chars,
        # so teacher prompt is 23 chars -> token ids [1..24] -> teacher_prompt_len = 24.
        # Teacher seq len = 24 + 2 = 26.
        # Teacher completion logps at positions 23, 24 -> values 23, 24.
        # Student completion positions: prompt_len-1=14 to prompt_len+comp_len-2=15.
        input_ids = torch.tensor(
            [[10] * 15 + [20, 21]], dtype=torch.int32
        )
        attention_mask = torch.ones(1, 17, dtype=torch.bool)
        loss_mask = torch.tensor(
            [[0] * 15 + [1, 1]], dtype=torch.int32
        )
        rollout_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "_student_prompts": ["student prompt"],
        }

        aligned_logp, teacher_input_ids, teacher_attention_mask, teacher_loss_mask = (
            trainer._compute_privileged_teacher_logp(rollout_batch)
        )

        expected = torch.zeros(1, 17, dtype=torch.float32)
        expected[0, 14:16] = torch.tensor([23.0, 24.0])
        torch.testing.assert_close(aligned_logp, expected)

        # Teacher batch should be teacher_prompt + student_completion.
        assert teacher_input_ids.shape[0] == 1
        assert teacher_input_ids.shape[1] == 26
        # First 24 tokens are the teacher prompt (loss mask 0), last 2 are completion.
        torch.testing.assert_close(
            teacher_loss_mask[0].float(),
            torch.tensor([0.0] * 24 + [1.0, 1.0]),
        )

    def test_single_teacher_continuation_mode(self, trainer):
        """In continuation mode, teacher prompt comes from env.get_prompt."""
        trainer.config.privileged_prompt_mode = "continuation"
        # env.get_prompt returns "privileged prompt" (length 16) -> prompt_len = 17.
        input_ids = torch.tensor([[10, 11, 20, 21]], dtype=torch.int32)
        attention_mask = torch.ones(1, 4, dtype=torch.bool)
        loss_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
        rollout_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "_student_prompts": ["student prompt"],
        }

        aligned_logp, *_ = trainer._compute_privileged_teacher_logp(rollout_batch)

        # teacher_prompt_len = 17, comp_len = 2 -> teacher positions 16, 17 -> values 16,17.
        # student_prompt_len = 2 -> student positions 1, 2.
        expected = torch.zeros(1, 4, dtype=torch.float32)
        expected[0, 1:3] = torch.tensor([16.0, 17.0])
        torch.testing.assert_close(aligned_logp, expected)

    def test_single_teacher_zero_completion(self, trainer):
        """Zero-length completion yields zero teacher logps."""
        input_ids = torch.tensor([[10, 11]], dtype=torch.int32)
        attention_mask = torch.ones(1, 2, dtype=torch.bool)
        loss_mask = torch.zeros(1, 2, dtype=torch.int32)
        rollout_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "_student_prompts": ["student prompt"],
        }

        aligned_logp, *_ = trainer._compute_privileged_teacher_logp(rollout_batch)
        expected = torch.zeros(1, 2, dtype=torch.float32)
        torch.testing.assert_close(aligned_logp, expected)


class TestComputePrivilegedTeacherLogpMulti:
    """Tests for the multi-teacher privileged teacher logp path."""

    @pytest.fixture
    def trainer_multi(self):
        """Return a multi-teacher trainer with mocked dependencies."""
        trainer = object.__new__(TTTDDistillTrainer)
        trainer.is_multi_teacher = True

        config = MagicMock()
        config.gconfig.n_samples = 2
        config.enable_thinking = False
        config.multi_teacher_hint_mode = "best"
        config.multi_teacher_hint_min_improvement = 0.0
        config.multi_teacher_hint_deterministic = True
        config.multi_teacher_hint_k = 1
        trainer.config = config

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        def _apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        ):
            content = messages[0]["content"]
            return list(range(1, len(content) + 2))

        tokenizer.apply_chat_template = _apply_chat_template
        trainer.tokenizer = tokenizer

        teacher_sampler = MagicMock()
        teacher_sampler.get_hint_states = MagicMock(
            return_value=[("best", _make_state(code="hint code", value=0.9))]
        )
        trainer.teacher_samplers = {"p1": teacher_sampler}
        trainer.teacher_lora_paths = {"p1": "/fake/lora"}
        trainer._multi_teacher_hint_counters = {"p1": 0}

        trainer._load_peft_lora_adapter = MagicMock()
        trainer._build_hint = lambda state: "\n\n[Hint]\n"

        def _compute_logp(batches):
            batch = batches[0]
            batch_size, seq_len = batch["input_ids"].shape
            logps = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).repeat(
                batch_size, 1
            )
            return [logps]

        teacher = MagicMock()
        teacher.compute_logp = _compute_logp
        trainer.teacher = teacher

        return trainer

    def test_multi_teacher_aligns_per_problem(self, trainer_multi):
        """Multi-teacher path loads LoRA and aligns per-problem logps."""
        # Two rollouts for problem p1.
        # Student prompt text "student prompt" -> prompt_len = 15.
        # Hint "\n\n[Hint]\n" is 9 chars -> teacher prompt len = 24.
        # Completion [20, 21] -> comp_len = 2.
        input_ids = torch.tensor(
            [[10] * 15 + [20, 21], [10] * 15 + [22, 23]], dtype=torch.int32
        )
        attention_mask = torch.ones(2, 17, dtype=torch.bool)
        loss_mask = torch.tensor(
            [[0] * 15 + [1, 1], [0] * 15 + [1, 1]], dtype=torch.int32
        )
        rollout_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "_student_prompts": ["student prompt", "student prompt"],
            "_problem_ids": ["p1", "p1"],
        }

        aligned_logp, teacher_input_ids, teacher_attention_mask, teacher_loss_mask = (
            trainer_multi._compute_privileged_teacher_logp(rollout_batch)
        )

        expected = torch.zeros(2, 17, dtype=torch.float32)
        expected[:, 14:16] = torch.tensor([23.0, 24.0])
        torch.testing.assert_close(aligned_logp, expected)
        trainer_multi._load_peft_lora_adapter.assert_called_once_with(
            trainer_multi.teacher, "/fake/lora"
        )
