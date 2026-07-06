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
        student_prompt = "student prompt"
        hint = trainer._build_hint(_make_state())  # "\n\n[Hint]\n"
        teacher_prompt_len = len(
            trainer.tokenizer.apply_chat_template(
                [{"role": "user", "content": student_prompt + hint}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )

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
            "_student_prompts": [student_prompt],
        }

        aligned_logp, teacher_input_ids, teacher_attention_mask, teacher_loss_mask = (
            trainer._compute_privileged_teacher_logp(rollout_batch)
        )

        # Completion logps are at teacher positions teacher_prompt_len-1 and
        # teacher_prompt_len; because our mock returns position index as logp.
        expected = torch.zeros(1, 17, dtype=torch.float32)
        expected[0, 14:16] = torch.tensor(
            [float(teacher_prompt_len - 1), float(teacher_prompt_len)]
        )
        torch.testing.assert_close(aligned_logp, expected)

        # Teacher batch should be teacher_prompt + student_completion.
        assert teacher_input_ids.shape[0] == 1
        assert teacher_input_ids.shape[1] == teacher_prompt_len + 2
        torch.testing.assert_close(
            teacher_loss_mask[0].float(),
            torch.tensor([0.0] * teacher_prompt_len + [1.0, 1.0]),
        )

    def test_single_teacher_continuation_mode(self, trainer):
        """In continuation mode, teacher prompt comes from env.get_prompt."""
        trainer.config.privileged_prompt_mode = "continuation"
        teacher_prompt = trainer.env.get_prompt(_make_state())
        teacher_prompt_len = len(
            trainer.tokenizer.apply_chat_template(
                [{"role": "user", "content": teacher_prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )

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

        # student_prompt_len = 2 -> student positions 1, 2.
        expected = torch.zeros(1, 4, dtype=torch.float32)
        expected[0, 1:3] = torch.tensor(
            [float(teacher_prompt_len - 1), float(teacher_prompt_len)]
        )
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


class TestComputePrivilegedTeacherLogpFormula:
    """Verify the OPD formula: teacher_logp[n] = log p_T(C_n | P, C_{<n}).

    The distillation loss sums over completion tokens the divergence between
    teacher distribution p_T(· | x, y*, ŷ_{<n}) and student distribution
    p_S(· | x, ŷ_{<n}).  These tests verify that the teacher logps we feed into
    AReaL's KDRL path are conditioned on the privileged prompt plus the exact
    completion prefix.
    """

    @pytest.fixture
    def trainer_formula(self):
        """Return a trainer whose teacher logps depend on the full prefix."""
        trainer = object.__new__(TTTDDistillTrainer)
        trainer.is_multi_teacher = False

        config = MagicMock()
        config.gconfig.n_samples = 1
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
        # Exact tokenization map so we can compute prefix sums by hand.
        token_map = {
            "ab": [1, 2],
            "abH": [1, 2, 10],
        }

        def _apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        ):
            return token_map.get(messages[0]["content"], [1])

        tokenizer.apply_chat_template = _apply_chat_template
        trainer.tokenizer = tokenizer

        teacher_sampler = MagicMock()
        teacher_sampler.sample_states = lambda n: [_make_state()] * n
        trainer.teacher_sampler = teacher_sampler

        env = MagicMock()
        trainer.env = env

        # Hint "H" makes privileged prompt "abH" -> [1, 2, 10].
        trainer._build_hint = lambda state: "H"
        trainer._milestone_hints_data = None

        def _compute_logp(batches):
            """Return prefix-sum logps: logp at position j = sum(input_ids[:j]).

            This simulates a causal LM where the distribution at position j is a
            deterministic function of the prefix input_ids[:j].  For completion
            token C_n at teacher position |P| - 1 + n, the prefix is
            P + C_{<n}, so the returned logp depends exactly on the privileged
            prompt and previous completion tokens.
            """
            input_ids = batches[0]["input_ids"].to(torch.int64)
            batch_size, seq_len = input_ids.shape
            zeros = torch.zeros(batch_size, 1, dtype=torch.float32)
            if seq_len == 1:
                prefix_sums = zeros
            else:
                prefix_sums = torch.cat(
                    [zeros, input_ids[:, :-1].cumsum(dim=1).float()], dim=1
                )
            return [prefix_sums]

        teacher = MagicMock()
        teacher.compute_logp = _compute_logp
        trainer.teacher = teacher

        return trainer

    def test_teacher_logp_conditions_on_privileged_prompt_and_prefix(self, trainer_formula):
        """Teacher logp for C_n uses privileged prompt P and completion prefix C_{<n}."""
        # Student prompt "ab" -> [1, 2], completion [20, 21].
        # Privileged prompt "abH" -> [1, 2, 10].
        # Teacher input: [1, 2, 10, 20, 21].
        # logp(C_1) at pos 3 = sum([1, 2, 10])            = 13
        # logp(C_2) at pos 4 = sum([1, 2, 10, 20])        = 33
        # Mapped to student positions 1 and 2.
        input_ids = torch.tensor([[1, 2, 20, 21]], dtype=torch.int32)
        attention_mask = torch.ones(1, 4, dtype=torch.bool)
        loss_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
        rollout_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "_student_prompts": ["ab"],
        }

        aligned_logp, teacher_input_ids, teacher_attention_mask, teacher_loss_mask = (
            trainer_formula._compute_privileged_teacher_logp(rollout_batch)
        )

        expected = torch.zeros(1, 4, dtype=torch.float32)
        expected[0, 1:3] = torch.tensor([13.0, 33.0])
        torch.testing.assert_close(aligned_logp, expected)

        # Teacher batch structure: privileged prompt + completion.
        torch.testing.assert_close(
            teacher_input_ids[0],
            torch.tensor([1, 2, 10, 20, 21], dtype=torch.int32),
        )
        torch.testing.assert_close(
            teacher_loss_mask[0].float(),
            torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]),
        )

    def test_teacher_logp_changes_with_privileged_prompt(self, trainer_formula):
        """Different privileged hint gives different teacher logps for same completion."""
        # Base case: hint "H" -> privileged prompt "abH" -> [1, 2, 10].
        input_ids = torch.tensor([[1, 2, 20, 21]], dtype=torch.int32)
        attention_mask = torch.ones(1, 4, dtype=torch.bool)
        loss_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
        rollout_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "_student_prompts": ["ab"],
        }
        aligned_base, *_ = trainer_formula._compute_privileged_teacher_logp(
            rollout_batch
        )

        # Change hint so privileged prompt becomes "abX" -> [1, 2, 99].
        trainer_formula.tokenizer._token_map = {
            "ab": [1, 2],
            "abX": [1, 2, 99],
        }
        trainer_formula._build_hint = lambda state: "X"
        aligned_changed, *_ = trainer_formula._compute_privileged_teacher_logp(
            rollout_batch
        )

        # The completion logps must differ because the privileged context changed.
        assert not torch.allclose(aligned_base, aligned_changed)


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
        trainer._multi_teacher_fixed_hints = {}

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
        student_prompt = "student prompt"
        hint = trainer_multi._build_hint(_make_state())
        teacher_prompt_len = len(
            trainer_multi.tokenizer.apply_chat_template(
                [{"role": "user", "content": student_prompt + hint}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )

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
            "_student_prompts": [student_prompt, student_prompt],
            "_problem_ids": ["p1", "p1"],
        }

        aligned_logp, teacher_input_ids, teacher_attention_mask, teacher_loss_mask = (
            trainer_multi._compute_privileged_teacher_logp(rollout_batch)
        )

        expected = torch.zeros(2, 17, dtype=torch.float32)
        expected[:, 14:16] = torch.tensor(
            [float(teacher_prompt_len - 1), float(teacher_prompt_len)]
        )
        torch.testing.assert_close(aligned_logp, expected)
        trainer_multi._load_peft_lora_adapter.assert_called_once_with(
            trainer_multi.teacher, "/fake/lora"
        )

    def test_multi_teacher_diverse_best_combined_fixed(self, trainer_multi):
        """Combined diverse-best hints can be fixed per problem across steps."""
        trainer_multi.config.multi_teacher_hint_mode = "diverse_best_combined"
        trainer_multi.config.multi_teacher_hint_k = 2
        trainer_multi.config.multi_teacher_hint_fixed = True

        state1 = _make_state(code="int a();", value=0.9)
        state2 = _make_state(code="int b();", value=0.8)
        teacher_sampler = trainer_multi.teacher_samplers["p1"]
        teacher_sampler.get_hint_states = MagicMock(
            return_value=[("diverse_best_combined", ([state1, state2], []))]
        )

        student_prompt = "student prompt"
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
            "_student_prompts": [student_prompt, student_prompt],
            "_problem_ids": ["p1", "p1"],
        }

        aligned1, *_ = trainer_multi._compute_privileged_teacher_logp(rollout_batch)
        aligned2, *_ = trainer_multi._compute_privileged_teacher_logp(rollout_batch)

        # Fixed hints should be sampled only once per problem.
        assert teacher_sampler.get_hint_states.call_count == 1
        assert aligned1.shape == (2, 17)
        torch.testing.assert_close(aligned1, aligned2)
