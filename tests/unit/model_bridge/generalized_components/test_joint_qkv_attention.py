"""Unit tests for Joint QKV Attention bridge."""

import copy

import torch

from transformer_lens.model_bridge.generalized_components.joint_qkv_attention import (
    JointQKVAttentionBridge,
)


class TestJointQKVAttention:
    """Test JointQKVAttentionBridge functionality."""

    @classmethod
    def _make_additive_mask(cls, boolean_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        min_dtype = torch.finfo(dtype).min
        return torch.where(
            boolean_mask,
            torch.zeros((), dtype=dtype, device=boolean_mask.device),
            torch.full((), min_dtype, dtype=dtype, device=boolean_mask.device),
        )

    @classmethod
    def _make_reconstruct_attention_qkv(cls) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.tensor(
            [
                [
                    [[1.0, 0.0], [0.5, -0.5]],
                    [[0.3, 0.7], [0.2, -0.1]],
                    [[-0.4, 0.6], [0.1, 0.9]],
                ]
            ],
            dtype=torch.float32,
        )
        k = torch.tensor(
            [
                [
                    [[0.9, 0.1], [0.2, -0.3]],
                    [[0.5, 0.4], [0.3, 0.2]],
                    [[-0.2, 0.8], [0.7, 0.1]],
                ]
            ],
            dtype=torch.float32,
        )
        v = torch.tensor(
            [
                [
                    [[0.2, 1.0], [0.1, 0.6]],
                    [[0.4, 0.3], [0.8, 0.2]],
                    [[0.7, 0.5], [0.9, 0.4]],
                ]
            ],
            dtype=torch.float32,
        )
        return q, k, v

    def _assert_non_4d_mask_preserves_causality(
        self,
        bridge,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        q, k, v = self._make_reconstruct_attention_qkv()
        boolean_mask = torch.tensor([[True, True, False]])
        additive_mask = self._make_additive_mask(boolean_mask, q.dtype)
        reconstruct_kwargs = {}
        if position_embeddings is not None:
            reconstruct_kwargs["position_embeddings"] = position_embeddings

        bool_output, bool_pattern = bridge._reconstruct_attention(
            q.clone(),
            k.clone(),
            v.clone(),
            attention_mask=boolean_mask,
            **reconstruct_kwargs,
        )
        additive_output, additive_pattern = bridge._reconstruct_attention(
            q.clone(),
            k.clone(),
            v.clone(),
            attention_mask=additive_mask,
            **reconstruct_kwargs,
        )

        assert torch.allclose(bool_output, additive_output)
        assert torch.allclose(bool_pattern, additive_pattern)
        assert torch.all(bool_pattern[:, :, 0, 1:] == 0)
        assert torch.all(bool_pattern[:, :, 1, 2] == 0)
        assert torch.all(bool_pattern[..., 2] == 0)

    def test_q_hook_out_mutation_applied_in_forward_pass(self):
        """Test that mutations made to q.hook_out are applied in the forward pass result."""

        class TestConfig:
            n_heads = 4
            d_model = 128

        # Create a mock linear layer for testing
        class MockLinear(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
                self.bias = torch.nn.Parameter(torch.randn(out_features))

            def forward(self, input):
                return torch.nn.functional.linear(input, self.weight, self.bias)

        q_transformation = MockLinear(in_features=128, out_features=384)
        k_transformation = MockLinear(in_features=128, out_features=384)
        v_transformation = MockLinear(in_features=128, out_features=384)

        def split_qkv_matrix(_component):
            return q_transformation, k_transformation, v_transformation

        # Create a mock attention layer for testing, doesn't do anything because we're only interested in the QKV components
        class MockAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.split_qkv_matrix = split_qkv_matrix  # We need this so that setting the original component doesn't cause an error

            def forward(self, input):
                return input

        # Initialize JointQKVAttentionBridge
        qkv_bridge = JointQKVAttentionBridge(
            name="qkv",
            config=TestConfig(),
            split_qkv_matrix=split_qkv_matrix,
        )

        # Set the original component for the attention layer
        qkv_bridge.set_original_component(MockAttention())

        # Create test input
        batch_size, seq_len, d_model = 2, 10, 128
        test_input = torch.randn(batch_size, seq_len, d_model)

        # Create a hook that returns the input
        # We need this to ensure that the hook splitting logic in the JointQKVAttentionBridge is getting executed
        def q_hook_id_fn(q_output, hook):
            return q_output

        # Add the hook to q.hook_out
        qkv_bridge.q.hook_out.add_hook(q_hook_id_fn)

        # Run forward pass with identity hook to get baseline
        baseline_output, _ = qkv_bridge(test_input)

        # Remove the identity hook
        qkv_bridge.q.hook_out.remove_hooks()

        # Add a hook to q.hook_out that modifies the output
        q_mutation_applied = False
        q_mutated_value = torch.tensor(999.0)  # Distinct value to track

        def q_hook_fn(q_output, hook):
            nonlocal q_mutation_applied
            q_mutation_applied = True
            # Modify the q output by adding a distinct value
            return q_output + q_mutated_value

        # Add the hook to q.hook_out
        qkv_bridge.q.hook_out.add_hook(q_hook_fn)

        # Run forward pass with hook
        hooked_output, _ = qkv_bridge(test_input)

        # Verify that the hook was called
        assert q_mutation_applied, "q.hook_out hook should have been called"

        # Verify that the output is different from baseline
        assert not torch.allclose(
            baseline_output, hooked_output
        ), "Output with q.hook_out mutation should be different from baseline"

    def test_k_hook_out_mutation_applied_in_forward_pass(self):
        """Test that mutations made to k_hook_out are applied in the forward pass result."""

        class TestConfig:
            n_heads = 4
            d_model = 128

        # Create a mock linear layer for testing
        class MockLinear(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
                self.bias = torch.nn.Parameter(torch.randn(out_features))

            def forward(self, input):
                return torch.nn.functional.linear(input, self.weight, self.bias)

        q_transformation = MockLinear(in_features=128, out_features=384)
        k_transformation = MockLinear(in_features=128, out_features=384)
        v_transformation = MockLinear(in_features=128, out_features=384)

        def split_qkv_matrix(_component):
            return q_transformation, k_transformation, v_transformation

        # Create a mock attention layer for testing, doesn't do anything because we're only interested in the QKV components
        class MockAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.split_qkv_matrix = split_qkv_matrix  # We need this so that setting the original component doesn't cause an error

            def forward(self, input):
                return input

        # Initialize JointQKVAttentionBridge
        qkv_bridge = JointQKVAttentionBridge(
            name="qkv",
            config=TestConfig(),
            split_qkv_matrix=split_qkv_matrix,
        )

        # Set the original component for the attention layer
        qkv_bridge.set_original_component(MockAttention())

        # Create test input
        batch_size, seq_len, d_model = 2, 10, 128
        test_input = torch.randn(batch_size, seq_len, d_model)

        # Create a hook that returns the input
        # We need this to ensure that the hook splitting logic in the JointQKVAttentionBridge is getting executed
        def k_hook_id_fn(k_output, hook):
            return k_output

        # Add the hook to k.hook_out
        qkv_bridge.k.hook_out.add_hook(k_hook_id_fn)

        # Run forward pass with identity hook to get baseline
        baseline_output, _ = qkv_bridge(test_input)

        # Remove the identity hook
        qkv_bridge.k.hook_out.remove_hooks()

        # Add a hook to k.hook_out that modifies the output
        k_mutation_applied = False
        k_mutated_value = torch.tensor(888.0)  # Distinct value to track

        def k_hook_fn(k_output, hook):
            nonlocal k_mutation_applied
            k_mutation_applied = True
            # Modify the k output by adding a distinct value
            return k_output + k_mutated_value

        # Add the hook to k_hook_out
        qkv_bridge.k.hook_out.add_hook(k_hook_fn)

        # Run forward pass with hook
        hooked_output, _ = qkv_bridge(test_input)

        # Verify that the hook was called
        assert k_mutation_applied, "k.hook_out hook should have been called"

        # Verify that the output is different from baseline
        assert not torch.allclose(
            baseline_output, hooked_output
        ), "Output with k.hook_out mutation should be different from baseline"

    def test_v_hook_out_mutation_applied_in_forward_pass(self):
        """Test that mutations made to v_hook_out are applied in the forward pass result."""

        class TestConfig:
            n_heads = 4
            d_model = 128

        # Create a mock linear layer for testing
        class MockLinear(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
                self.bias = torch.nn.Parameter(torch.randn(out_features))

            def forward(self, input):
                return torch.nn.functional.linear(input, self.weight, self.bias)

        q_transformation = MockLinear(in_features=128, out_features=384)
        k_transformation = MockLinear(in_features=128, out_features=384)
        v_transformation = MockLinear(in_features=128, out_features=384)

        def split_qkv_matrix(_component):
            return q_transformation, k_transformation, v_transformation

        # Create a mock attention layer for testing, doesn't do anything because we're only interested in the QKV components
        class MockAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.split_qkv_matrix = split_qkv_matrix  # We need this so that setting the original component doesn't cause an error

            def forward(self, input):
                return input

        # Initialize JointQKVAttentionBridge
        qkv_bridge = JointQKVAttentionBridge(
            name="qkv",
            config=TestConfig(),
            split_qkv_matrix=split_qkv_matrix,
        )

        # Set the original component for the attention layer
        qkv_bridge.set_original_component(MockAttention())

        # Create test input
        batch_size, seq_len, d_model = 2, 10, 128
        test_input = torch.randn(batch_size, seq_len, d_model)

        # Create a hook that returns the input
        # We need this to ensure that the hook splitting logic in the JointQKVAttentionBridge is getting executed
        def v_hook_id_fn(v_output, hook):
            return v_output

        # Add the hook to v.hook_out
        qkv_bridge.v.hook_out.add_hook(v_hook_id_fn)

        # Run forward pass with identity hook to get baseline
        baseline_output, _ = qkv_bridge(test_input)

        # Remove the identity hook
        qkv_bridge.v.hook_out.remove_hooks()

        # Add a hook to v.hook_out that modifies the output
        v_mutation_applied = False
        v_mutated_value = torch.tensor(777.0)  # Distinct value to track

        def v_hook_fn(v_output, hook):
            nonlocal v_mutation_applied
            v_mutation_applied = True
            # Modify the v output by adding a distinct value
            return v_output + v_mutated_value

        # Add the hook to v.hook_out
        qkv_bridge.v.hook_out.add_hook(v_hook_fn)

        # Run forward pass with hook
        hooked_output, _ = qkv_bridge(test_input)

        # Verify that the hook was called
        assert v_mutation_applied, "v_hook_out hook should have been called"

        # Verify that the output is different from baseline
        assert not torch.allclose(
            baseline_output, hooked_output
        ), "Output with v_hook_out mutation should be different from baseline"

    def test_reconstruct_attention_boolean_mask_matches_additive_mask(self):
        """Boolean 4D masks should be equivalent to additive masks.

        This regression test covers the HuggingFace causal-mask path used by
        TransformerBridge. Without the boolean-mask conversion in
        ``_reconstruct_attention()``, boolean masks are added as ``0``/``1``
        and produce substantively different scores and patterns than the equivalent additive
        float mask.
        """

        class TestConfig:
            n_heads = 2
            d_model = 4

        class MockOriginalAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_dropout = torch.nn.Identity()

        bridge = JointQKVAttentionBridge(name="qkv", config=TestConfig())
        bridge.add_module("_original_component", MockOriginalAttention())
        q, k, v = self._make_reconstruct_attention_qkv()
        boolean_mask = torch.tensor(
            [[[[False, False, False], [False, True, False], [False, True, True]]]]
        )
        additive_mask = self._make_additive_mask(boolean_mask, q.dtype)

        bool_output, bool_pattern = bridge._reconstruct_attention(
            q.clone(),
            k.clone(),
            v.clone(),
            attention_mask=boolean_mask,
        )
        additive_output, additive_pattern = bridge._reconstruct_attention(
            q.clone(),
            k.clone(),
            v.clone(),
            attention_mask=additive_mask,
        )

        assert torch.isfinite(bool_output).all()
        assert torch.isfinite(bool_pattern).all()
        assert torch.allclose(bool_output, additive_output)
        assert torch.allclose(bool_pattern, additive_pattern)

    def test_rotary_reconstruct_attention_boolean_mask_matches_additive_mask(self):
        """Rotary joint-QKV attention should treat boolean and additive masks identically."""

        from transformer_lens.model_bridge.generalized_components.joint_qkv_position_embeddings_attention import (
            JointQKVPositionEmbeddingsAttentionBridge,
        )

        class TestConfig:
            n_heads = 2
            d_model = 4

        class MockOriginalAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_dropout = torch.nn.Identity()

        bridge = JointQKVPositionEmbeddingsAttentionBridge(name="qkv", config=TestConfig())
        bridge.add_module("_original_component", MockOriginalAttention())
        q, k, v = self._make_reconstruct_attention_qkv()
        boolean_mask = torch.tensor(
            [[[[False, False, False], [False, True, False], [False, True, True]]]]
        )
        additive_mask = self._make_additive_mask(boolean_mask, q.dtype)
        position_embeddings = (
            torch.ones(1, 3, 2, dtype=torch.float32),
            torch.zeros(1, 3, 2, dtype=torch.float32),
        )

        bool_output, bool_pattern = bridge._reconstruct_attention(
            q.clone(),
            k.clone(),
            v.clone(),
            attention_mask=boolean_mask,
            position_embeddings=position_embeddings,
        )
        additive_output, additive_pattern = bridge._reconstruct_attention(
            q.clone(),
            k.clone(),
            v.clone(),
            attention_mask=additive_mask,
            position_embeddings=position_embeddings,
        )

        assert torch.isfinite(bool_output).all()
        assert torch.isfinite(bool_pattern).all()
        assert torch.allclose(bool_output, additive_output)
        assert torch.allclose(bool_pattern, additive_pattern)

    def test_reconstruct_attention_non_4d_mask_preserves_causality(self):
        """Non-4D masks should still receive the local causal mask in the base bridge."""

        class TestConfig:
            n_heads = 2
            d_model = 4

        class MockOriginalAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_dropout = torch.nn.Identity()

        bridge = JointQKVAttentionBridge(name="qkv", config=TestConfig())
        bridge.add_module("_original_component", MockOriginalAttention())

        self._assert_non_4d_mask_preserves_causality(bridge)

    def test_rotary_reconstruct_attention_non_4d_mask_preserves_causality(self):
        """Rotary joint-QKV attention should match base masking semantics for non-4D masks."""

        from transformer_lens.model_bridge.generalized_components.joint_qkv_position_embeddings_attention import (
            JointQKVPositionEmbeddingsAttentionBridge,
        )

        class TestConfig:
            n_heads = 2
            d_model = 4

        class MockOriginalAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_dropout = torch.nn.Identity()

        bridge = JointQKVPositionEmbeddingsAttentionBridge(name="qkv", config=TestConfig())
        bridge.add_module("_original_component", MockOriginalAttention())
        position_embeddings = (
            torch.ones(1, 3, 2, dtype=torch.float32),
            torch.zeros(1, 3, 2, dtype=torch.float32),
        )

        self._assert_non_4d_mask_preserves_causality(
            bridge,
            position_embeddings=position_embeddings,
        )

    def test_deepcopy_does_not_copy_bound_method_self(self):
        """Deepcopy shares split_qkv_matrix and config instead of copying them."""

        class FakeAdapter:
            def __init__(self):
                self.heavy_data = torch.randn(100, 100)

            def split_qkv(self, component):
                return (torch.nn.Linear(4, 4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))

        class TestConfig:
            n_heads = 2
            d_model = 4

        adapter = FakeAdapter()
        bridge = JointQKVAttentionBridge(
            name="attn",
            config=TestConfig(),
            split_qkv_matrix=adapter.split_qkv,
        )

        clone = copy.deepcopy(bridge)

        assert clone.split_qkv_matrix is bridge.split_qkv_matrix
        assert clone.split_qkv_matrix.__self__ is adapter
        assert clone.config is bridge.config

    def test_deepcopy_produces_independent_hooks(self):
        """Deepcopy produces independent HookPoint and LinearBridge instances."""

        class TestConfig:
            n_heads = 2
            d_model = 4

        bridge = JointQKVAttentionBridge(name="attn", config=TestConfig())
        clone = copy.deepcopy(bridge)

        assert clone.hook_in is not bridge.hook_in
        assert clone.hook_out is not bridge.hook_out
        assert clone.q is not bridge.q
        assert clone.k is not bridge.k
        assert clone.v is not bridge.v
