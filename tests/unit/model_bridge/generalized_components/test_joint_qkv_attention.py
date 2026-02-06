"""Unit tests for Joint QKV Attention bridge."""

import torch

from transformer_lens.model_bridge.generalized_components.joint_qkv_attention import (
    JointQKVAttentionBridge,
)


class TestJointQKVAttention:
    """Test JointQKVAttentionBridge functionality."""

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

        split_qkv_matrix = lambda x: (q_transformation, k_transformation, v_transformation)

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

        split_qkv_matrix = lambda x: (q_transformation, k_transformation, v_transformation)

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

        split_qkv_matrix = lambda x: (q_transformation, k_transformation, v_transformation)

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
