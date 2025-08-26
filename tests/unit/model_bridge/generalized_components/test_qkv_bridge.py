"""Unit tests for QKV bridge."""

import torch

from transformer_lens.model_bridge.generalized_components.qkv_bridge import QKVBridge


class TestQKVBridge:
    """Test QKV bridge functionality."""

    def test_q_hook_out_mutation_applied_in_forward_pass(self):
        """Test that mutations made to q_hook_out are applied in the forward pass result."""

        # Create a simple config for testing
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

        # Initialize QKV bridge
        qkv_bridge = QKVBridge(
            name="qkv",
            config=TestConfig(),
        )

        # Set the original component
        mock_linear = MockLinear(in_features=128, out_features=384)  # 3 * 128 for QKV
        qkv_bridge.set_original_component(mock_linear)

        # Create test input
        batch_size, seq_len, d_model = 2, 10, 128
        test_input = torch.randn(batch_size, seq_len, d_model)

        # Run forward pass without hooks to get baseline
        baseline_output = qkv_bridge(test_input)

        # Add a hook to q_hook_out that modifies the output
        q_mutation_applied = False
        q_mutated_value = torch.tensor(999.0)  # Distinct value to track

        def q_hook_fn(q_output, hook):
            nonlocal q_mutation_applied
            q_mutation_applied = True
            # Modify the q output by adding a distinct value
            return q_output + q_mutated_value

        # Add the hook to q_hook_out
        qkv_bridge.q_hook_out.add_hook(q_hook_fn)

        # Run forward pass with hook
        hooked_output = qkv_bridge(test_input)

        # Verify that the hook was called
        assert q_mutation_applied, "q_hook_out hook should have been called"

        # Verify that the output is different from baseline
        assert not torch.allclose(
            baseline_output, hooked_output
        ), "Output with q_hook_out mutation should be different from baseline"

        # Verify that the mutation is actually in the output
        # The QKV bridge separates the output into Q, K, V and then recombines
        # We need to check that the Q portion contains our mutation
        qkv_bridge.qkv_separation_rule = qkv_bridge._create_qkv_separation_rule()
        q_output, k_output, v_output = qkv_bridge.qkv_separation_rule.handle_conversion(
            hooked_output
        )

        # Check that the Q output contains our mutation
        # The mutation should be present in the Q portion of the output
        assert torch.any(
            q_output > q_mutated_value - 1.0
        ), "Q output should contain the mutation from q_hook_out"

    def test_k_hook_out_mutation_applied_in_forward_pass(self):
        """Test that mutations made to k_hook_out are applied in the forward pass result."""

        # Create a simple config for testing
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

        # Initialize QKV bridge
        qkv_bridge = QKVBridge(
            name="qkv",
            config=TestConfig(),
        )

        # Set the original component
        mock_linear = MockLinear(in_features=128, out_features=384)  # 3 * 128 for QKV
        qkv_bridge.set_original_component(mock_linear)

        # Create test input
        batch_size, seq_len, d_model = 2, 10, 128
        test_input = torch.randn(batch_size, seq_len, d_model)

        # Run forward pass without hooks to get baseline
        baseline_output = qkv_bridge(test_input)

        # Add a hook to k_hook_out that modifies the output
        k_mutation_applied = False
        k_mutated_value = torch.tensor(888.0)  # Distinct value to track

        def k_hook_fn(k_output, hook):
            nonlocal k_mutation_applied
            k_mutation_applied = True
            # Modify the k output by adding a distinct value
            return k_output + k_mutated_value

        # Add the hook to k_hook_out
        qkv_bridge.k_hook_out.add_hook(k_hook_fn)

        # Run forward pass with hook
        hooked_output = qkv_bridge(test_input)

        # Verify that the hook was called
        assert k_mutation_applied, "k_hook_out hook should have been called"

        # Verify that the output is different from baseline
        assert not torch.allclose(
            baseline_output, hooked_output
        ), "Output with k_hook_out mutation should be different from baseline"

        # Verify that the mutation is actually in the output
        qkv_bridge.qkv_separation_rule = qkv_bridge._create_qkv_separation_rule()
        q_output, k_output, v_output = qkv_bridge.qkv_separation_rule.handle_conversion(
            hooked_output
        )

        # Check that the K output contains our mutation
        assert torch.any(
            k_output > k_mutated_value - 1.0
        ), "K output should contain the mutation from k_hook_out"

    def test_v_hook_out_mutation_applied_in_forward_pass(self):
        """Test that mutations made to v_hook_out are applied in the forward pass result."""

        # Create a simple config for testing
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

        # Initialize QKV bridge
        qkv_bridge = QKVBridge(
            name="qkv",
            config=TestConfig(),
        )

        # Set the original component
        mock_linear = MockLinear(in_features=128, out_features=384)  # 3 * 128 for QKV
        qkv_bridge.set_original_component(mock_linear)

        # Create test input
        batch_size, seq_len, d_model = 2, 10, 128
        test_input = torch.randn(batch_size, seq_len, d_model)

        # Run forward pass without hooks to get baseline
        baseline_output = qkv_bridge(test_input)

        # Add a hook to v_hook_out that modifies the output
        v_mutation_applied = False
        v_mutated_value = torch.tensor(777.0)  # Distinct value to track

        def v_hook_fn(v_output, hook):
            nonlocal v_mutation_applied
            v_mutation_applied = True
            # Modify the v output by adding a distinct value
            return v_output + v_mutated_value

        # Add the hook to v_hook_out
        qkv_bridge.v_hook_out.add_hook(v_hook_fn)

        # Run forward pass with hook
        hooked_output = qkv_bridge(test_input)

        # Verify that the hook was called
        assert v_mutation_applied, "v_hook_out hook should have been called"

        # Verify that the output is different from baseline
        assert not torch.allclose(
            baseline_output, hooked_output
        ), "Output with v_hook_out mutation should be different from baseline"

        # Verify that the mutation is actually in the output
        qkv_bridge.qkv_separation_rule = qkv_bridge._create_qkv_separation_rule()
        q_output, k_output, v_output = qkv_bridge.qkv_separation_rule.handle_conversion(
            hooked_output
        )

        # Check that the V output contains our mutation
        assert torch.any(
            v_output > v_mutated_value - 1.0
        ), "V output should contain the mutation from v_hook_out"
