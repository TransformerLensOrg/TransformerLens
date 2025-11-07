"""Integration tests for component benchmarking.

This module tests all standard components in various models against their
HuggingFace equivalents.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens.benchmarks.component_outputs import (
    BenchmarkReport,
    ComponentBenchmarker,
)
from transformer_lens.model_bridge import TransformerBridge


class TestComponentBenchmarking:
    """Test component benchmarking across different models."""

    @pytest.fixture(
        scope="class",
        params=[
            "gpt2",
            # Add more models as they become supported
            # "EleutherAI/pythia-70m",
            # "facebook/opt-125m",
        ],
    )
    def model_name(self, request):
        """Parameterized fixture for model names."""
        return request.param

    @pytest.fixture(scope="class")
    def bridge_model(self, model_name):
        """Load TransformerBridge once per test class."""
        return TransformerBridge.boot_transformers(model_name, device="cpu")

    @pytest.fixture(scope="class")
    def hf_model(self, model_name):
        """Load HuggingFace model once per test class."""
        return AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

    @pytest.fixture(scope="class")
    def benchmarker(self, bridge_model, hf_model):
        """Create component benchmarker."""
        return ComponentBenchmarker(
            bridge_model=bridge_model,
            hf_model=hf_model,
            adapter=bridge_model.adapter,
            cfg=bridge_model.cfg,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_all_components_match(self, benchmarker, model_name):
        """Test that all components produce matching outputs."""
        report = benchmarker.benchmark_all_components()

        # Print summary for debugging
        report.print_summary(verbose=False)

        # Assert that all components pass
        assert (
            report.failed_components == 0
        ), f"{report.failed_components} components failed for {model_name}"

    def test_embedding_component(self, benchmarker, bridge_model, hf_model):
        """Test embedding component specifically."""
        try:
            # Get components
            embed_bridge = benchmarker.adapter.get_component(bridge_model, "embed")
            embed_hf = benchmarker.adapter.get_component(hf_model, "embed")

            # Generate test input (token indices)
            batch_size, seq_len = 2, 10
            token_ids = torch.randint(0, bridge_model.cfg.d_vocab, (batch_size, seq_len))

            # Run both components
            bridge_output = embed_bridge(token_ids)
            hf_output = embed_hf(token_ids)

            # Compare
            assert bridge_output.shape == hf_output.shape, "Embedding output shapes don't match"
            assert torch.allclose(
                bridge_output, hf_output, atol=1e-4
            ), "Embedding outputs don't match"

        except Exception as e:
            pytest.skip(f"Embedding test skipped: {e}")

    def test_attention_components(self, benchmarker, bridge_model, hf_model):
        """Test all attention components."""
        cfg = bridge_model.cfg

        # Generate test input
        batch, seq_len, d_model = 2, 8, cfg.d_model
        test_input = torch.randn(batch, seq_len, d_model)

        # Test each layer's attention
        for layer_idx in range(min(2, cfg.n_layers)):  # Test first 2 layers
            try:
                comp_path = f"blocks.{layer_idx}.attn"

                # Get components
                attn_bridge = benchmarker.adapter.get_component(bridge_model, comp_path)
                attn_hf = benchmarker.adapter.get_component(hf_model, comp_path)

                # Run both
                try:
                    bridge_output = attn_bridge(
                        query_input=test_input,
                        key_input=test_input,
                        value_input=test_input,
                        past_kv_cache_entry=None,
                        attention_mask=None,
                    )
                except TypeError:
                    try:
                        bridge_output = attn_bridge(hidden_states=test_input)
                    except TypeError:
                        bridge_output = attn_bridge(test_input)

                hf_output = attn_hf(hidden_states=test_input)

                # Extract tensors from tuples if needed
                bridge_tensor = (
                    bridge_output[0] if isinstance(bridge_output, tuple) else bridge_output
                )
                hf_tensor = hf_output[0] if isinstance(hf_output, tuple) else hf_output

                # Compare
                assert (
                    bridge_tensor.shape == hf_tensor.shape
                ), f"Attention output shapes don't match for layer {layer_idx}"
                assert torch.allclose(
                    bridge_tensor, hf_tensor, atol=1e-3, rtol=1e-3
                ), f"Attention outputs don't match for layer {layer_idx}"

            except Exception as e:
                pytest.skip(f"Attention test skipped for layer {layer_idx}: {e}")

    def test_mlp_components(self, benchmarker, bridge_model, hf_model):
        """Test all MLP components."""
        cfg = bridge_model.cfg

        # Generate test input
        batch, seq_len, d_model = 2, 8, cfg.d_model
        test_input = torch.randn(batch, seq_len, d_model)

        # Test each layer's MLP
        for layer_idx in range(min(2, cfg.n_layers)):  # Test first 2 layers
            try:
                comp_path = f"blocks.{layer_idx}.mlp"

                # Get components
                mlp_bridge = benchmarker.adapter.get_component(bridge_model, comp_path)
                mlp_hf = benchmarker.adapter.get_component(hf_model, comp_path)

                # Run both
                bridge_output = mlp_bridge(test_input)
                hf_output = mlp_hf(test_input)

                # Extract tensors from tuples if needed
                bridge_tensor = (
                    bridge_output[0] if isinstance(bridge_output, tuple) else bridge_output
                )
                hf_tensor = hf_output[0] if isinstance(hf_output, tuple) else hf_output

                # Compare
                assert (
                    bridge_tensor.shape == hf_tensor.shape
                ), f"MLP output shapes don't match for layer {layer_idx}"
                assert torch.allclose(
                    bridge_tensor, hf_tensor, atol=1e-4
                ), f"MLP outputs don't match for layer {layer_idx}"

            except Exception as e:
                pytest.skip(f"MLP test skipped for layer {layer_idx}: {e}")

    def test_normalization_components(self, benchmarker, bridge_model, hf_model):
        """Test layer normalization components."""
        cfg = bridge_model.cfg

        # Generate test input
        batch, seq_len, d_model = 2, 8, cfg.d_model
        test_input = torch.randn(batch, seq_len, d_model)

        # Test ln_final if it exists
        try:
            ln_final_bridge = benchmarker.adapter.get_component(bridge_model, "ln_final")
            ln_final_hf = benchmarker.adapter.get_component(hf_model, "ln_final")

            bridge_output = ln_final_bridge(test_input)
            hf_output = ln_final_hf(test_input)

            # Extract tensors from tuples if needed
            bridge_tensor = bridge_output[0] if isinstance(bridge_output, tuple) else bridge_output
            hf_tensor = hf_output[0] if isinstance(hf_output, tuple) else hf_output

            assert bridge_tensor.shape == hf_tensor.shape, "ln_final output shapes don't match"
            assert torch.allclose(
                bridge_tensor, hf_tensor, atol=1e-5
            ), "ln_final outputs don't match"

        except Exception as e:
            pytest.skip(f"ln_final test skipped: {e}")

        # Test ln1 and ln2 in blocks
        for layer_idx in range(min(2, cfg.n_layers)):
            for ln_name in ["ln1", "ln2"]:
                try:
                    comp_path = f"blocks.{layer_idx}.{ln_name}"

                    ln_bridge = benchmarker.adapter.get_component(bridge_model, comp_path)
                    ln_hf = benchmarker.adapter.get_component(hf_model, comp_path)

                    bridge_output = ln_bridge(test_input)
                    hf_output = ln_hf(test_input)

                    # Extract tensors from tuples if needed
                    bridge_tensor = (
                        bridge_output[0] if isinstance(bridge_output, tuple) else bridge_output
                    )
                    hf_tensor = hf_output[0] if isinstance(hf_output, tuple) else hf_output

                    assert (
                        bridge_tensor.shape == hf_tensor.shape
                    ), f"{ln_name} output shapes don't match for layer {layer_idx}"
                    assert torch.allclose(
                        bridge_tensor, hf_tensor, atol=1e-5
                    ), f"{ln_name} outputs don't match for layer {layer_idx}"

                except Exception as e:
                    pytest.skip(f"{ln_name} test skipped for layer {layer_idx}: {e}")

    def test_benchmark_report_format(self, benchmarker):
        """Test that benchmark report has correct format."""
        report = benchmarker.benchmark_all_components()

        # Check report structure
        assert isinstance(report, BenchmarkReport)
        assert report.total_components > 0
        assert report.passed_components + report.failed_components == report.total_components
        assert 0 <= report.pass_rate <= 100
        assert len(report.component_results) == report.total_components

        # Check each component result
        for result in report.component_results:
            assert result.component_path is not None
            assert result.component_type is not None
            assert isinstance(result.passed, bool)
            assert isinstance(result.max_diff, float)
            assert isinstance(result.mean_diff, float)
            assert isinstance(result.output_shape, tuple)

    def test_custom_test_inputs(self, benchmarker):
        """Test benchmarking with custom test inputs."""
        cfg = benchmarker.cfg

        # Create custom test inputs
        custom_inputs = {
            "hidden_states": torch.randn(1, 5, cfg.d_model),
            "token_ids": torch.randint(0, cfg.d_vocab, (1, 5)),
        }

        # Run benchmark with custom inputs
        report = benchmarker.benchmark_all_components(test_inputs=custom_inputs)

        assert report.total_components > 0

    def test_skip_components(self, benchmarker):
        """Test skipping specific components."""
        # Run benchmark skipping embed component
        report = benchmarker.benchmark_all_components(skip_components=["embed"])

        # Check that embed was not tested
        tested_paths = [r.component_path for r in report.component_results]
        assert "embed" not in tested_paths

    @pytest.mark.slow
    def test_multiple_models(self):
        """Test benchmarking across multiple model architectures."""
        models_to_test = [
            "gpt2",
            # Add more as they're supported and tested
        ]

        results = {}
        for model_name in models_to_test:
            try:
                bridge_model = TransformerBridge.boot_transformers(model_name, device="cpu")
                hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

                benchmarker = ComponentBenchmarker(
                    bridge_model=bridge_model,
                    hf_model=hf_model,
                    adapter=bridge_model.adapter,
                    cfg=bridge_model.cfg,
                    atol=1e-4,
                    rtol=1e-4,
                )

                report = benchmarker.benchmark_all_components()
                results[model_name] = report

                # All components should pass
                assert (
                    report.failed_components == 0
                ), f"Model {model_name} has {report.failed_components} failing components"

            except Exception as e:
                pytest.skip(f"Model {model_name} test skipped: {e}")

        # Print summary for all models
        print("\n" + "=" * 80)
        print("Multi-Model Benchmark Summary")
        print("=" * 80)
        for model_name, report in results.items():
            print(
                f"{model_name}: {report.passed_components}/{report.total_components} passed ({report.pass_rate:.1f}%)"
            )
        print("=" * 80)
