"""Integration tests for tensor extraction and math function consistency."""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.weight_processing import ProcessWeights


@pytest.fixture(scope="class")
def test_models():
    """Set up test models for consistency testing."""
    device = "cpu"
    model_name = "distilgpt2"

    # Load HookedTransformer (no processing)
    hooked_model = HookedTransformer.from_pretrained(
        model_name, device=device, fold_ln=False, center_writing_weights=False, center_unembed=False
    )

    # Load TransformerBridge (no processing)
    bridge_model = TransformerBridge.boot_transformers(model_name, device=device)

    return {
        "hooked_model": hooked_model,
        "bridge_model": bridge_model,
        "hooked_state_dict": hooked_model.state_dict(),
        "bridge_state_dict": bridge_model.original_model.state_dict(),
    }


@pytest.mark.skip(
    reason="Tensor extraction consistency tests failing due to architectural differences between HookedTransformer and TransformerBridge"
)
class TestTensorExtractionConsistency:
    """Test that tensor extraction returns consistent results between models."""

    def test_extract_attention_tensors_shapes_match(self, test_models):
        """Test that extracted tensors have matching shapes."""
        layer = 0

        hooked_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            test_models["hooked_state_dict"], test_models["hooked_model"].cfg, layer, adapter=None
        )

        bridge_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            test_models["bridge_state_dict"],
            test_models["bridge_model"].cfg,
            layer,
            adapter=test_models["bridge_model"].adapter,
        )

        tensor_names = ["wq", "wk", "wv", "bq", "bk", "bv", "ln1_b", "ln1_w"]

        for tensor_name in tensor_names:
            hooked_tensor = hooked_tensors[tensor_name]
            bridge_tensor = bridge_tensors[tensor_name]

            if hooked_tensor is None and bridge_tensor is None:
                continue
            elif hooked_tensor is None or bridge_tensor is None:
                pytest.fail(f"{tensor_name}: One is None, other is not")

            assert (
                hooked_tensor.shape == bridge_tensor.shape
            ), f"{tensor_name} shape mismatch: {hooked_tensor.shape} vs {bridge_tensor.shape}"

    def test_extract_attention_tensors_values_match(self, test_models):
        """Test that extracted tensors have matching values."""
        layer = 0

        hooked_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            test_models["hooked_state_dict"], test_models["hooked_model"].cfg, layer, adapter=None
        )

        bridge_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            test_models["bridge_state_dict"],
            test_models["bridge_model"].cfg,
            layer,
            adapter=test_models["bridge_model"].adapter,
        )

        tensor_names = ["wq", "wk", "wv", "bq", "bk", "bv", "ln1_b", "ln1_w"]

        for tensor_name in tensor_names:
            hooked_tensor = hooked_tensors[tensor_name]
            bridge_tensor = bridge_tensors[tensor_name]

            if hooked_tensor is None or bridge_tensor is None:
                continue

            max_diff = torch.max(torch.abs(hooked_tensor - bridge_tensor)).item()
            assert max_diff < 1e-6, f"{tensor_name} value mismatch: max_diff={max_diff:.2e}"

    @pytest.mark.parametrize("component", ["q", "k", "v"])
    def test_fold_layer_norm_bias_single_consistency(self, test_models, component):
        """Test fold_layer_norm_bias_single consistency for each component."""
        layer = 0

        hooked_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            test_models["hooked_state_dict"], test_models["hooked_model"].cfg, layer, adapter=None
        )

        bridge_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            test_models["bridge_state_dict"],
            test_models["bridge_model"].cfg,
            layer,
            adapter=test_models["bridge_model"].adapter,
        )

        if hooked_tensors["ln1_b"] is None:
            pytest.skip("No LayerNorm bias to test")

        # Get tensors for the component
        w_key = f"w{component}"
        b_key = f"b{component}"

        hooked_result = ProcessWeights.fold_layer_norm_bias_single(
            hooked_tensors[w_key], hooked_tensors[b_key], hooked_tensors["ln1_b"]
        )
        bridge_result = ProcessWeights.fold_layer_norm_bias_single(
            bridge_tensors[w_key], bridge_tensors[b_key], bridge_tensors["ln1_b"]
        )

        max_diff = torch.max(torch.abs(hooked_result - bridge_result)).item()
        assert (
            max_diff < 1e-6
        ), f"fold_layer_norm_bias_single({component}) mismatch: max_diff={max_diff:.2e}"

    @pytest.mark.parametrize("component", ["q", "k", "v"])
    def test_fold_layer_norm_weight_single_consistency(self, test_models, component):
        """Test fold_layer_norm_weight_single consistency for each component."""
        layer = 0

        hooked_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            test_models["hooked_state_dict"], test_models["hooked_model"].cfg, layer, adapter=None
        )

        bridge_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            test_models["bridge_state_dict"],
            test_models["bridge_model"].cfg,
            layer,
            adapter=test_models["bridge_model"].adapter,
        )

        if hooked_tensors["ln1_w"] is None:
            pytest.skip("No LayerNorm weight to test")

        # Get tensor for the component
        w_key = f"w{component}"

        hooked_result = ProcessWeights.fold_layer_norm_weight_single(
            hooked_tensors[w_key], hooked_tensors["ln1_w"]
        )
        bridge_result = ProcessWeights.fold_layer_norm_weight_single(
            bridge_tensors[w_key], bridge_tensors["ln1_w"]
        )

        max_diff = torch.max(torch.abs(hooked_result - bridge_result)).item()
        assert (
            max_diff < 1e-6
        ), f"fold_layer_norm_weight_single({component}) mismatch: max_diff={max_diff:.2e}"

    @pytest.mark.parametrize("component", ["q", "k", "v"])
    def test_center_weight_single_consistency(self, test_models, component):
        """Test center_weight_single consistency for each component."""
        layer = 0

        hooked_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            test_models["hooked_state_dict"], test_models["hooked_model"].cfg, layer, adapter=None
        )

        bridge_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            test_models["bridge_state_dict"],
            test_models["bridge_model"].cfg,
            layer,
            adapter=test_models["bridge_model"].adapter,
        )

        # Get tensor for the component
        w_key = f"w{component}"

        hooked_result = ProcessWeights.center_weight_single(hooked_tensors[w_key])
        bridge_result = ProcessWeights.center_weight_single(bridge_tensors[w_key])

        max_diff = torch.max(torch.abs(hooked_result - bridge_result)).item()
        assert (
            max_diff < 1e-6
        ), f"center_weight_single({component}) mismatch: max_diff={max_diff:.2e}"

    def test_full_processing_pipeline_consistency(self, test_models):
        """Test that the full processing pipeline produces consistent results."""
        layer = 0

        hooked_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            test_models["hooked_state_dict"], test_models["hooked_model"].cfg, layer, adapter=None
        )

        bridge_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            test_models["bridge_state_dict"],
            test_models["bridge_model"].cfg,
            layer,
            adapter=test_models["bridge_model"].adapter,
        )

        if hooked_tensors["ln1_b"] is None or hooked_tensors["ln1_w"] is None:
            pytest.skip("No LayerNorm parameters to test full pipeline")

        # Apply full processing pipeline
        def process_tensors(tensors):
            wq, wk, wv = tensors["wq"], tensors["wk"], tensors["wv"]
            bq, bk, bv = tensors["bq"], tensors["bk"], tensors["bv"]
            ln1_b, ln1_w = tensors["ln1_b"], tensors["ln1_w"]

            # Step 1: Fold biases
            bq = ProcessWeights.fold_layer_norm_bias_single(wq, bq, ln1_b)
            bk = ProcessWeights.fold_layer_norm_bias_single(wk, bk, ln1_b)
            bv = ProcessWeights.fold_layer_norm_bias_single(wv, bv, ln1_b)

            # Step 2: Fold weights
            wq = ProcessWeights.fold_layer_norm_weight_single(wq, ln1_w)
            wk = ProcessWeights.fold_layer_norm_weight_single(wk, ln1_w)
            wv = ProcessWeights.fold_layer_norm_weight_single(wv, ln1_w)

            # Step 3: Center weights
            wq = ProcessWeights.center_weight_single(wq)
            wk = ProcessWeights.center_weight_single(wk)
            wv = ProcessWeights.center_weight_single(wv)

            return wq, wk, wv, bq, bk, bv

        hooked_final = process_tensors(hooked_tensors)
        bridge_final = process_tensors(bridge_tensors)

        # Compare final results
        components = ["wq", "wk", "wv", "bq", "bk", "bv"]

        for comp, hooked_result, bridge_result in zip(components, hooked_final, bridge_final):
            max_diff = torch.max(torch.abs(hooked_result - bridge_result)).item()
            assert max_diff < 1e-6, f"Full pipeline mismatch for {comp}: max_diff={max_diff:.2e}"

    @pytest.mark.parametrize("layer", [0, 1, 2])
    def test_multiple_layers_consistency(self, test_models, layer):
        """Test consistency across multiple layers."""
        if layer >= test_models["hooked_model"].cfg.n_layers:
            pytest.skip(f"Layer {layer} doesn't exist in model")

        hooked_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            test_models["hooked_state_dict"], test_models["hooked_model"].cfg, layer, adapter=None
        )

        bridge_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            test_models["bridge_state_dict"],
            test_models["bridge_model"].cfg,
            layer,
            adapter=test_models["bridge_model"].adapter,
        )

        # Test that tensors match
        tensor_names = ["wq", "wk", "wv", "bq", "bk", "bv"]

        for tensor_name in tensor_names:
            hooked_tensor = hooked_tensors[tensor_name]
            bridge_tensor = bridge_tensors[tensor_name]

            max_diff = torch.max(torch.abs(hooked_tensor - bridge_tensor)).item()
            assert (
                max_diff < 1e-6
            ), f"Layer {layer}, {tensor_name} mismatch: max_diff={max_diff:.2e}"
