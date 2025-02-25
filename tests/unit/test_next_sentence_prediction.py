from unittest.mock import Mock

import pytest
import torch
from transformers import AutoTokenizer, BertForNextSentencePrediction

from transformer_lens import BertNextSentencePrediction, HookedEncoder


@pytest.fixture
def mock_hooked_encoder():
    mock_encoder = Mock(spec=HookedEncoder)

    mock_encoder.cfg = Mock()
    mock_encoder.cfg.device = "cpu"
    mock_encoder.cfg.n_ctx = 512

    mock_encoder.tokenizer = Mock()

    mock_encodings = {
        "input_ids": torch.tensor([[101, 2034, 102, 2035, 102]]),
        "token_type_ids": torch.tensor([[0, 0, 0, 1, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    mock_encoder.tokenizer.return_value = mock_encodings

    # Mock encoder output
    mock_encoder.encoder_output.return_value = (torch.randn(1, 7, 768), {})  # resid  # cache

    # Mock pooler and NSP head
    mock_encoder.pooler = Mock()
    mock_encoder.pooler.return_value = torch.randn(1, 768)
    mock_encoder.nsp_head = Mock()
    mock_encoder.nsp_head.return_value = torch.tensor([[0.6, 0.4]])

    # Mock run_with_cache
    mock_encoder.run_with_cache = Mock()

    return mock_encoder


@pytest.fixture
def bert_nsp(mock_hooked_encoder):
    return BertNextSentencePrediction(mock_hooked_encoder)


@pytest.fixture
def huggingface_bert():
    return BertForNextSentencePrediction.from_pretrained("bert-base-cased")


@pytest.fixture
def encodings():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    sentence_a = "She went to the grocery store."
    sentence_b = "She bought an apple."
    return tokenizer(sentence_a, sentence_b, return_tensors="pt")


def test_init(mock_hooked_encoder):
    """Test initialization of BertNextSentencePrediction"""
    bert_nsp = BertNextSentencePrediction(mock_hooked_encoder)
    assert bert_nsp.model == mock_hooked_encoder


def test_call_chain(bert_nsp, mock_hooked_encoder):
    """Test that encoder_output, pooler and nsp_head are called with the correct parameters"""
    input_tensor = torch.tensor([[1, 2, 3]])
    token_type_ids = torch.tensor([[0, 0, 1]])
    attention_mask = torch.tensor([[1, 1, 1]])

    # Set up specific mock returns
    mock_resid = torch.randn(1, 3, 768)
    mock_hooked_encoder.encoder_output.return_value = mock_resid

    mock_pooled = torch.randn(1, 768)
    mock_hooked_encoder.pooler.return_value = mock_pooled

    mock_nsp_output = torch.tensor([[0.7, 0.3]])
    mock_hooked_encoder.nsp_head.return_value = mock_nsp_output

    # Call forward
    output = bert_nsp.forward(
        input_tensor, token_type_ids=token_type_ids, one_zero_attention_mask=attention_mask
    )

    # Verify the entire chain of calls
    mock_hooked_encoder.encoder_output.assert_called_once_with(
        input_tensor, token_type_ids, attention_mask
    )
    mock_hooked_encoder.pooler.assert_called_once_with(mock_resid)
    mock_hooked_encoder.nsp_head.assert_called_once_with(mock_pooled)

    # Verify output matches the mock NSP head output
    assert torch.equal(output, mock_nsp_output)


def test_tokenizer_integration(bert_nsp, mock_hooked_encoder):
    """Test that tokenizer is properly integrated and called"""
    input_sentences = ["First sentence.", "Second sentence."]

    mock_hooked_encoder.tokenizer = Mock()

    # Mock tokenizer output
    mock_encodings = {
        "input_ids": torch.tensor([[101, 2034, 102, 2035, 102]]),
        "token_type_ids": torch.tensor([[0, 0, 0, 1, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    mock_hooked_encoder.tokenizer.return_value = mock_encodings

    # Call to_tokens
    tokens, type_ids, mask = bert_nsp.to_tokens(input_sentences)

    # Verify tokenizer was called correctly
    mock_hooked_encoder.tokenizer.assert_called_once_with(
        input_sentences[0],
        input_sentences[1],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=mock_hooked_encoder.cfg.n_ctx,
    )

    # Verify outputs match tokenizer output
    assert torch.equal(tokens, mock_encodings["input_ids"])
    assert torch.equal(type_ids, mock_encodings["token_type_ids"])
    assert torch.equal(mask, mock_encodings["attention_mask"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
def test_device_handling_to_tokens(bert_nsp, mock_hooked_encoder):
    """Test proper device handling in to_tokens"""
    mock_hooked_encoder.cfg.device = "cuda"  # Mock GPU device

    input_data = ["First sentence.", "Second sentence."]

    # Mock tokenizer output
    mock_encodings = {
        "input_ids": torch.tensor([[101, 2034, 102, 2035, 102]]),
        "token_type_ids": torch.tensor([[0, 0, 0, 1, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    mock_hooked_encoder.tokenizer.return_value = mock_encodings

    # Call to_tokens with move_to_device=True
    tokens, type_ids, mask = bert_nsp.to_tokens(input_data, move_to_device=True)

    # Verify each tensor was moved to the correct device
    for tensor in [tokens, type_ids, mask]:
        assert tensor.device.type == mock_hooked_encoder.cfg.device

    # Call with move_to_device=False
    tokens, type_ids, mask = bert_nsp.to_tokens(input_data, move_to_device=False)

    # Verify tensors remained on CPU
    for tensor in [tokens, type_ids, mask]:
        assert tensor.device.type == "cpu"


def test_output_for_prediction_return_type(bert_nsp, mock_hooked_encoder):
    """Test that output for return_type='predictions' is correct based on NSP head output"""
    input_data = ["First sentence.", "Second sentence."]

    # Test case 1: Sequential prediction
    mock_hooked_encoder.nsp_head.return_value = torch.tensor([[0.9, 0.1]])
    pred = bert_nsp.forward(input_data, return_type="predictions")
    assert pred == "The sentences are sequential"

    # Test case 2: Non-sequential prediction
    mock_hooked_encoder.nsp_head.return_value = torch.tensor([[0.2, 0.8]])
    pred = bert_nsp.forward(input_data, return_type="predictions")
    assert pred == "The sentences are NOT sequential"


@pytest.mark.parametrize("input_type", ["str_list", "tensor"])
def test_forward_input_types(bert_nsp, encodings, input_type):
    """Test forward pass with different input types"""
    sentence_a = "She went to the grocery store."
    sentence_b = "She bought an apple."
    if input_type == "str_list":
        output = bert_nsp([sentence_a, sentence_b])
    else:
        output = bert_nsp.forward(
            encodings.input_ids,
            token_type_ids=encodings.token_type_ids,
            one_zero_attention_mask=encodings.attention_mask,
        )

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 2)  # Batch size 1, binary classification


def test_forward_tokens_as_input_without_token_type_ids_error(bert_nsp):
    """Test that forward raises error when tokens are input directly but no token_type_ids are provided"""
    with pytest.raises(
        ValueError, match="You are using the NSP task without specifying token_type_ids"
    ):
        bert_nsp.forward(torch.tensor([[1, 2, 768]]))


@pytest.mark.parametrize("return_type", [None, "logits", "predictions"])
def test_forward_return_types(bert_nsp, mock_hooked_encoder, return_type):
    """Test different return types from forward pass"""
    # Setup mock logits that favor sequential prediction
    mock_logits = torch.tensor([[0.7, 0.3]])
    mock_hooked_encoder.nsp_head.return_value = mock_logits

    input_data = ["She went to the grocery store.", "She bought an apple."]
    output = bert_nsp.forward(input_data, return_type=return_type)

    if return_type is None:
        assert output is None
    elif return_type == "logits":
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 2)
        assert torch.equal(output, mock_logits)
    elif return_type == "predictions":
        assert isinstance(output, str)
        assert output == "The sentences are sequential"  # Based on mock logits


def test_to_tokens_validation(bert_nsp):
    """Test input validation in to_tokens method"""
    with pytest.raises(
        ValueError, match="Next sentence prediction task requires exactly two sentences"
    ):
        bert_nsp.to_tokens(["Single sentence"])

    with pytest.raises(
        ValueError, match="Next sentence prediction task requires exactly two sentences"
    ):
        bert_nsp.to_tokens(["One", "Two", "Three"])


def test_run_with_cache(bert_nsp, mock_hooked_encoder):
    """Test run_with_cache for correct handling of cache"""
    # Set up mock returns
    mock_resid = torch.randn(1, 3, 768)
    mock_cache = {"resid_pre": torch.randn(1, 3, 768), "attn_output": torch.randn(1, 3, 768)}
    mock_hooked_encoder.run_with_cache.return_value = (
        torch.tensor([[0.6, 0.4]]),  # Mock logits
        mock_cache,
    )

    input_data = ["First sentence.", "Second sentence."]

    # Run with cache
    output, cache = bert_nsp.run_with_cache(
        input_data, return_type="logits", return_cache_object=True
    )

    # Verify output shape and values
    assert output.shape == (1, 2)
    assert isinstance(cache, dict) or hasattr(cache, "cache_dict")

    # Verify the cache contains expected keys
    if hasattr(cache, "cache_dict"):
        cache_dict = cache.cache_dict
    else:
        cache_dict = cache

    assert "resid_pre" in cache_dict
    assert "attn_output" in cache_dict


def test_return_type_consistency(bert_nsp, mock_hooked_encoder):
    """Test consistency between logits and prediction outputs"""
    # Setup mock logits that favor non-sequential prediction
    mock_logits = torch.tensor([[0.2, 0.8]])
    mock_hooked_encoder.nsp_head.return_value = mock_logits

    input_data = ["She went to the grocery store.", "She bought an apple."]

    # Get predictions using different return types
    logits = bert_nsp.forward(input_data, return_type="logits")
    prediction_str = bert_nsp.forward(input_data, return_type="predictions")

    # Calculate predicted class from logits
    predicted_class = logits.argmax(dim=-1).item()
    expected_prediction = ["The sentences are sequential", "The sentences are NOT sequential"][
        predicted_class
    ]

    assert prediction_str == expected_prediction  # Based on mock logits
