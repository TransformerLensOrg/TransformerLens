from typing import Literal, Optional, Union
import torch
import fancy_einsum as einsum
from transformer_lens import FactoredMatrix, HookedTransformer

OUTPUT_EMBEDDING = 'unembed.W_U'
VECTOR_TYPES = ['OV', 'w_in', 'w_out']


class SVDInterpreter:

    def __init__(self, model: HookedTransformer):
        self.model = model
        self.cfg = model.cfg
        self.params = {name: param for name, param in model.named_parameters()}

    def get_singular_vectors(self,
                             vector_type: Union[Literal["OV"], Literal["w_in"], Literal["w_out"]],
                             layer_index: int,
                             num_vectors: int = 10,
                             head_index: Optional[int] = None) -> torch.Tensor:

        if head_index is None:
            assert vector_type in [
                "w_in", "w_out"], f"Head index optional only for w_in and w_out, got {vector_type}"

        if vector_type == 'OV':
            matrix = self._get_OV_matrix(layer_index, head_index)
            V = matrix.Vh.T

        elif vector_type == 'w_in':
            matrix = self._get_w_in_matrix(layer_index)
            _, _, V = torch.linalg.svd(matrix)

        elif vector_type == 'w_out':
            matrix = self._get_w_out_matrix(layer_index)
            _, _, V = torch.linalg.svd(matrix)

        else:
            raise ValueError(
                f"Vector type must be in {VECTOR_TYPES}, instead got {vector_type}")

        return self._get_singular_vectors_from_matrix(V, self.params[OUTPUT_EMBEDDING], num_vectors)

    def _get_singular_vectors_from_matrix(self,
                                          V: Union[torch.Tensor, FactoredMatrix],
                                          embedding: torch.Tensor,
                                          num_vectors: int = 10) -> torch.Tensor:

        vectors = []
        for i in range(num_vectors):
            activations = V[i, :].float() @ embedding
            vectors.append(activations)

        vectors = torch.stack(vectors, dim=1).unsqueeze(1)
        assert vectors.shape == (
            self.cfg.d_vocab, 1, num_vectors), f"Vectors shape should be {self.cfg.d_vocab, 1, num_vectors} but got {vectors.shape}"
        return vectors

    def _get_OV_matrix(self, layer_index: int, head_index: int) -> FactoredMatrix:
        assert 0 <= layer_index < self.cfg.n_layers, f"Layer index must be between 0 and {self.cfg.n_layers-1} but got {layer_index}"
        assert 0 <= head_index < self.cfg.n_heads, f"Head index must be between 0 and {self.cfg.n_heads-1} but got {head_index}"

        W_V, W_O = self.params[f"blocks.{layer_index}.attn.W_V"], self.params[f"blocks.{layer_index}.attn.W_O"]
        W_V, W_O = W_V[head_index, :, :], W_O[head_index, :, :]

        return FactoredMatrix(W_V, W_O)

    def _get_w_in_matrix(self, layer_index: int) -> torch.Tensor:
        assert 0 <= layer_index < self.cfg.n_layers, f"Layer index must be between 0 and {self.cfg.n_layers-1} but got {layer_index}"

        w_in = self.params[f"blocks.{layer_index}.mlp.W_in"].T

        if f"blocks.{layer_index}.ln2.w" in self.params:  # If fold_ln == False
            ln_2 = self.params[f"blocks.{layer_index}.ln2.w"]
            return einsum.einsum("out in, in -> out in", w_in, ln_2)

        return w_in

    def _get_w_out_matrix(self, layer_index: int) -> torch.Tensor:
        assert 0 <= layer_index < self.cfg.n_layers, f"Layer index must be between 0 and {self.cfg.n_layers-1} but got {layer_index}"

        return self.params[f"blocks.{layer_index}.mlp.W_out"]