"""SVD Interpreter.

Module for getting the singular vectors of the OV, w_in, and w_out matrices of a
:class:`transformer_lens.HookedTransformer`.
"""

from typing import Optional, Union

import fancy_einsum as einsum
import torch
from typeguard import typechecked
from typing_extensions import Literal

from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.HookedTransformer import HookedTransformer

OUTPUT_EMBEDDING = "unembed.W_U"
VECTOR_TYPES = ["OV", "w_in", "w_out"]


class SVDInterpreter:
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.cfg = model.cfg
        self.params = {name: param for name, param in model.named_parameters()}

    @typechecked
    def get_singular_vectors(
        self,
        vector_type: Union[Literal["OV"], Literal["w_in"], Literal["w_out"]],
        layer_index: int,
        num_vectors: int = 10,
        head_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Gets the singular vectors for a given vector type, layer, and optionally head.

        This tensor can then be plotted using Neel's PySvelte, as demonstrated in the demo for this
        feature. The demo also points out some "gotchas" in this feature - numerical instability
        means inconsistency across devices, and the default HookedTransformer parameters don't
        replicate the original SVD post very well. So I'd recommend checking out the demo if you
        want to use this!

        Example:

        .. code-block:: python

            from transformer_lens import HookedTransformer, SVDInterpreter

            model = HookedTransformer.from_pretrained('gpt2-medium')
            svd_interpreter = SVDInterpreter(model)

            ov = svd_interpreter.get_singular_vectors('OV', layer_index=22, head_index=10)

            all_tokens = [model.to_str_tokens(np.array([i])) for i in range(model.cfg.d_vocab)]
            all_tokens = [all_tokens[i][0] for i in range(model.cfg.d_vocab)]

            def plot_matrix(matrix, tokens, k=10, filter="topk"):
                pysvelte.TopKTable(
                    tokens=all_tokens,
                    activations=matrix,
                    obj_type="SVD direction",
                    k=k,
                    filter=filter
                ).show()

            plot_matrix(ov, all_tokens)

        Args:
            vector_type: Type of the vector:
                - "OV": Singular vectors of the OV matrix for a particular layer and head.
                - "w_in": Singular vectors of the w_in matrix for a particular layer.
                - "w_out": Singular vectors of the w_out matrix for a particular layer.
            layer_index: The index of the layer.
            num_vectors: Number of vectors.
            head_index: Index of the head.
        """

        if head_index is None:
            assert vector_type in [
                "w_in",
                "w_out",
            ], f"Head index optional only for w_in and w_out, got {vector_type}"

        matrix: Union[FactoredMatrix, torch.Tensor]
        if vector_type == "OV":
            assert head_index is not None  # keep mypy happy
            matrix = self._get_OV_matrix(layer_index, head_index)
            V = matrix.Vh.T

        elif vector_type == "w_in":
            matrix = self._get_w_in_matrix(layer_index)
            _, _, V = torch.linalg.svd(matrix)

        elif vector_type == "w_out":
            matrix = self._get_w_out_matrix(layer_index)
            _, _, V = torch.linalg.svd(matrix)

        else:
            raise ValueError(f"Vector type must be in {VECTOR_TYPES}, instead got {vector_type}")

        return self._get_singular_vectors_from_matrix(V, self.params[OUTPUT_EMBEDDING], num_vectors)

    def _get_singular_vectors_from_matrix(
        self,
        V: Union[torch.Tensor, FactoredMatrix],
        embedding: torch.Tensor,
        num_vectors: int = 10,
    ) -> torch.Tensor:
        """Returns the top num_vectors singular vectors from a matrix."""

        vectors_list = []
        for i in range(num_vectors):
            activations = V[i, :].float() @ embedding  # type: ignore
            vectors_list.append(activations)

        vectors = torch.stack(vectors_list, dim=1).unsqueeze(1)
        assert vectors.shape == (
            self.cfg.d_vocab,
            1,
            num_vectors,
        ), f"Vectors shape should be {self.cfg.d_vocab, 1, num_vectors} but got {vectors.shape}"
        return vectors

    def _get_OV_matrix(self, layer_index: int, head_index: int) -> FactoredMatrix:
        """Gets the OV matrix for a particular layer and head."""

        assert (
            0 <= layer_index < self.cfg.n_layers
        ), f"Layer index must be between 0 and {self.cfg.n_layers-1} but got {layer_index}"
        assert (
            0 <= head_index < self.cfg.n_heads
        ), f"Head index must be between 0 and {self.cfg.n_heads-1} but got {head_index}"

        W_V: torch.Tensor = self.params[f"blocks.{layer_index}.attn.W_V"]
        W_O: torch.Tensor = self.params[f"blocks.{layer_index}.attn.W_O"]
        W_V, W_O = W_V[head_index, :, :], W_O[head_index, :, :]

        return FactoredMatrix(W_V, W_O)

    def _get_w_in_matrix(self, layer_index: int) -> torch.Tensor:
        """Gets the w_in matrix for a particular layer."""

        assert (
            0 <= layer_index < self.cfg.n_layers
        ), f"Layer index must be between 0 and {self.cfg.n_layers-1} but got {layer_index}"

        w_in = self.params[f"blocks.{layer_index}.mlp.W_in"].T

        if f"blocks.{layer_index}.ln2.w" in self.params:  # If fold_ln == False
            ln_2 = self.params[f"blocks.{layer_index}.ln2.w"]
            return einsum.einsum("out in, in -> out in", w_in, ln_2)

        return w_in

    def _get_w_out_matrix(self, layer_index: int) -> torch.Tensor:
        """Gets the w_out matrix for a particular layer."""

        assert (
            0 <= layer_index < self.cfg.n_layers
        ), f"Layer index must be between 0 and {self.cfg.n_layers-1} but got {layer_index}"

        return self.params[f"blocks.{layer_index}.mlp.W_out"]
