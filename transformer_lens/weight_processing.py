#!/usr/bin/env python3
"""
Weight Processing Functions for Transformer Models.

This module contains all the weight processing functions extracted from HookedTransformer,
organized into a single ProcessWeights class with static methods. These functions are used
to modify transformer model weights for better interpretability and analysis.
"""

from typing import Dict

import einops
import torch

import transformer_lens.utils as utils
from transformer_lens.FactoredMatrix import FactoredMatrix


class ProcessWeights:
    """
    A collection of static methods for processing transformer model weights.

    These methods are extracted from HookedTransformer and provide various weight
    transformations for improved model interpretability:
    - LayerNorm folding: Merges LayerNorm parameters into subsequent linear layers
    - Weight centering: Centers weights that write to the residual stream
    - Unembed centering: Centers unembedding weights (translation invariant)
    - Value bias folding: Consolidates value biases into output biases
    - Attention matrix refactoring: Experimental QK/OV matrix factorization

    When an architecture adapter is provided, the methods will translate TransformerLens
    parameter names to the target format (e.g., HuggingFace) for processing.
    """

    @staticmethod
    def _get_param_key(tl_key: str, adapter=None) -> str:
        """Get the actual parameter key to use, translating via adapter if provided.

        Args:
            tl_key: TransformerLens format parameter key
            adapter: Optional architecture adapter for key translation

        Returns:
            The key to use for accessing parameters in the state dict
        """
        if adapter is None:
            return tl_key

        # Use the adapter to translate from TL format to target format
        return adapter.translate_transformer_lens_path(tl_key)

    @staticmethod
    def fold_layer_norm(
        state_dict: Dict[str, torch.Tensor],
        cfg,
        fold_biases: bool = True,
        center_weights: bool = True,
        adapter=None,
    ) -> Dict[str, torch.Tensor]:
        """Fold Layer Norm. Can also be used to fold RMS Norm, when fold_biases and center_weights are set to False.

        Takes in a state dict from a pretrained model, formatted to be consistent with
        HookedTransformer but with LayerNorm weights and biases. Folds these into the neighbouring
        weights. See further_comments.md for more details.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of pretrained model.
            cfg: Model configuration object with n_layers, n_key_value_heads, etc.
            fold_biases (bool): Enables folding of LN biases. Should be disabled when RMS Norm is used.
            center_weights (bool): Enables the centering of weights after folding in LN. Should be disabled when RMS Norm is used.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with LayerNorm folded into linear layers.
        """
        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        # Models that use Grouped Query Attention (Only Mistral at the time of writing) prefix their K/V weights and
        # biases with an underscore in order to distinguish them, but folding the LN into them still works the same,
        # so we just add the underscore if GQA is used (i.e. if `cfg.n_key_value_heads is specified`).
        gqa = "" if getattr(cfg, "n_key_value_heads", None) is None else "_"

        for l in range(cfg.n_layers):
            # Get translated parameter keys
            b_Q_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_Q", adapter)
            W_Q_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_Q", adapter)
            b_K_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.{gqa}b_K", adapter)
            W_K_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.{gqa}W_K", adapter)
            b_V_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.{gqa}b_V", adapter)
            W_V_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.{gqa}W_V", adapter)
            ln1_b_key = ProcessWeights._get_param_key(f"blocks.{l}.ln1.b", adapter)
            ln1_w_key = ProcessWeights._get_param_key(f"blocks.{l}.ln1.w", adapter)

            # Fold ln1 into attention - it's important to fold biases first, since biases depend on
            # weights but not vice versa The various indexing is just to broadcast ln.b and ln.w
            # along every axis other than d_model. Each weight matrix right multiplies. To fold in
            # the bias, we use the W_ matrix to map it to the hidden space of the layer, so we need
            # to sum along axis -2, which is the residual stream space axis.
            if fold_biases:
                state_dict[b_Q_key] = state_dict[b_Q_key] + (
                    state_dict[W_Q_key] * state_dict[ln1_b_key][None, :, None]
                ).sum(-2)
                state_dict[b_K_key] = state_dict[b_K_key] + (
                    state_dict[W_K_key] * state_dict[ln1_b_key][None, :, None]
                ).sum(-2)
                state_dict[b_V_key] = state_dict[b_V_key] + (
                    state_dict[W_V_key] * state_dict[ln1_b_key][None, :, None]
                ).sum(-2)
                del state_dict[ln1_b_key]

            state_dict[W_Q_key] = state_dict[W_Q_key] * state_dict[ln1_w_key][None, :, None]
            state_dict[W_K_key] = state_dict[W_K_key] * state_dict[ln1_w_key][None, :, None]
            state_dict[W_V_key] = state_dict[W_V_key] * state_dict[ln1_w_key][None, :, None]
            del state_dict[ln1_w_key]

            # Finally, we center the weights reading from the residual stream. The output of the
            # first part of the LayerNorm is mean 0 and standard deviation 1, so the mean of any
            # input vector of the matrix doesn't matter and can be set to zero. Equivalently, the
            # output of LayerNormPre is orthogonal to the vector of all 1s (because dotting with
            # that gets the sum), so we can remove the component of the matrix parallel to this.
            if center_weights:
                state_dict[W_Q_key] -= einops.reduce(
                    state_dict[W_Q_key],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
                state_dict[W_K_key] -= einops.reduce(
                    state_dict[W_K_key],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
                state_dict[W_V_key] -= einops.reduce(
                    state_dict[W_V_key],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )

            # Fold ln2 into MLP
            if not getattr(cfg, "attn_only", False):
                # Get translated MLP parameter keys
                mlp_b_in_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.b_in", adapter)
                mlp_W_in_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.W_in", adapter)

                # Only get gate key if model has gated MLPs
                mlp_W_gate_key = None
                if getattr(cfg, "gated_mlp", False):
                    mlp_W_gate_key = ProcessWeights._get_param_key(
                        f"blocks.{l}.mlp.W_gate", adapter
                    )

                ln2_b_key = ProcessWeights._get_param_key(f"blocks.{l}.ln2.b", adapter)
                ln2_w_key = ProcessWeights._get_param_key(f"blocks.{l}.ln2.w", adapter)

                if fold_biases:
                    state_dict[mlp_b_in_key] = state_dict[mlp_b_in_key] + (
                        state_dict[mlp_W_in_key] * state_dict[ln2_b_key][:, None]
                    ).sum(-2)
                    del state_dict[ln2_b_key]

                state_dict[mlp_W_in_key] = state_dict[mlp_W_in_key] * state_dict[ln2_w_key][:, None]

                if getattr(cfg, "gated_mlp", False) and mlp_W_gate_key is not None:
                    state_dict[mlp_W_gate_key] = (
                        state_dict[mlp_W_gate_key] * state_dict[ln2_w_key][:, None]
                    )

                del state_dict[ln2_w_key]

                if center_weights:
                    # Center the weights that read in from the LayerNormPre
                    state_dict[mlp_W_in_key] -= einops.reduce(
                        state_dict[mlp_W_in_key],
                        "d_model d_mlp -> 1 d_mlp",
                        "mean",
                    )

                if getattr(cfg, "act_fn", None) is not None and cfg.act_fn.startswith("solu"):
                    # Get translated SoLU LayerNorm parameter keys
                    mlp_b_out_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.b_out", adapter)
                    mlp_W_out_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.W_out", adapter)
                    mlp_ln_b_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.ln.b", adapter)
                    mlp_ln_w_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.ln.w", adapter)

                    # Fold ln3 into activation
                    if fold_biases:
                        state_dict[mlp_b_out_key] = state_dict[mlp_b_out_key] + (
                            state_dict[mlp_W_out_key] * state_dict[mlp_ln_b_key][:, None]
                        ).sum(-2)

                        del state_dict[mlp_ln_b_key]

                    state_dict[mlp_W_out_key] = (
                        state_dict[mlp_W_out_key] * state_dict[mlp_ln_w_key][:, None]
                    )

                    if center_weights:
                        # Center the weights that read in from the LayerNormPre
                        state_dict[mlp_W_out_key] -= einops.reduce(
                            state_dict[mlp_W_out_key],
                            "d_mlp d_model -> 1 d_model",
                            "mean",
                        )

                    del state_dict[mlp_ln_w_key]

        # Fold ln_final into Unembed
        unembed_b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)
        unembed_W_U_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
        ln_final_b_key = ProcessWeights._get_param_key("ln_final.b", adapter)
        ln_final_w_key = ProcessWeights._get_param_key("ln_final.w", adapter)

        # Check if unembedding bias actually exists (some models like GPT-2 don't have it)
        has_unembed_bias = unembed_b_U_key in state_dict

        if not getattr(cfg, "final_rms", False) and fold_biases and has_unembed_bias:
            # Dumb bug from my old SoLU training code, some models have RMSNorm instead of LayerNorm
            # pre unembed.
            state_dict[unembed_b_U_key] = state_dict[unembed_b_U_key] + (
                state_dict[unembed_W_U_key] * state_dict[ln_final_b_key][:, None]
            ).sum(dim=-2)
            del state_dict[ln_final_b_key]

        state_dict[unembed_W_U_key] = (
            state_dict[unembed_W_U_key] * state_dict[ln_final_w_key][:, None]
        )
        del state_dict[ln_final_w_key]

        if center_weights:
            # Center the weights that read in from the LayerNormPre
            state_dict[unembed_W_U_key] -= einops.reduce(
                state_dict[unembed_W_U_key], "d_model d_vocab -> 1 d_vocab", "mean"
            )

        return state_dict

    @staticmethod
    def center_writing_weights(
        state_dict: Dict[str, torch.Tensor], cfg, adapter=None
    ) -> Dict[str, torch.Tensor]:
        """Center Writing Weights.

        Centers the weights of the model that write to the residual stream - W_out, W_E, W_pos and
        W_out. This is done by subtracting the mean of the weights from the weights themselves. This
        is done in-place. See fold_layer_norm for more details.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            cfg: Model configuration object.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with centered writing weights.
        """
        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        # Get translated parameter keys
        embed_W_E_key = ProcessWeights._get_param_key("embed.W_E", adapter)
        pos_embed_W_pos_key = ProcessWeights._get_param_key("pos_embed.W_pos", adapter)

        state_dict[embed_W_E_key] = state_dict[embed_W_E_key] - state_dict[embed_W_E_key].mean(
            -1, keepdim=True
        )
        if getattr(cfg, "positional_embedding_type", "standard") != "rotary":
            state_dict[pos_embed_W_pos_key] = state_dict[pos_embed_W_pos_key] - state_dict[
                pos_embed_W_pos_key
            ].mean(-1, keepdim=True)

        for l in range(cfg.n_layers):
            # Get translated parameter keys for this layer
            attn_W_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_O", adapter)
            attn_b_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_O", adapter)
            mlp_W_out_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.W_out", adapter)
            mlp_b_out_key = ProcessWeights._get_param_key(f"blocks.{l}.mlp.b_out", adapter)

            state_dict[attn_W_O_key] = state_dict[attn_W_O_key] - state_dict[attn_W_O_key].mean(
                -1, keepdim=True
            )  # W_O is [head_index, d_model, d_head]
            state_dict[attn_b_O_key] = (
                state_dict[attn_b_O_key] - state_dict[attn_b_O_key].mean()
            )  # b_O is [d_model]
            if not getattr(cfg, "attn_only", False):
                state_dict[mlp_W_out_key] = state_dict[mlp_W_out_key] - state_dict[
                    mlp_W_out_key
                ].mean(-1, keepdim=True)
                state_dict[mlp_b_out_key] = (
                    state_dict[mlp_b_out_key] - state_dict[mlp_b_out_key].mean()
                )
        return state_dict

    @staticmethod
    def center_unembed(
        state_dict: Dict[str, torch.Tensor], adapter=None
    ) -> Dict[str, torch.Tensor]:
        """Center the unembedding weights W_U.

        This is done by subtracting the mean of the weights from the weights themselves. This is
        done in-place. As softmax is translation invariant, this changes the logits but not the log
        probs, and makes the model logits (slightly) more interpretable - when trying to understand
        how components contribute to the logits, we'll be less misled by components that just add
        something to every logit.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with centered unembedding weights.
        """
        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        # Get translated parameter keys
        unembed_W_U_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
        unembed_b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)

        state_dict[unembed_W_U_key] = state_dict[unembed_W_U_key] - state_dict[
            unembed_W_U_key
        ].mean(-1, keepdim=True)

        # Only center bias if it exists (some models like GPT-2 don't have unembedding bias)
        if unembed_b_U_key in state_dict:
            state_dict[unembed_b_U_key] = (
                state_dict[unembed_b_U_key] - state_dict[unembed_b_U_key].mean()
            )
        return state_dict

    @staticmethod
    def fold_value_biases(
        state_dict: Dict[str, torch.Tensor], cfg, adapter=None
    ) -> Dict[str, torch.Tensor]:
        """Fold the value biases into the output bias.

        Because attention patterns add up to 1, the value biases always have a constant effect on a
        head's output. Further, as the outputs of each head in a layer add together, each head's
        value bias has a constant effect on the *layer's* output, which can make it harder to
        interpret the effect of any given head, and it doesn't matter which head a bias is
        associated with. We can factor this all into a single output bias to the layer, and make it
        easier to interpret the head's output. Formally, we take b_O_new = b_O_original +
        sum_head(b_V_head @ W_O_head).

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            cfg: Model configuration object.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with value biases folded into output bias.
        """
        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        for layer in range(cfg.n_layers):
            # Get translated parameter keys
            if getattr(cfg, "n_key_value_heads", None) is None:
                b_V_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_V", adapter)
                b_V = state_dict[b_V_key]
            else:
                b_V_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn._b_V", adapter)
                b_V = state_dict[b_V_key]
                b_V = torch.repeat_interleave(
                    b_V, dim=0, repeats=cfg.n_heads // cfg.n_key_value_heads
                )

            # Get other translated parameter keys
            W_O_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.W_O", adapter)
            b_O_key = ProcessWeights._get_param_key(f"blocks.{layer}.attn.b_O", adapter)

            # [head_index, d_head, d_model]
            W_O = state_dict[W_O_key]
            # [d_model]
            b_O_original = state_dict[b_O_key]
            folded_b_O = b_O_original + (b_V[:, :, None] * W_O).sum([0, 1])

            state_dict[b_O_key] = folded_b_O
            if getattr(cfg, "n_key_value_heads", None) is None:
                state_dict[b_V_key] = torch.zeros_like(b_V)
            else:
                state_dict[b_V_key] = torch.zeros_like(state_dict[b_V_key])
        return state_dict

    @staticmethod
    def refactor_factored_attn_matrices(
        state_dict: Dict[str, torch.Tensor], cfg, adapter=None
    ) -> Dict[str, torch.Tensor]:
        """Experimental method for managing queries, keys and values.

        As argued in [A Mathematical Framework for Transformer
        Circuits](https://transformer-circuits.pub/2021/framework/index.html), queries, keys and
        values are somewhat arbitrary intermediate terms when computing with the low rank factored
        matrices W_QK = W_Q @ W_K.T and W_OV = W_V @ W_O, and these matrices are the only thing
        determining head behaviour. But there are many ways to find a low rank factorization to a
        given matrix, and hopefully some of these are more interpretable than others! This method is
        one attempt, which makes all of the matrices have orthogonal rows or columns, W_O into a
        rotation and W_Q and W_K having the nth column in each having the same norm. The formula is
        $W_V = U @ S,W_O=Vh.T,W_Q=U@S.sqrt(),W_K=Vh@S.sqrt()$.

        More details:

        If W_OV = U @ S @ Vh.T in its singular value decomposition, (where S is in R^d_head not
        R^d_model, as W_OV is low rank), W_OV = (U @ S) @ (Vh.T) is an equivalent low rank
        factorisation, where rows/columns of each matrix are orthogonal! So setting $W_V=US$ and
        $W_O=Vh.T$ works just as well. I *think* this is a more interpretable setup, because now
        $W_O$ is just a rotation, and doesn't change the norm, so $z$ has the same norm as the
        result of the head.

        For $W_QK = W_Q @ W_K.T$ we use the refactor $W_Q = U @ S.sqrt()$ and $W_K = Vh @ S.sqrt()$,
        which is also equivalent ($S==S.sqrt() @ S.sqrt()$ as $S$ is diagonal). Here we keep the
        matrices as having the same norm, since there's not an obvious asymmetry between the keys
        and queries.

        Biases are more fiddly to deal with. For OV it's pretty easy - we just need (x @ W_V + b_V)
        @ W_O + b_O to be preserved, so we can set b_V' = 0. and b_O' = b_V @ W_O + b_O (note that
        b_V in R^{head_index x d_head} while b_O in R^{d_model}, so we need to sum b_V @ W_O along
        the head_index dimension too).

        For QK it's messy - we need to preserve the bilinear form of (x @ W_Q + b_Q) * (y @ W_K +
        b_K), which is fairly messy. To deal with the biases, we concatenate them to W_Q and W_K to
        simulate a d_model+1 dimensional input (whose final coordinate is always 1), do the SVD
        factorization on this effective matrix, then separate out into final weights and biases.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            cfg: Model configuration object.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Modified state dict with refactored attention matrices.
        """
        assert (
            getattr(cfg, "positional_embedding_type", "standard") != "rotary"
        ), "You can't refactor the QK circuit when using rotary embeddings (as the QK matrix depends on the position of the query and key)"

        # Make a copy to avoid modifying the original
        state_dict = state_dict.copy()

        for l in range(cfg.n_layers):
            # Get translated parameter keys
            W_Q_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_Q", adapter)
            b_Q_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_Q", adapter)
            W_K_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_K", adapter)
            b_K_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_K", adapter)
            W_V_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_V", adapter)
            W_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.W_O", adapter)
            b_V_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_V", adapter)
            b_O_key = ProcessWeights._get_param_key(f"blocks.{l}.attn.b_O", adapter)

            # W_QK = W_Q @ W_K.T
            # Concatenate biases to make a d_model+1 input dimension
            W_Q_eff = torch.cat(
                [
                    state_dict[W_Q_key],
                    state_dict[b_Q_key][:, None, :],
                ],
                dim=1,
            )
            W_K_eff = torch.cat(
                [
                    state_dict[W_K_key],
                    state_dict[b_K_key][:, None, :],
                ],
                dim=1,
            )

            W_Q_eff_even, W_K_eff_even_T = (
                FactoredMatrix(W_Q_eff, W_K_eff.transpose(-1, -2)).make_even().pair
            )
            W_K_eff_even = W_K_eff_even_T.transpose(-1, -2)

            state_dict[W_Q_key] = W_Q_eff_even[:, :-1, :]
            state_dict[b_Q_key] = W_Q_eff_even[:, -1, :]
            state_dict[W_K_key] = W_K_eff_even[:, :-1, :]
            state_dict[b_K_key] = W_K_eff_even[:, -1, :]

            # W_OV = W_V @ W_O
            W_V = state_dict[W_V_key]
            W_O = state_dict[W_O_key]

            # Factors the bias to be consistent.
            b_V = state_dict[b_V_key]
            b_O = state_dict[b_O_key]

            # Add singleton dimension for broadcasting
            b_V_expanded = einops.rearrange(b_V, "head_index d_head -> head_index d_head 1")

            # Element-wise multiplication of b_V and W_O
            b_V_times_W_O = b_V_expanded * W_O

            # Sum over d_head and head_index dimensions
            b_V_contribution = b_V_times_W_O.sum(1).sum(0)

            effective_bias = b_O + b_V_contribution
            state_dict[b_V_key] = torch.zeros_like(b_V)
            state_dict[b_O_key] = effective_bias

            # Helper class to efficiently deal with low rank factored matrices.
            W_OV = FactoredMatrix(W_V, W_O)
            U, S, Vh = W_OV.svd()
            state_dict[W_V_key] = U @ S.diag_embed()
            state_dict[W_O_key] = utils.transpose(Vh)

        return state_dict

    @staticmethod
    def process_weights(
        state_dict: Dict[str, torch.Tensor],
        cfg,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
        adapter=None,
    ) -> Dict[str, torch.Tensor]:
        """Apply all weight processing transformations in the correct order.

        This is a convenience function that applies all the weight processing steps
        in the same order as HookedTransformer.load_and_process_state_dict().

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of the model.
            cfg: Model configuration object.
            fold_ln (bool): Whether to fold LayerNorm weights into subsequent layers.
            center_writing_weights (bool): Whether to center weights writing to residual stream.
            center_unembed (bool): Whether to center unembedding weights.
            fold_value_biases (bool): Whether to fold value biases into output bias.
            refactor_factored_attn_matrices (bool): Whether to refactor attention matrices.
            adapter: Optional architecture adapter for parameter key translation.

        Returns:
            Dict[str, torch.Tensor]: Fully processed state dict.
        """
        processed_dict = state_dict.copy()

        if fold_ln:
            if getattr(cfg, "num_experts", None) and cfg.num_experts > 1:
                # Skip for MoE models
                pass
            elif getattr(cfg, "normalization_type", "LN") in ["LN", "LNPre"]:
                processed_dict = ProcessWeights.fold_layer_norm(
                    processed_dict, cfg, adapter=adapter
                )
            elif getattr(cfg, "normalization_type", "LN") in ["RMS", "RMSPre"]:
                processed_dict = ProcessWeights.fold_layer_norm(
                    processed_dict, cfg, fold_biases=False, center_weights=False, adapter=adapter
                )

        if center_writing_weights:
            if getattr(cfg, "normalization_type", "LN") in ["LN", "LNPre"] and not getattr(
                cfg, "final_rms", False
            ):
                processed_dict = ProcessWeights.center_writing_weights(
                    processed_dict, cfg, adapter=adapter
                )

        if center_unembed:
            processed_dict = ProcessWeights.center_unembed(processed_dict, adapter=adapter)

        if fold_value_biases:
            processed_dict = ProcessWeights.fold_value_biases(processed_dict, cfg, adapter=adapter)

        if refactor_factored_attn_matrices:
            processed_dict = ProcessWeights.refactor_factored_attn_matrices(
                processed_dict, cfg, adapter=adapter
            )

        return processed_dict
