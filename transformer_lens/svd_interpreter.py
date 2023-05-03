from typing import List, Tuple
import torch
from transformer_lens import HookedTransformer

class SVD_Interpreter:
  
  def __init__(self, model: HookedTransformer):
    self.model = model
    self.cfg = model.cfg
    self.params = {name: param for name, param in model.named_parameters()}

    def get_OV_matrix(self, layer_index: int, head_index: int) -> torch.Tensor(self.cfg.d_model, self.cfg.d_model):
        assert 0 <= layer_index <= self.cfg.n_layers
        assert 0 <= head_index <= self.cfg.n_heads

        W_V, W_O = self.params[f"blocks.{layer_index}.attn.W_V"], self.params[f"blocks.{layer_index}.attn.W_O"]
        W_V, W_O = W_V[head_index, :, :].squeeze(0), W_O[head_index, :, :].squeeze(0)

        W_OV = W_V @ W_O
        return W_OV

    def get_top_singular_vectors(self, 
                            matrix: torch.Tensor, 
                            embedding: torch.Tensor(self.cfg.d_vocab, self.cfg.d_model), 
                            num_vectors: int=10,
                            top_k: int=20) -> Tuple[List[List[str]], List[List[int]]]:
        
        U, S, V = torch.linalg.svd(matrix)
        vectors = []

        for i in range(num_vectors):
            activations = V[i,:].float() @ embedding.T
            vectors.append(activations)

        vectors = torch.stack(vectors, dim=1).unsqueeze(1)
        assert vectors.shape == (self.cfg.d_vocab, 1, num_vectors)
    
        top_k_words, top_k_vals = get_top_k_from_vectors(vectors)
        return top_k_words, top_k_vals
    
    def get_top_k_from_vectors(vectors: torch.Tensor, top_k: int) -> Tuple[List[List[str]], List[List[int]]]:
        num_vectors = vectors.shape(-1)
        top_k_words, top_k_vals = [], []
        
        for i in range(num_vectors):
            top_k_tokens = vectors[:, :, i].topk(k=top_k, dim=0)

            # Strings represented by top K tokens.
            top_k_words.append([self.model.to_string(top_k_tokens.indices[i].item()) for i in range(top_k)])

            # SVD values of top K tokens.
            top_k_vals.append([top_k_tokens.values[i].item() for i in range(top_k)])

        assert len(top_k_words) == len(top_k_vals) == num_vectors
        assert len(top_k_words[0]) == len(top_k_vals[0]) == top_k
        
        return top_k_words, top_k_vals
