import torch
from transformer_lens.loading_from_pretrained import get_pretrained_model_config, load_hugging_face_model
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.factories.weight_conversion_factory import WeightConversionFactory
from transformer_lens.weight_conversion.conversion_utils.conversion_steps.types import FIELD_SET
from transformer_lens.weight_conversion.conversion_utils.helpers.find_property import find_property
from transformer_lens.weight_conversion.conversion_utils.conversion_steps.weight_conversion_set import WeightConversionSet
from transformers import AutoModelForCausalLM, AutoTokenizer
from .types import LAYER_RESULT

class TransformersModelCompare:
    
    def __init__(self, model_name: str, device="cpu") -> None:
        self.model_name = model_name
        self.device = device

        self.cfg = get_pretrained_model_config(self.model_name)
        
    def compare(self, text: str)  -> LAYER_RESULT:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        transformers_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        transformer_lens_model = HookedTransformer.from_pretrained_no_processing(self.model_name, device=self.device)

        weight_conversion_config = WeightConversionFactory.select_weight_conversion_config(self.cfg)
        
        tokens = transformer_lens_model.to_tokens(text)
        input = tokenizer(text, return_tensors="pt")
        logit_difference = self.compare_model_logits(tokens, transformer_lens_model, input, transformers_model)
        self.compare_modules(weight_conversion_config.modules, tokens, transformer_lens_model, input, transformers_model)
        
        
    def compare_model_logits(self, transformer_lens_tokens, transformer_lens_model: HookedTransformer, transformers_input, transformers_model) -> float:
        
        transformer_lens_logits = transformer_lens_model(transformer_lens_tokens)[:, -1, :]
        with torch.no_grad():
            outputs = transformers_model(**transformers_input)
        transformers_logits = outputs.logits
        if not torch.equal(transformer_lens_logits, transformers_logits):
            atol = .1
            iterations = 0
            while torch.allclose(transformer_lens_logits, transformers_logits, rtol=1e-5, atol=atol):
                atol/=10
                iterations+=1
                if iterations >= 20:
                    break
            return atol
        else:
           return 0
        
        
    def compare_modules(self, modules: FIELD_SET, transformer_lens_tokens, transformer_lens_modules, transformers_input, transformers_modules):
        
        for transformer_lens_module_name in modules:
            remote_module_details = modules[transformer_lens_module_name]
            remote_module_name = remote_module_details[0] if isinstance(remote_module_details, tuple) else remote_module_details

            transformer_lens_module = find_property(transformer_lens_module_name, transformer_lens_modules)
            transformers_module = find_property(remote_module_name, transformers_modules)
            if isinstance(remote_module_details, tuple) and isinstance(remote_module_details[1], WeightConversionSet):
                layer_conversion_details = remote_module_details[1]
                self.compare_layers(layer_conversion_details.fields, transformer_lens_tokens, transformer_lens_module, transformers_input, transformers_module)
                
                
    def compare_layers(self, sub_modules: FIELD_SET, transformer_lens_tokens, transformer_lens_layers, transformers_input, transformers_layers):
        
        test_tensor = torch.randn((1, 1, self.cfg.d_model,))
        layer_number = 0
        for transformer_lens_layer in transformer_lens_layers:
            transformers_layer = transformers_layers[layer_number]
            layer_diff = self.compare_module(test_tensor, transformer_lens_layer, test_tensor, transformers_layer)
            self.compare_modules(sub_modules, test_tensor, transformer_lens_layer, test_tensor, transformers_layer)
            layer_number+=1
            
    def compare_module(self, transformer_lens_input, transformer_lens_module, transformers_input, transformers_module):
        transformer_lens_output = transformer_lens_module(transformer_lens_input)
        position_ids = torch.arange(0, 1, dtype=torch.long, device=transformers_input.device)
        position_ids = position_ids.unsqueeze(0).expand(1, 1)  # (B, L)
        transformers_output = transformers_module(transformers_input, position_ids=position_ids)[0]
        return transformer_lens_output - transformers_output
        