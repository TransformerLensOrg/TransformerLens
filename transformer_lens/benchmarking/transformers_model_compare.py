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
        
    def compare(self, text: str)  -> LAYER_RESULT:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        transformers_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        transformer_lens_model = HookedTransformer.from_pretrained_no_processing(self.model_name, device=self.device)

        cfg = get_pretrained_model_config(self.model_name)
        weight_conversion_config = WeightConversionFactory.select_weight_conversion_config(cfg)
        
        # test_tensor = torch.randn((1, 1, cfg.d_model,))
        
        # inputs_embeds = transformers_model.model.embed_tokens(test_tensor)
        # cache_position = torch.arange(
        #     0, inputs_embeds.shape[1], device=inputs_embeds.device
        # )
        # position_ids = cache_position.unsqueeze(0)
        tokens = transformer_lens_model.to_tokens(text)
        input = tokenizer(text, return_tensors="pt")
        logit_difference = self.compare_model_logits(tokens, transformer_lens_model, input, transformers_model)
        print("logit_difference", logit_difference)
        self.compare_modules(weight_conversion_config.modules, tokens, transformer_lens_model, input, transformers_model)
        
        text = "Hello my name is, "
        # transformer_lens_model.generate(text, max_new_tokens=50)
        # output = transformers_model.generate(**inputs, max_new_tokens=50, return_dict_in_generate=True, output_logits=True)
        # print("output", output.shape())
        # print("sequences", output.sequences)
        # print("logits", output.logits)
        
        
        # print("block0", transformer_lens_model.blocks[0].attn.W_Q)
        # print("layers0", transformers_model.model.layers[0].self_attn.q_proj.weight)
        
    def compare_model_logits(self, transformer_lens_tokens, transformer_lens_model: HookedTransformer, transformers_input, transformers_model) -> float:
        
        transformer_lens_logits = transformer_lens_model(transformer_lens_tokens)[:, -1, :]
        with torch.no_grad():
            outputs = transformers_model(**transformers_input)
        transformers_logits = outputs.logits
        # print("transformer_lens_logits", transformer_lens_logits)
        # print("transformers_logits", transformers_logits)
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
        
        # # print("diff", transformer_lens_output - transformers_output[0])
        # # print("transformers_output", transformers_output)
        for transformer_lens_module_name in modules:
            remote_module_details = modules[transformer_lens_module_name]
            remote_module_name = remote_module_details[0] if isinstance(remote_module_details, tuple) else remote_module_details

            transformer_lens_module = find_property(transformer_lens_module_name, transformer_lens_modules)
            transformers_module = find_property(remote_module_name, transformers_modules)
            if isinstance(remote_module_details, tuple) and isinstance(remote_module_details[1], WeightConversionSet):
                layer_conversion_details = remote_module_details[1]
                self.compare_layers(layer_conversion_details.fields, transformer_lens_tokens, transformer_lens_module, transformers_input, transformers_module)
            # else:
                
            #     if isinstance(remote_module_name, tuple):
            #         output_conversion = remote_module_details[1]
            #     diff = self.compare_module(transformer_lens_tokens, transformer_lens_module, transformers_input, transformers_module)
                # print("module diff", diff)
                # todo compare individual module
                # remote_module_name, remote_module_conversion = remote_module
                # transformer_lens_layers = find_property(transformer_lens_field_name, transformer_lens_model)
                # t
                # self.compare_layers(test_tensor, transformer_lens_field_name, transformer_lens_layers, transformers_layers)
                
                
    def compare_layers(self, sub_modules: FIELD_SET, transformer_lens_tokens, transformer_lens_layers, transformers_input, transformers_layers):
        
        test_tensor = torch.randn((1, 1, 768,))
        layer_number = 0
        for transformer_lens_layer in transformer_lens_layers:
            transformers_layer = transformers_layers[layer_number]
            layer_diff = self.compare_module(test_tensor, transformer_lens_layer, test_tensor, transformers_layer)
            print("layer_diff", layer_diff)
            self.compare_modules(sub_modules, test_tensor, transformer_lens_layer, test_tensor, transformers_layer)
            layer_number+=1
            
    def compare_module(self, transformer_lens_tokens, transformer_lens_module, transformers_input, transformers_module):
        transformer_lens_output = transformer_lens_module(transformer_lens_tokens)
        position_ids = torch.arange(0, 1, dtype=torch.long, device=transformers_input.device)
        position_ids = position_ids.unsqueeze(0).expand(1, 1)  # (B, L)
        transformers_output = transformers_module(transformers_input, position_ids=position_ids)[0]
        return transformer_lens_output - transformers_output
        