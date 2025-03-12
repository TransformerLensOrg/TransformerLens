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
        self.compare_outputs(text, weight_conversion_config.field_set.fields, transformer_lens_model, transformers_model, tokenizer)
        
        text = "Hello my name is, "
        
        # transformer_lens_model.generate(text, max_new_tokens=50)
        # output = transformers_model.generate(**inputs, max_new_tokens=50, return_dict_in_generate=True, output_logits=True)
        # print("output", output.shape())
        # print("sequences", output.sequences)
        # print("logits", output.logits)
        
        
        # print("block0", transformer_lens_model.blocks[0].attn.W_Q)
        # print("layers0", transformers_model.model.layers[0].self_attn.q_proj.weight)
        
    def compare_outputs(self, text: str, fields: FIELD_SET, transformer_lens_model: HookedTransformer, transformers_model, tokenizer):
        
        tokens = transformer_lens_model.to_tokens(text)
        input = tokenizer(text, return_tensors="pt")
        print("tokens", tokens)
        print("input",  input)
        transformer_lens_logits = transformer_lens_model(tokens)[:, -1, :]
        transformers_logits = self.get_transformers_logits(transformers_model, input)[:, -1, :]
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
            print(f"Logics are close until {atol}")
        else:
            print("Logits are equal")
        # # print("diff", transformer_lens_output - transformers_output[0])
        # # print("transformers_output", transformers_output)
        # # for transformer_lens_field_name in fields:
        # #     remote_field = fields[transformer_lens_field_name]
        # #     if isinstance(remote_field, tuple) and isinstance(remote_field[1], WeightConversionSet):
        # #         remote_field_name, remote_field_conversion = remote_field
        # #         transformer_lens_layers = find_property(transformer_lens_field_name, transformer_lens_model)
        # #         transformers_layers = find_property(remote_field_name, transformers_model)
        # #         self.compare_layers(test_tensor, transformer_lens_field_name, transformer_lens_layers, transformers_layers)
                
                
    def get_transformers_logits(self, transformers_model, test_tensor):
        # Generate the logits
        with torch.no_grad():
            outputs = transformers_model(**test_tensor)
        
        # Get the logits for all tokens
        return outputs.logits
                
    def compare_layers(self, test_tensor: torch.Tensor, transformer_lens_field_name: str, transformer_lens_layers, transformers_layers, position_ids):
        
        
        
        
        layer_number = 0
        for transformer_lens_layer in transformer_lens_layers:
            layer_number+=1
            transformers_layer = transformers_layers[layer_number]
            transformer_lens_output = transformer_lens_layer(test_tensor)
            print("transformer_lens_output", transformer_lens_output)
            print("transformers_layer", transformers_layer)
            transformers_output = transformers_layer(test_tensor, position_ids=position_ids)
            print("diff", transformer_lens_output - transformers_output)