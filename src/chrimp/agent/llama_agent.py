import torch
import os
from chrimp.agent.nlp_agent import NLPAgent

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

class LlamaAgent(NLPAgent):
    def __init__(
        self,
        # Formats used during training
        inference_input_format:str, # any of "retro", "forward", "reac" (But only one)
    ):
        super().__init__(
            is_decoder_only = True,
            inference_input_format = inference_input_format,
        )

    def init_model(self, tokenizer_path: str, param_dict: dict):
        mandatory_params = [
            'hidden_size',
            'intermediate_size',
            'num_attention_heads',
            'num_hidden_layers',
            'rope_theta',
        ]

        missing_params = set(mandatory_params)-set(param_dict)
        superfluous_params = set(param_dict)-set(mandatory_params)

        if len(missing_params) > 0:
            raise ValueError(f"Parameters missing during initialization of the model: {missing_params}")
        elif len(superfluous_params) > 0:
            print(f"Caution! Parameters {superfluous_params} are not taken into account for initialization of the model")

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        self.config = LlamaConfig(
            hidden_size = param_dict['hidden_size'],
            intermediate_size = param_dict['intermediate_size'],
            num_attention_heads = param_dict['num_attention_heads'],
            num_hidden_layers = param_dict['num_hidden_layers'],
            rope_theta = param_dict['rope_theta'],
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            tie_word_embeddings = False, # Untied by default for Llama
        )

        self.model = LlamaForCausalLM(self.config)

    def load_model(self, tokenizer_path: str, model_path: str):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path, token=os.getenv("HF_TOKEN"))
        self.model = LlamaForCausalLM.from_pretrained(model_path, token=os.getenv("HF_TOKEN"))
        self.tokenizer.padding_side = "left"
        self.model
