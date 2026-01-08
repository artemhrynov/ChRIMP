import os
from chrimp.agent.nlp_agent import NLPAgent

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.models.t5 import T5Config
from transformers.models.t5 import T5ForConditionalGeneration


class T5Agent(NLPAgent):
    def __init__(
        self,
        # Formats used during training
        inference_input_format: str,  # any of "retro", "forward", "reac" (But only one)
    ):
        super().__init__(
            is_decoder_only=False,
            inference_input_format=inference_input_format,
        )

    def init_model(self, tokenizer_path: str, param_dict: dict):
        mandatory_params = [
            "d_model",  # Embedding size
            "d_ff",  # FFN inner size (often 2x or 4x d_model)
            "num_heads",  # Attention heads (must divide d_model)
            "num_layers",  # Number of transformer layers
        ]

        missing_params = set(mandatory_params) - set(param_dict)
        superfluous_params = set(param_dict) - set(mandatory_params)

        if len(missing_params) > 0:
            raise ValueError(
                f"Parameters missing during initialization of the model: {missing_params}"
            )
        elif len(superfluous_params) > 0:
            print(
                f"Caution! Parameters {superfluous_params} are not taken into account for initialization of the model"
            )

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        self.config = T5Config(
            d_model=param_dict["d_model"],
            d_ff=param_dict["d_ff"],
            num_heads=param_dict["num_heads"],
            num_layers=param_dict["num_layers"],
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            decoder_start_token_id=self.tokenizer.pad_token_id,  # T5 uses pad as start token by default
            tie_word_embeddings=True,  # Tied by default
        )

        self.model = T5ForConditionalGeneration(self.config)

    def load_model(self, tokenizer_path: str, model_path: str):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            tokenizer_path, token=os.getenv("HF_TOKEN")
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path, token=os.getenv("HF_TOKEN")
        )
        self.tokenizer.padding_side = "right"
        self.model
