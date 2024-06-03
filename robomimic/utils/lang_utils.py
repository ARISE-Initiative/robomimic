import os
import torch
from transformers import AutoModel, pipeline, AutoTokenizer, CLIPTextModelWithProjection

class LangEncoder:
    def __init__(self, device):
        os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
        model_variant = "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32"
        self.device = device
        self.lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
            model_variant,
            cache_dir=os.path.expanduser("~/tmp/clip")
        ).to(device).eval()
        self.tz = AutoTokenizer.from_pretrained(model_variant, TOKENIZERS_PARALLELISM=True)

    def get_lang_emb(self, lang):
        if lang is None:
            return None
        
        with torch.no_grad():
            tokens = self.tz(
                text=lang,                   # the sentence to be encoded
                add_special_tokens=True,             # Add [CLS] and [SEP]
                # max_length=25,  # maximum length of a sentence
                padding="max_length",
                return_attention_mask=True,        # Generate the attention mask
                return_tensors="pt",               # ask the function to return PyTorch tensors
            ).to(self.device)

            lang_emb = self.lang_emb_model(**tokens)['text_embeds'].detach()
        
        # check if input is batched or single string
        if isinstance(lang, str):
            lang_emb = lang_emb[0]

        return lang_emb

