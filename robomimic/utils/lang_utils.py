import os

os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
tokenizer = "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32"

LANG_EMB_OBS_KEY = "lang_emb"

# lazily initialized on first call to @get_lang_emb, so that runs without language
# conditioning never pay for downloading / loading the CLIP model
lang_emb_model = None
tz = None

def _load_lang_emb_model():
    """
    Load the CLIP text model and tokenizer on first use, and cache them for subsequent calls.
    """
    global lang_emb_model, tz

    if lang_emb_model is None:
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
            tokenizer,
            cache_dir=os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/tmp"), "clip"))
        ).eval()
        tz = AutoTokenizer.from_pretrained(tokenizer, TOKENIZERS_PARALLELISM=True)

    return lang_emb_model, tz

def get_lang_emb(lang):
    if lang is None:
        return None

    lang_emb_model, tz = _load_lang_emb_model()

    tokens = tz(
        text=lang,                   # the sentence to be encoded
        add_special_tokens=True,             # Add [CLS] and [SEP]
        max_length=25,  # maximum length of a sentence
        padding="max_length",
        return_attention_mask=True,        # Generate the attention mask
        return_tensors="pt",               # ask the function to return PyTorch tensors
    )
    lang_emb = lang_emb_model(**tokens)['text_embeds'].detach()[0]

    return lang_emb

def get_lang_emb_shape():
    return list(get_lang_emb('dummy').shape)
