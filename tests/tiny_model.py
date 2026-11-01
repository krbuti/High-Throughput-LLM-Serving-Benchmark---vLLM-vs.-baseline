"""Build a tiny random Qwen2-architecture model + BPE tokenizer on disk (no downloads)."""

from pathlib import Path


def make_tiny_model(out: Path) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    from bench.dataset import _QUESTIONS, _WORDS

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        [" ".join(_WORDS)] * 50 + _QUESTIONS * 20,
        trainers.BpeTrainer(vocab_size=600, special_tokens=["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                 unk_token="<unk>")
    hf.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    hf.save_pretrained(out)
    cfg = Qwen2Config(vocab_size=len(hf), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      eos_token_id=hf.eos_token_id, pad_token_id=hf.pad_token_id)
    Qwen2ForCausalLM(cfg).save_pretrained(out)
    return out
