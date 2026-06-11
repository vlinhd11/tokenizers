"""Test the fork's ViLLM PreTrainedTokenizerFast integration."""
import json, math, sys, os
from pathlib import Path

BASE = Path(r"D:\AGI\viLLM\tokenizer\outputs\villm-tokenizer")
assert BASE.exists(), f"{BASE} not found"

# 1. Load vocab & config
token2id = json.loads((BASE / "vocab.json").read_text("utf-8"))
token_meta = json.loads((BASE / "token_meta.json").read_text("utf-8"))
vc = {}
if (BASE / "villm_config.json").exists():
    vc = json.loads((BASE / "villm_config.json").read_text("utf-8"))

vi_syllables = [k for k, m in token_meta.items() if m.get("type") == "vi_syllable"]
vi_compounds = {
    k: math.log(m["freq"])
    for k, m in token_meta.items()
    if m.get("type") == "vi_compound" and m.get("freq", 0) >= 500
}

import sentencepiece as _spm
sp = _spm.SentencePieceProcessor()
sp.Load(str(BASE / "sp_en.model"))
sp_pieces = [(sp.IdToPiece(i), sp.GetScore(i)) for i in range(sp.GetPieceSize())]

cs_vi_en = vc.get("cs_vi_en", "[VI\u2192EN]")
cs_en_vi = vc.get("cs_en_vi", "[EN\u2192VI]")
unk_token = vc.get("unk_token", "[UNK]")

# 2. Build tokenizer using fork's native API
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

villm = models.ViLLM(
    token2id=token2id,
    unk_token=unk_token,
    vi_syllables=vi_syllables,
    base_forms=[],
    vi_compounds=vi_compounds,
    compound_unigram_score=0.0,
    sp_pieces=sp_pieces,
    cs_vi_en=cs_vi_en,
    cs_en_vi=cs_en_vi,
)

tok = Tokenizer(model=villm)
tok.pre_tokenizer = pre_tokenizers.ViLLM()
tok.decoder = decoders.ViLLM(cs_vi_en=cs_vi_en, cs_en_vi=cs_en_vi)

# 3. Basic encode/decode tests
def test_basic():
    text = "Xin chào thế giới"
    enc = tok.encode(text)
    print(f"  Input: {text!r}")
    print(f"  IDs: {enc.ids[:20]}")
    print(f"  Tokens: {enc.tokens[:20]}")
    decoded = tok.decode(enc.ids)
    print(f"  Decoded: {decoded!r}")
    assert len(enc.ids) > 0, "Empty encoding!"
    return enc

print("=== Basic encode/decode ===")
test_basic()

def test_vi_en_mixed():
    text = "Tôi muốn học machine learning và deep learning"
    enc = tok.encode(text)
    print(f"  Input: {text!r}")
    print(f"  IDs: {enc.ids[:30]}")
    print(f"  Tokens: {enc.tokens[:30]}")
    decoded = tok.decode(enc.ids)
    print(f"  Decoded: {decoded!r}")
    return enc

print("\n=== VI/EN mixed ===")
test_vi_en_mixed()

def test_byte_fallback():
    text = "abcXYZ123"
    enc = tok.encode(text)
    print(f"  Input: {text!r}")
    print(f"  Tokens: {enc.tokens[:20]}")
    has_byte = any(t.startswith("<0x") for t in enc.tokens)
    print(f"  Has byte tokens: {has_byte}")
    decoded = tok.decode(enc.ids)
    print(f"  Decoded: {decoded!r}")
    assert decoded.strip() == text, f"Round-trip failed: {decoded!r} != {text!r}"

print("\n=== Byte fallback round-trip ===")
test_byte_fallback()

def test_encode_batch():
    texts = ["Xin chào", "Hello world", "Tôi là sinh viên"]
    encodings = tok.encode_batch(texts)
    for t, e in zip(texts, encodings):
        d = tok.decode(e.ids)
        print(f"  {t!r} -> {d!r}")
    assert len(encodings) == len(texts)

print("\n=== Batch encode ===")
test_encode_batch()

# 4. PreTrainedTokenizerFast integration
print("\n=== PreTrainedTokenizerFast ===")
from transformers import PreTrainedTokenizerFast
hf_tok = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    unk_token=unk_token,
    bos_token="[BOS]",
    eos_token="[EOS]",
    pad_token="[PAD]",
)
print(f"  Vocab size: {hf_tok.vocab_size}")
enc = hf_tok("Xin chào thế giới", return_tensors="np")
print(f"  Input IDs: {enc['input_ids'].tolist()}")
decoded = hf_tok.decode(enc["input_ids"][0].tolist())
print(f"  Decoded: {decoded!r}")
assert len(enc["input_ids"][0]) > 0

# 5. Serde round-trip (save & reload via tokenizer.json)
print("\n=== Serde round-trip ===")
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = Path(tmpdir) / "tokenizer.json"
    tok.save(str(save_path))
    assert save_path.exists(), "tokenizer.json not saved"
    print(f"  Saved to {save_path} ({save_path.stat().st_size} bytes)")
    loaded = Tokenizer.from_file(str(save_path))
    # Verify type
    print(f"  Loaded model type: {type(loaded.model).__name__}")
    text = "Xin chào thế giới"
    orig_enc = tok.encode(text)
    loaded_enc = loaded.encode(text)
    match = orig_enc.ids == loaded_enc.ids
    print(f"  IDs match: {match}")
    if not match:
        print(f"  Orig: {orig_enc.ids[:20]}")
        print(f"  Load: {loaded_enc.ids[:20]}")
    assert match, "Serde round-trip mismatch!"

print("\n=== ALL TESTS PASSED ===")
