"""Test bonus boundary detection with multi-token string values.

Prompt: ask for text values like "dark brown" (multi-token) instead of true/false.
Verify: bonus (target's next-token prediction after value) correctly detects
whether value has ended (structural: , " }) or not (value content).
"""
import os, sys, json, io, torch
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
sys.path.insert(0, "/root/autodl-tmp/TorchSpec_DChord")
sys.path.insert(0, "/root/autodl-tmp/dchord_v22_torch_repro_code_v2/overlay/experiments/dchord/repro")
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import common as DCC

TARGET = "/root/autodl-tmp/models/Qwen3.5-4B"
DEV = "cuda:0"
# Prompt with multi-token string values
PROMPT = """Analyze the face in the image. Output ONLY a JSON object (no markdown, no explanation) with these fields:
{"hair_color": "<color like dark brown, black, blond>", "gender": "<male or female>", "expression": "<like happy smile, neutral face>", "age": "<young or old>"}
Example output: {"hair_color":"dark brown","gender":"male","expression":"happy smile","age":"young"}
Now output the JSON for this image:"""

# Structural tokens for bonus boundary detection
STRUCTURAL = {11, 3307, 1, 198}  # , }} " \n
def is_structural(bonus_token, tok):
    """Check if bonus token indicates value boundary (structural char in token text)."""
    if bonus_token in STRUCTURAL:
        return True
    s = tok.decode([bonus_token])
    return any(c in s for c in ',}"\n')  # merged tokens like '",' or ' }}'

proc = AutoProcessor.from_pretrained(TARGET, trust_remote_code=True)
tok = proc.tokenizer
target = AutoModelForImageTextToText.from_pretrained(
    TARGET, dtype=torch.bfloat16, device_map=DEV, attn_implementation="sdpa").to(DEV).eval()

img = Image.open("/root/autodl-tmp/celeba_test.jpg").convert("RGB")
msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":PROMPT}]}]
encoded = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                    return_dict=True, return_tensors="pt", enable_thinking=False)
encoded = {k: v.to(DEV) if torch.is_tensor(v) else v for k, v in encoded.items()}

# AR generate token by token, track value boundaries
print("=== AR generation with bonus boundary detection ===")
completion_ids = []
max_new = 80
# prefill
out = target(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
             use_cache=True, output_hidden_states=False, logits_to_keep=1,
             **{k:v for k,v in encoded.items() if k not in ("input_ids","attention_mask")})
pkv = out.past_key_values
next_tok = int(out.logits[0,-1].argmax())
completion_ids.append(next_tok)
print(f"  step 0: first token={tok.decode([next_tok])!r} (id={next_tok})")
in_value = False
value_tokens = []
step = 0

for i in range(max_new):
    tok_str = tok.decode([next_tok])
    is_struct = next_tok in STRUCTURAL
    # track value boundary
    if tok_str == '"' and not in_value:
        # opening quote of a value
        in_value = True
        value_tokens = []
        print(f"  step {step}: token={tok_str!r} (id={next_tok}) -> VALUE START")
    elif tok_str == '"' and in_value:
        # closing quote of a value
        val_str = tok.decode(value_tokens)
        print(f"  step {step}: token={tok_str!r} (id={next_tok}) -> VALUE END, value={val_str!r} ({len(value_tokens)} tokens)")
        in_value = False
    elif in_value:
        value_tokens.append(next_tok)
    
    # get bonus: target's prediction at current position (predicts NEXT token)
    # we already have next_tok from previous step; now forward to get next
    step += 1
    inp = torch.tensor([[next_tok]], device=DEV)
    out = target(input_ids=inp, past_key_values=pkv, use_cache=True, logits_to_keep=1)
    pkv = out.past_key_values
    bonus = int(out.logits[0,-1].argmax())
    bonus_str = tok.decode([bonus])
    bonus_is_struct = is_structural(bonus, tok)
    
    if in_value or (tok_str == '"' and not in_value):
        # inside value or just started value -> check bonus
        status = "ENDED" if bonus_is_struct else "NOT ENDED (content)"
        print(f"    bonus={bonus_str!r} (id={bonus}) -> {status}")
    
    if bonus in (tok.eos_token_id, 151645, 151643) and not in_value:
        print(f"  step {step}: EOS/newline, stopping")
        next_tok = bonus
        completion_ids.append(next_tok)
        break
    next_tok = bonus
    completion_ids.append(next_tok)

print(f"\n=== Generated JSON ===")
print(tok.decode(completion_ids, skip_special_tokens=True))
print("DONE")
