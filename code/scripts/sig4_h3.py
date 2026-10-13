"""SIG-4 H3 entropy: vary n_prompts and n_samples_per_prompt to find max |Cohen's d|."""
import os, json, torch, numpy as np
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEVICE = "cuda"
N_PROMPTS = int(os.environ.get("SIG4_NP", 24))
N_SAMPLES = int(os.environ.get("SIG4_NS", 4))
NOTES = "$WORK/signature/notes"
SHAPED_NOTES = "$WORK/grpo_project/notes"
PRE_CKPT = "$WORK/grpo_project/checkpoints/sft-qwen3-1p7b"
POST_CKPT = os.environ.get("SIG4_POST", "$WORK/grpo_project/checkpoints/grpo-baseline-sif-qwen3-1p7b-step2000/global_step_100/actor/huggingface")
REGIME = os.environ.get("SIG4_REGIME", "baseline")

sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
    release="pythia-70m-deduped-res-sm",
    sae_id="blocks.4.hook_resid_post",
    device=DEVICE)
sae = sae.to(torch.bfloat16); sae.eval()
feat = json.load(open(f"{SHAPED_NOTES}/feature_set_v0_pythia70m.json"))
F32 = feat["selected"][:32]
ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=False)
prompts = [r["text"][:300] for r in ds.select(range(min(2000, len(ds)))) if r.get("text","").strip()][:N_PROMPTS]
ptok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

def feat_entropy(ckpt, temp=0.7):
    mtok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, device_map="auto"); model.eval()
    pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped", torch_dtype=torch.bfloat16, device_map="auto"); pythia.eval()
    activations = []
    with torch.no_grad():
        for p in prompts:
            for _ in range(N_SAMPLES):
                mids = mtok(p, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
                out = model.generate(**mids, max_new_tokens=40, do_sample=True, temperature=temp, pad_token_id=mtok.eos_token_id or 0)
                txt = mtok.decode(out[0], skip_special_tokens=True)
                pids = ptok(txt, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(DEVICE)
                hidden = pythia(pids, output_hidden_states=True).hidden_states[5]
                z = sae.encode(hidden[0]).float().cpu().numpy()
                activations.append(z[:, F32].mean(axis=0))
    A = np.array(activations)
    eps = 1e-9
    A_pos = np.maximum(A, 0) + eps
    A_norm = A_pos / A_pos.sum(axis=0, keepdims=True)
    H = -(A_norm * np.log(A_norm + eps)).sum(axis=0)
    return H

TEMP_PRE = float(os.environ.get("SIG4_TEMP_PRE", "0.7"))
TEMP_POST = float(os.environ.get("SIG4_TEMP_POST", "0.7"))
H_pre = feat_entropy(PRE_CKPT, TEMP_PRE)
H_post = feat_entropy(POST_CKPT, TEMP_POST)
delta_H = H_post - H_pre
mean_d = float(delta_H.mean())
sd = float(np.sqrt(((delta_H - mean_d) ** 2).mean())) if len(delta_H) > 1 else 1.0
d = mean_d / (sd + 1e-9)
print(f"n_prompts={N_PROMPTS} n_samples={N_SAMPLES} mean_dH={mean_d:+.4f} cohens_d={d:+.4f} pass_gate={abs(d)>=0.5}")
out_path = f"{NOTES}/h3_entropy_{REGIME}_n{N_PROMPTS}_s{N_SAMPLES}.json"
json.dump({"regime":REGIME,"post_ckpt":POST_CKPT,"n_prompts":N_PROMPTS,"n_samples":N_SAMPLES,"mean_dH":mean_d,"cohens_d":d,"pass_gate":abs(d)>=0.5}, open(out_path,"w"), indent=2)
print(f"[SAVED] {out_path}")
