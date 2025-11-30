from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# ===== Load model once =====
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sample", methods=["POST"])
def sample():
    data = request.json
    prompt = data["prompt"]
    temp = float(data["temperature"])
    top_k = int(data["top_k"])
    top_p = float(data["top_p"])
    min_p = float(data["min_p"])

    # ===== Get logits from model =====
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]

    # ===== Temperature scaling =====
    logits = logits / temp
    probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()

    # ===== Sorting =====
    sorted_idx = np.argsort(probs)[::-1]
    sorted_tokens = [
        tokenizer.decode([int(i)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for i in sorted_idx
    ]
    sorted_probs = probs[sorted_idx]

    # ===== Top-k =====
    topk_tokens = sorted_tokens[:top_k]

    # ===== Top-p (nucleus) =====
    cum = 0
    top_p_tokens = []
    for i in sorted_idx:
        top_p_tokens.append(
            tokenizer.decode([int(i)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        )
        cum += probs[i]
        if cum >= top_p:
            break

    # ===== Min-p =====
    minp_tokens = [
        tokenizer.decode([int(i)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for i in sorted_idx if probs[i] >= min_p
    ]

    return jsonify({
        "tokens": sorted_tokens[:20],         # top 20 for display
        "probs": sorted_probs[:20].tolist(),  # same here
        "topk": topk_tokens,
        "topp": top_p_tokens,
        "minp": minp_tokens
    })


if __name__ == "__main__":
    app.run(debug=True)
