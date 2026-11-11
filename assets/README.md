# QRF Length Predictor (Pretrained)

JITServe uses a **QRF (Quantile Regression Forest)–based length predictor**
trained by the authors to estimate conservative upper bounds on LLM output
lengths. This predictor is used **only for scheduling decisions** and does
not affect model outputs.

This directory serves as the **local download location** for the pretrained
QRF artifacts.

---

## Download

The pretrained QRF predictor (model + vectorizer) is hosted on Hugging Face:

https://huggingface.co/En-2863/jitserve-qrf-length-predictor

### Step 1: Install Hugging Face CLI

```bash
pip install huggingface_hub
```

### Step 2: Download into this directory
From the repository root, run:

```bash
huggingface-cli download En-2863/jitserve-qrf-length-predictor \
  --local-dir assets/qrf/ \
  --local-dir-use-symlinks False
```

### Alternative: Use JITServe's uv environment
From the repository root:

```bash
uv pip install -e .
source .venv/bin/activate
python assets/download_qrf.py
```
