from __future__ import annotations

import os, sys, argparse, time, secrets, gc

from typing import Optional, Tuple

import torch
import gradio as gr

from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import AutoencoderKL
from diffusers import FluxPipeline as HF_FluxPipeline

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub import login as hf_login

from dype_flux.pipeline_flux import DyPE_FluxPipeline
from dype_flux.transformer_flux import DyPE_FluxTransformer2DModel

# key: transformer, value: base model
MODEL_PAIRS = {
    # from: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev
    # default one
    "black-forest-labs/FLUX.1-Krea-dev": "black-forest-labs/FLUX.1-Krea-dev",

    # uncensored version from: https://huggingface.co/aoxo/flux.1dev-abliterated
    # it can generate images but not sure if this is 100% working tho
    "aoxo/flux.1dev-abliterated": "aoxo/flux.1dev-abliterated",

    # from: https://civitai.com/models/1931032?modelVersionId=2207453
    # 2048x2048 works but 4096x4096 is cooked somehow
    "INtREPUS/unStable-Evolution-KREA-ckpt": "black-forest-labs/FLUX.1-Krea-dev",
    
    # from: https://civitai.com/models/679262/fux-capacity-nsfwporn-flux-base-model?modelVersionId=2298051
    "massdync/Fux-Capacity-ckpt": "black-forest-labs/FLUX.1-dev",

    # from: https://civitai.com/models/686814/jib-mix-flux?modelVersionId=2319074
    "massdync/Jib-Mix-Flux-ckpt": "black-forest-labs/FLUX.1-dev",

    # from: https://civitai.com/models/1775002?modelVersionId=2008873
    "massdync/Persephone-ckpt": "black-forest-labs/FLUX.1-dev",

    # from: https://civitai.com/models/1799857/cyberrealistic-flux?modelVersionId=2287992
    "massdync/CyberRealistic-Flux-ckpt": "black-forest-labs/FLUX.1-dev",

    # from: https://civitai.com/models/2086049?modelVersionId=2365910
    "INtREPUS/FLUXTRAIT-ckpt": "black-forest-labs/FLUX.1-dev",

    # from: https://civitai.com/models/978314/ultrareal-fine-tune?modelVersionId=1413133
    "INtREPUS/UltraReal-Fine-Tune-ckpt": "black-forest-labs/FLUX.1-dev",

    # from: https://civitai.com/models/161068?modelVersionId=979329
    "ads4ow2o1/STOIQO-NewReality-ckpt": "black-forest-labs/FLUX.1-dev",

    # from: https://civitai.com/models/861840?modelVersionId=2060393
    "ads4ow2o1/getphat-FLUX-Reality-NSFW-ckpt": "black-forest-labs/FLUX.1-dev",

    # https://huggingface.co/ads4ow2o2/Fluxmania-Kreamania
    "ads4ow2o2/Fluxmania-Kreamania-ckpt": "black-forest-labs/FLUX.1-dev"
}

TITLE = "DyPE (Dynamic Position Extrapolation) — Gradio UI"
DESCRIPTION = """
Ultra-high resolution text-to-image generation using **DyPE** on **FLUX.1-Krea-dev**.

- Toggle **DyPE** and choose **position method** (`yarn` / `ntk` / `base`).
- Choose **resolution**, **steps**, **guidance**, and **seed**.
- If the model is gated on Hugging Face, paste your **HF token**.
- Outputs are saved under `./outputs/` with informative filenames.
"""

DEFAULT_PROMPT = "A mysterious woman stands confidently in elaborate, dark armor adorned with intricate designs, holding a staff, against a backdrop of smoke and an ominous red sky, with shadowy, gothic buildings in the distance."

DROPDOWN_TITLE = 'Model (Use the default one, the other ones are test)'

THEME = gr.themes.Ocean(
    primary_hue="blue",
    secondary_hue="violet",
    radius_size="lg",
)

# Global cache so we don't reload every click
_PIPELINE = None
_PIPELINE_KEY: Tuple[str, str, bool, str, str] | None = None  # (base, ckpt, use_dype, method, dtype_opt)

def notify(msg: str):
    print(msg)
    gr.Info(msg)

def _download_models(api: HfApi, repo_ckpt: str, repo_base: str):
    # same repo, just snapshot_download
    if (repo_ckpt == repo_base):
        base_path = snapshot_download(repo_id=repo_base, repo_type='model')
        ckpt_path = base_path
    # download ckpt repo first then base repo
    else:
        files = api.list_repo_files(repo_id=repo_ckpt, repo_type='model')
        ckpts = [i for i in files if (i.endswith('.safetensors'))]
        if (not ckpts):
            msg = f'No .safetensors file found in repo [{repo_ckpt}]'
            print(msg)
            raise gr.Error(msg)
        ckpt_filename = ckpts[0]
        ckpt_path = hf_hub_download(repo_ckpt, ckpt_filename, repo_type='model')
        base_path = snapshot_download(repo_id=repo_base, repo_type='model')

    return ckpt_path, base_path

def _load_transformer_from_ckpt(repo_base: str, ckpt_path: str, method: str, use_dype: bool, dtype):
    text_encoder   = CLIPTextModel.from_pretrained(repo_base, subfolder="text_encoder", torch_dtype=dtype)
    tokenizer      = CLIPTokenizer.from_pretrained(repo_base, subfolder="tokenizer")
    text_encoder_2 = T5EncoderModel.from_pretrained(repo_base, subfolder="text_encoder_2", torch_dtype=dtype)
    tokenizer_2    = T5TokenizerFast.from_pretrained(repo_base, subfolder="tokenizer_2")
    vae            = AutoencoderKL.from_pretrained(repo_base, subfolder="vae", torch_dtype=dtype)

    tmp = HF_FluxPipeline.from_single_file(
        ckpt_path,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        torch_dtype=dtype,
    )

    del text_encoder, tokenizer, text_encoder_2, tokenizer_2, vae
    gc.collect()
    torch.cuda.empty_cache()

    src = tmp.transformer
    dype_transformer = DyPE_FluxTransformer2DModel.from_config(
        src.config, dype=use_dype, method=method
    ).to(dtype)

    missing, unexpected = dype_transformer.load_state_dict(src.state_dict(), strict=False)
    #print("missing:", len(missing), "unexpected:", len(unexpected))

    del tmp, src
    gc.collect()
    torch.cuda.empty_cache()

    return dype_transformer

def _load_transformer_from_base(model: str, method: str, use_dype: bool, dtype):
    transformer = DyPE_FluxTransformer2DModel.from_pretrained(
        model,
        subfolder="transformer",
        torch_dtype=dtype,
        dype=use_dype,
        method=method,
    )
    return transformer

def _get_pipeline(repo_base, repo_ckpt, hf_token, enable_dype, method, dtype_opt):
    global _PIPELINE, _PIPELINE_KEY

    key = (repo_base, repo_ckpt, enable_dype, method, dtype_opt)
    if _PIPELINE is not None and _PIPELINE_KEY == key:
        msg = f'Using cached pipeline: {key} ...'
        notify(msg)
        return _PIPELINE

    # If we’re switching configs/models, free the old one
    if (_PIPELINE is not None) and (_PIPELINE_KEY != key):
        try:
            _PIPELINE.to("cpu")
        except Exception:
            pass
        _PIPELINE = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    try:
        api = HfApi(token=hf_token)
    except Exception as e:
        msg = f"HF login failed: {e}"
        print(msg)
        gr.Warning(msg)

    device = _pick_device()
    dtype = _pick_dtype(device, dtype_opt)

    ckpt_path, base_path = _download_models(api, repo_ckpt, repo_base)
    if (ckpt_path == base_path):
        transformer = _load_transformer_from_base(base_path, method, enable_dype, dtype)
        msg = f'Loading transformer from base...'
    else:
        transformer = _load_transformer_from_ckpt(base_path, ckpt_path, method, enable_dype, dtype)
        msg = f'Loading transformer from ckpt...'
    
    notify(msg)

    pipe = DyPE_FluxPipeline.from_pretrained(
        base_path,
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipe.enable_model_cpu_offload()

    _PIPELINE = pipe
    _PIPELINE_KEY = key
    return pipe


def generate(
    prompt: str,
    height: int,
    width: int,
    steps: int,
    seed: int,
    method: str,
    enable_dype: bool,
    guidance_scale: float,
    hf_token: str,
    dtype_opt: str,
    model: str,
    randomize_seed: bool
):

    repo_ckpt = model
    repo_base = MODEL_PAIRS[model]
    print(f'Model: {repo_ckpt} | Base: {repo_base} | token: {hf_token}')

    if (hf_token):
        hf_login(hf_token)

    pipe = _get_pipeline(repo_base, repo_ckpt, hf_token, enable_dype, method, dtype_opt)

    # random seed, -ve seed also means random
    used_seed = int(seed)
    if randomize_seed or used_seed < 0:
        used_seed = next_seed()

    device = _pick_device()
    try:
        generator = torch.Generator(device).manual_seed(used_seed)
    except Exception:
        generator = torch.Generator().manual_seed(used_seed)
    
    # Generate
    image = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        generator=generator,
        num_inference_steps=int(steps),
    ).images[0]

    method_name = f"dy_{method}" if enable_dype else method
    ts = str(int(time.time()))

    os.makedirs("outputs", exist_ok=True)
    filename = f"outputs/seed_{used_seed}_method_{method_name}_res_{width}x{height}_{ts}.png"
    image.save(filename)

    return image, filename, used_seed

# ===== utils =====
def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _pick_dtype(device: str, dtype_opt: str):
    # Keep "auto" sensible; FLUX examples typically use bfloat16 on CUDA
    if dtype_opt == "bf16":
        return torch.bfloat16
    if dtype_opt == "fp16":
        return torch.float16
    if dtype_opt == "fp32":
        return torch.float32
    # auto
    if device == "cuda":
        return torch.bfloat16
    return torch.float32

def next_seed() -> int:
    # 0 .. 2^63-1 — safe for torch.Generator().manual_seed
    return secrets.randbits(63)

def _format_dropdown_title(choice_key: str):
    return f'{DROPDOWN_TITLE} | Base model: {MODEL_PAIRS[choice_key]}'

def _update_dropdown_title(selected_key: str):
    return gr.update(label=_format_dropdown_title(selected_key))

with gr.Blocks(title=TITLE, fill_height=True, theme=THEME) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        DEFAULT_CHOICE = "black-forest-labs/FLUX.1-Krea-dev"
        model = gr.Dropdown(
            label=_format_dropdown_title(choice_key=DEFAULT_CHOICE),
            choices=MODEL_PAIRS.keys(),
            value=DEFAULT_CHOICE
        )
        hf_token = gr.Textbox(label="Hugging Face token (if gated)", type="password", placeholder="hf_... (optional)")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=4, autofocus=True)

    with gr.Row():
        MAX_RES = 4096
        STEPS = 16      # was 64
        width = gr.Slider(512, MAX_RES, value=MAX_RES // 2, step=STEPS, label="Width (px)")
        height = gr.Slider(512, MAX_RES, value=MAX_RES // 2, step=STEPS, label="Height (px)")

    with gr.Row():
        steps = gr.Slider(1, 64, value=28, step=1, label="Inference steps")
        guidance = gr.Slider(0.0, 10.0, value=4.5, step=0.1, label="Guidance scale")

    with gr.Row():
        seed = gr.Number(value=42, precision=0, label="Seed")
        method = gr.Dropdown(choices=["yarn", "ntk", "base"], value="yarn", label="Position method")
        dtype_opt = gr.Dropdown(choices=["auto", "bf16", "fp16", "fp32"], value="auto", label="Torch dtype")
        enable_dype = gr.Checkbox(value=True, label="Enable DyPE")

    with gr.Row():
        randomize_seed = gr.Checkbox(value=False, label="🎲 Randomize each run")
        roll_btn = gr.Button("🎲 Roll seed now")

    submit = gr.Button("🚀 Generate", variant="primary")
    out_img = gr.Image(label="Result", interactive=False)
    out_file = gr.File(label="Saved image (.png)")

    model.change(_update_dropdown_title, inputs=model, outputs=model)
    roll_btn.click(fn=next_seed, inputs=None, outputs=[seed])

    submit.click(
        fn=generate,
        inputs=[prompt, height, width, steps, seed, method, enable_dype, guidance, hf_token, dtype_opt, model, randomize_seed],
        outputs=[out_img, out_file, seed],
        api_name="generate",
    )

    gr.Markdown("Tip: First run may take a while to download weights. Images are saved under `./outputs/`.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true', required=False)
    #parser.add_argument('--debug', action='store_true', required=False)
    parsed, args = parser.parse_known_args(sys.argv)

    demo.queue(max_size=8).launch(
        share=parsed.share,
        #debug=parsed.debug
    )
