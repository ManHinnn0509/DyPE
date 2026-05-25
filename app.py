from __future__ import annotations

import os, sys, argparse, time, secrets, gc

from textwrap import dedent
from typing import Optional, Tuple

import torch
import gradio as gr
import requests as req

from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers import FluxPipeline as HF_FluxPipeline

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download, snapshot_download, hf_hub_url
from huggingface_hub import login as hf_login

from dype_flux.pipeline_flux import DyPE_FluxPipeline
from dype_flux.transformer_flux import DyPE_FluxTransformer2DModel

try:
    from dype_qwen.transformer_qwenimage import QwenImageTransformer2DModel
    _QWEN_IMPORT_ERROR = None
except Exception as e:
    # Keep the Flux UI usable even if qwen files/deps are not present yet.
    QwenImageTransformer2DModel = None
    _QWEN_IMPORT_ERROR = e

QWEN_MODEL_PAIRS = {
    # from: https://huggingface.co/Qwen/Qwen-Image
    # default one
    "Qwen/Qwen-Image": "Qwen/Qwen-Image",

    # from: https://huggingface.co/Qwen/Qwen-Image-2512
    "Qwen/Qwen-Image-2512": "Qwen/Qwen-Image-2512",
}

# key: transformer, value: base model
FLUX_MODEL_PAIRS = {
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
    #"ads4ow2o2/Fluxmania-Kreamania-ckpt": "black-forest-labs/FLUX.1-dev"
    # ^ somehow outputs something like a noise image, need to figure out why first

    # 20260526 from: https://civitai.red/models/1841315/candied-krea-version-10-realism?modelVersionId=2083736
    'ads4ow1o5/Candied-Krea-v1-ckpt': 'black-forest-labs/FLUX.1-Krea-dev',

    # 20260526 from: https://civitai.red/models/1841315/candied-krea-version-10-realism?modelVersionId=2086740
    'ads4ow1o5/Candied-Krea-v1-TURBO-ckpt': 'black-forest-labs/FLUX.1-Krea-dev'
}

MODEL_PAIRS = {
    'Flux': FLUX_MODEL_PAIRS,
    'Qwen': QWEN_MODEL_PAIRS
}

DEFAULT_MODEL_TYPE = "Flux"
DEFAULT_CHOICE = "black-forest-labs/FLUX.1-Krea-dev"

TITLE = "DyPE (Dynamic Position Extrapolation) — Gradio UI"
DESCRIPTION = """
Ultra-high resolution text-to-image generation using **DyPE** on **FLUX.1-Krea-dev**.

- Toggle **DyPE** and choose **position method** (`yarn` / `ntk` / `base`).
- Choose **resolution**, **steps**, **guidance**, and **seed**.
- If the model is gated on Hugging Face, paste your **HF token**.
- Outputs are saved under `./outputs/` with informative filenames.
"""

DEFAULT_PROMPT = "A mysterious woman stands confidently in elaborate, dark armor adorned with intricate designs, holding a staff, against a backdrop of smoke and an ominous red sky, with shadowy, gothic buildings in the distance."

THEME = gr.themes.Ocean(
    primary_hue="blue",
    secondary_hue="violet",
    radius_size="lg",
)

# Global cache so we don't reload every click
_PIPELINE = None
_PIPELINE_KEY: Tuple[str, str, str, bool, str, str] | None = None  # (model_type, base, ckpt, use_dype, method, dtype_opt)

def notify(msg: str):
    print(msg)
    gr.Info(msg)

def _download_models(api: HfApi, repo_ckpt: str, repo_base: str, token: str | None = None):
    # same repo, just snapshot_download
    if (repo_ckpt == repo_base):
        base_path = snapshot_download(repo_id=repo_base, repo_type='model', token=token or None)
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
        ckpt_path = hf_hub_download(repo_ckpt, ckpt_filename, repo_type='model', token=token or None)
        base_path = snapshot_download(repo_id=repo_base, repo_type='model', token=token or None)

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

def _get_flux_pipeline(repo_base, repo_ckpt, hf_token, enable_dype, method, dtype_opt):
    global _PIPELINE, _PIPELINE_KEY

    key = ("Flux", repo_base, repo_ckpt, enable_dype, method, dtype_opt)
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
        api = HfApi(token=hf_token or None)
    except Exception as e:
        msg = f"HF API init failed: {e}"
        print(msg)
        gr.Warning(msg)
        api = HfApi()

    device = _pick_device()
    dtype = _pick_dtype(device, dtype_opt)

    ckpt_path, base_path = _download_models(api, repo_ckpt, repo_base, hf_token or None)
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
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pipe.to(device)

    _PIPELINE = pipe
    _PIPELINE_KEY = key
    return pipe


def _get_model_pairs(model_type: str):
    return MODEL_PAIRS.get(model_type, FLUX_MODEL_PAIRS)


def _get_default_model_for_type(model_type: str):
    pairs = _get_model_pairs(model_type)

    if model_type == DEFAULT_MODEL_TYPE and DEFAULT_CHOICE in pairs:
        return DEFAULT_CHOICE

    return next(iter(pairs), None)


def _normalize_qwen_method(method: str, enable_dype: bool = True) -> tuple[str, bool]:
    # Qwen's reference script only has two modes: dype/base.
    # For Qwen, the method dropdown is the source of truth; the Flux-only
    # Enable DyPE checkbox is hidden when Qwen is selected.
    if method == "base":
        return "base", False
    return "dype", True



def _patch_qwen_transformer_txt_seq_lens(transformer):
    """
    Compatibility patch for DyPE Qwen transformer copies based on older
    diffusers/Qwen-Image code.

    Newer diffusers QwenImagePipeline passes encoder_hidden_states_mask to
    the transformer and does not pass the deprecated txt_seq_lens argument.
    Some DyPE Qwen transformer copies still call:
        self.pos_embed(img_shapes, txt_seq_lens, ...)
    and crash when txt_seq_lens is None.

    This wrapper reconstructs txt_seq_lens from encoder_hidden_states_mask
    when available, otherwise falls back to encoder_hidden_states.shape[1].
    """
    if getattr(transformer, "_dype_txt_seq_lens_patched", False):
        return transformer

    original_forward = transformer.forward

    def forward_with_txt_seq_lens(*args, **kwargs):
        if kwargs.get("txt_seq_lens") is None:
            mask = kwargs.get("encoder_hidden_states_mask")
            encoder_hidden_states = kwargs.get("encoder_hidden_states")

            if mask is not None:
                # mask shape: [batch, text_sequence_length], 1 for valid tokens
                kwargs["txt_seq_lens"] = mask.sum(dim=1).to(torch.int64).detach().cpu().tolist()
            elif encoder_hidden_states is not None:
                # diffusers may set mask to None when all tokens are valid.
                # In that case, the whole sequence length is valid.
                kwargs["txt_seq_lens"] = [encoder_hidden_states.shape[1]] * encoder_hidden_states.shape[0]

        return original_forward(*args, **kwargs)

    transformer.forward = forward_with_txt_seq_lens
    transformer._dype_txt_seq_lens_patched = True
    return transformer

def _get_qwen_pipeline(model_name: str, hf_token, enable_dype, method, dtype_opt):
    global _PIPELINE, _PIPELINE_KEY

    qwen_method, use_dype = _normalize_qwen_method(method, enable_dype)
    key = ("Qwen", model_name, model_name, use_dype, qwen_method, dtype_opt)

    if _PIPELINE is not None and _PIPELINE_KEY == key:
        msg = f'Using cached pipeline: {key} ...'
        notify(msg)
        return _PIPELINE

    if (_PIPELINE is not None) and (_PIPELINE_KEY != key):
        try:
            _PIPELINE.to("cpu")
        except Exception:
            pass
        _PIPELINE = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if QwenImageTransformer2DModel is None:
        raise gr.Error(
            "Could not import dype_qwen.transformer_qwenimage.QwenImageTransformer2DModel. "
            f"Original import error: {_QWEN_IMPORT_ERROR}"
        )

    device = _pick_device()
    dtype = _pick_dtype(device, dtype_opt)

    notify(f"Loading Qwen transformer from {model_name} (method={qwen_method})...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        torch_dtype=dtype,
        dype=use_dype,
        token=hf_token or None,
    )
    transformer = _patch_qwen_transformer_txt_seq_lens(transformer)

    notify(f"Loading Qwen pipeline from {model_name}...")
    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        transformer=transformer,
        torch_dtype=dtype,
        token=hf_token or None,
    )
    pipe = pipe.to(device)

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
    model_type: str,
    model: str,
    randomize_seed: bool
):
    pairs = _get_model_pairs(model_type)
    if not model or model not in pairs:
        raise gr.Error(f"No valid model selected for model type: {model_type}")

    repo_ckpt = model
    repo_base = pairs[model]

    # random seed, -ve seed also means random
    used_seed = int(seed)
    if randomize_seed or used_seed < 0:
        used_seed = next_seed()

    device = _pick_device()
    try:
        generator = torch.Generator(device).manual_seed(used_seed)
    except Exception:
        generator = torch.Generator().manual_seed(used_seed)

    os.makedirs("outputs", exist_ok=True)
    ts = str(int(time.time()))

    if model_type == "Qwen":
        qwen_method, _ = _normalize_qwen_method(method, enable_dype)
        pipe = _get_qwen_pipeline(repo_ckpt, hf_token, enable_dype, method, dtype_opt)

        # Same Qwen "magic" suffix as run_dype_qwen.py
        positive_magic = ", Ultra HD, 4K, cinematic composition."
        full_prompt = prompt + positive_magic
        negative_prompt = " "

        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=int(steps),
            true_cfg_scale=guidance_scale,
            generator=generator,
        ).images[0]

        method_name = qwen_method
    else:
        if method not in ["yarn", "ntk", "base"]:
            method = "yarn"

        pipe = _get_flux_pipeline(repo_base, repo_ckpt, hf_token, enable_dype, method, dtype_opt)

        image = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=int(steps),
        ).images[0]

        method_name = f"dy_{method}" if enable_dype else method

    filename = f"outputs/{model_type.lower()}_seed_{used_seed}_method_{method_name}_res_{width}x{height}_{ts}.png"
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

def _format_model_info_markdown(model_type: str, choice_key: str | None, token: str | None=None):
    if not choice_key:
        return f"# {model_type}\n\nNo models are defined for this model type yet."

    pairs = _get_model_pairs(model_type)
    base_model = pairs.get(choice_key)

    if not base_model:
        return f"# {choice_key}\n\nERROR: This model is not defined under `{model_type}`."

    markdown = f'# {choice_key}' + '\n\n'

    if (choice_key == base_model):
        return markdown + f"Model url: https://huggingface.co/{base_model}"

    url = hf_hub_url(choice_key, 'README.md')
    r = req.get(
        url,
        headers={} if (not token) else {
            "Authorization": f"Bearer {token}"
        }
    )

    if (r.status_code == 200):
        return markdown + r.text
    elif (r.status_code == 404):
        return markdown + f'ERROR: No README.md found in this repo'
    else:   # usually this is 401
        return markdown + f"ERROR: Unable to fetch content of README.md in `{choice_key}`, you need to provide a valid token"


def _update_model_info_markdown(model_type: str, choice_key: str, token: str | None=None):
    return _format_model_info_markdown(model_type, choice_key, token)


def _method_update_for_model_type(model_type: str):
    if model_type == "Qwen":
        return gr.update(choices=["dype", "base"], value="dype", label="Qwen method")
    return gr.update(choices=["yarn", "ntk", "base"], value="yarn", label="Position method")


def _enable_dype_update_for_model_type(model_type: str):
    if model_type == "Qwen":
        return gr.update(value=True, visible=False)
    return gr.update(value=True, visible=True, label="Enable DyPE")


def _update_model_type(model_type: str, token: str | None=None):
    pairs = _get_model_pairs(model_type)
    default_model = _get_default_model_for_type(model_type)

    return (
        gr.update(choices=list(pairs.keys()), value=default_model),
        _format_model_info_markdown(model_type, default_model, token),
        _method_update_for_model_type(model_type),
        _enable_dype_update_for_model_type(model_type),
    )

with gr.Blocks(title=TITLE, fill_height=True, theme=THEME) as demo:
    gr.Markdown(f"# {TITLE}")

    with gr.Row():
        gr.Markdown(DESCRIPTION)
        md = gr.Markdown(_format_model_info_markdown(DEFAULT_MODEL_TYPE, DEFAULT_CHOICE))

    with gr.Row():
        model_type = gr.Dropdown(
            label="Model type",
            choices=list(MODEL_PAIRS.keys()),
            value=DEFAULT_MODEL_TYPE,
        )
        model = gr.Dropdown(
            label='Model (Use the default one, the other ones are test)',
            choices=list(_get_model_pairs(DEFAULT_MODEL_TYPE).keys()),
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

    #model.change(_update_dropdown_title, inputs=model, outputs=model)
    model_type.change(_update_model_type, inputs=[model_type, hf_token], outputs=[model, md, method, enable_dype])
    model.change(_update_model_info_markdown, inputs=[model_type, model, hf_token], outputs=md)
    roll_btn.click(fn=next_seed, inputs=None, outputs=[seed])

    submit.click(
        fn=generate,
        inputs=[prompt, height, width, steps, seed, method, enable_dype, guidance, hf_token, dtype_opt, model_type, model, randomize_seed],
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