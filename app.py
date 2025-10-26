import os, sys, argparse, time

from typing import Optional, Tuple

import torch
import gradio as gr

from flux.transformer_flux import FluxTransformer2DModel  # type: ignore
from flux.pipeline_flux import FluxPipeline  # type: ignore

try:
    from huggingface_hub import login as hf_login
except Exception:
    hf_login = None


TITLE = "DyPE (Dynamic Position Extrapolation) • FLUX.1-Krea-dev — Gradio UI"
DESCRIPTION = """
Ultra-high resolution text-to-image generation using **DyPE** on **FLUX.1-Krea-dev**.

- Toggle **DyPE** and choose **position method** (`yarn` / `ntk` / `base`).
- Choose **resolution**, **steps**, **guidance**, and **seed**.
- If the model is gated on Hugging Face, paste your **HF token**.
- Outputs are saved under `./outputs/` with informative filenames.
"""

DEFAULT_PROMPT = "A mysterious woman stands confidently in elaborate, dark armor adorned with intricate designs, holding a staff, against a backdrop of smoke and an ominous red sky, with shadowy, gothic buildings in the distance."

# Global cache so we don't reload every click
_PIPELINE = None
_PIPELINE_KEY: Tuple[bool, str, str] | None = None  # (use_dype, method, dtype_opt)


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


def load_pipeline(use_dype: bool, method: str, hf_token: Optional[str], dtype_opt: str, model: str):
    global _PIPELINE, _PIPELINE_KEY

    key = (use_dype, method, dtype_opt)
    if _PIPELINE is not None and _PIPELINE_KEY == key:
        return _PIPELINE

    if hf_token and hf_login is not None:
        try:
            hf_login(token=hf_token)
        except Exception as e:
            print(f"[WARN] HF login failed: {e}")

    device = _pick_device()
    dtype = _pick_dtype(device, dtype_opt)

    # Load transformer with DyPE toggles/method
    transformer = FluxTransformer2DModel.from_pretrained(
        #"black-forest-labs/FLUX.1-Krea-dev",
        model,
        subfolder="transformer",
        torch_dtype=dtype,
        dype=use_dype,
        method=method,
    )

    pipe = FluxPipeline.from_pretrained(
        #"black-forest-labs/FLUX.1-Krea-dev",
        model,
        transformer=transformer,
        torch_dtype=dtype,
    )

    # Try to enable offload (saves VRAM). Fallback to moving to device.
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        try:
            if device != "cpu":
                pipe.to(device)
        except Exception as e:
            print(f"[WARN] Could not move pipeline to device: {e}")

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
    model: str
):
    pipe = load_pipeline(use_dype=enable_dype, method=method, hf_token=hf_token or None, dtype_opt=dtype_opt, model=model)

    device = _pick_device()
    try:
        generator = torch.Generator(device).manual_seed(int(seed))
    except Exception:
        generator = torch.Generator().manual_seed(int(seed))

    os.makedirs("outputs", exist_ok=True)

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
    filename = f"outputs/seed_{seed}_method_{method_name}_res_{width}x{height}_{ts}.png"
    image.save(filename)

    return image, filename


with gr.Blocks(title=TITLE, fill_height=True) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        model = gr.Dropdown(
            label='Model (Use the default one, the other ones are test)',
            choices=[
                "black-forest-labs/FLUX.1-Krea-dev",
                # uncensored version from: https://huggingface.co/aoxo/flux.1dev-abliterated
                # it can generate images but not sure if this is 100% working tho
                "aoxo/flux.1dev-abliterated"
            ],
            value="black-forest-labs/FLUX.1-Krea-dev"
        )
        hf_token = gr.Textbox(label="Hugging Face token (if gated)", type="password", placeholder="hf_... (optional)")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=4, autofocus=True)

    with gr.Row():
        width = gr.Slider(512, 8192, value=4096, step=64, label="Width (px)")
        height = gr.Slider(512, 8192, value=4096, step=64, label="Height (px)")

    with gr.Row():
        steps = gr.Slider(1, 64, value=28, step=1, label="Inference steps")
        guidance = gr.Slider(0.0, 10.0, value=4.5, step=0.1, label="Guidance scale")

    with gr.Row():
        seed = gr.Number(value=42, precision=0, label="Seed")
        method = gr.Dropdown(choices=["yarn", "ntk", "base"], value="yarn", label="Position method")
        dtype_opt = gr.Dropdown(choices=["auto", "bf16", "fp16", "fp32"], value="auto", label="Torch dtype")
        enable_dype = gr.Checkbox(value=True, label="Enable DyPE")

    submit = gr.Button("🚀 Generate", variant="primary")
    out_img = gr.Image(label="Result", interactive=False)
    out_file = gr.File(label="Saved image (.png)")

    submit.click(
        fn=generate,
        inputs=[prompt, height, width, steps, seed, method, enable_dype, guidance, hf_token, dtype_opt, model],
        outputs=[out_img, out_file],
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
