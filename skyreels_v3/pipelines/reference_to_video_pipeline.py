import gc
import html
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ftfy
import numpy as np
import regex as re
import torch
from diffusers import UniPCMultistepScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from PIL import Image, ImageOps
from torchvision.transforms import functional as F
from transformers import AutoTokenizer, UMT5EncoderModel

from skyreels_v3.modules.reference_to_video.transformer import SkyReelsA2WanI2v3DModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

MAX_ALLOWED_REF_IMG_LENGTH = 4

# --- Funciones de Utilidad ---

def basic_clean(text):
    """Limpieza básica de texto: arregla codificación y entidades HTML."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    """Elimina espacios en blanco redundantes."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def prompt_clean(text):
    """Pipeline completo de limpieza para prompts."""
    text = whitespace_clean(basic_clean(text))
    return text

def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    """Extrae latentes de la salida del encoder."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("No se pudo acceder a los latentes de la salida del encoder proporcionada.")

def resize_ref_images(ref_imgs, size):
    """Redimensiona imágenes de referencia manteniendo la relación de aspecto y añadiendo padding."""
    h, w = size[1], size[0]
    ref_images = []
    for img in ref_imgs:
        img = img.convert("RGB")
        img_ratio = img.width / img.height
        target_ratio = w / h

        if img_ratio > target_ratio:
            new_width = w
            new_height = int(new_width / img_ratio)
        else:
            new_height = h
            new_width = int(new_height * img_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        delta_w = w - img.size[0]
        delta_h = h - img.size[1]
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2),
        )
        new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))
        ref_images.append(new_img)
    return ref_images

# --- Clases de Pipeline ---

class WanSkyReelsA2WanT2VPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = (2 ** sum(self.vae.temperal_downsample) if self.vae else 4)
        self.vae_scale_factor_spatial = (2 ** len(self.vae.temperal_downsample) if self.vae else 8)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        return prompt_embeds

    def encode_prompt(self, prompt, negative_prompt=None, do_classifier_free_guidance=True, num_videos_per_prompt=1, prompt_embeds=None, negative_prompt_embeds=None, max_sequence_length=226, device=None, dtype=None):
        device = device or self._execution_device
        if prompt is not None:
            batch_size = len([prompt]) if isinstance(prompt, str) else len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(prompt=prompt, num_videos_per_prompt=num_videos_per_prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds = self._get_t5_prompt_embeds(prompt=negative_prompt, num_videos_per_prompt=num_videos_per_prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype)

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(self, prompt, negative_prompt, height, width, prompt_embeds=None, negative_prompt_embeds=None, callback_on_step_end_tensor_inputs=None):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` y `width` deben ser divisibles por 16.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("No se puede pasar `prompt` y `prompt_embeds` al mismo tiempo.")

    def prepare_latents(self, image_vae, batch_size, num_channels_latents=16, height=480, width=832, num_frames=81, dtype=None, device=None, generator=None, latents=None):
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        ref_vae_latents = []
        for ref_image in image_vae:
            ref_image = F.to_tensor(ref_image).sub_(0.5).div_(0.5).to(device)
            img_vae_latent = self.vae.encode(ref_image.unsqueeze(1).unsqueeze(0))
            img_vae_latent = retrieve_latents(img_vae_latent, generator)
            
            l_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, latents.dtype)
            l_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, latents.dtype)
            img_vae_latent = (img_vae_latent - l_mean) * l_std
            ref_vae_latents.append(img_vae_latent)

        while len(ref_vae_latents) < MAX_ALLOWED_REF_IMG_LENGTH:
            empty_latent = torch.zeros(1, num_channels_latents, 1, latent_height, latent_width).to(device, dtype)
            ref_vae_latents.append(empty_latent)

        ref_vae_latents = torch.cat(ref_vae_latents, dim=2).repeat(batch_size, 1, 1, 1, 1)
        return latents, ref_vae_latents

    @property
    def _execution_device(self):
        return f"cuda:{os.environ.get('LOCAL_RANK', 0)}"

    @torch.no_grad()
    def __call__(self, ref_imgs, prompt=None, negative_prompt=None, height=544, width=960, num_frames=105, num_inference_steps=50, guidance_scale=7.5, guidance_scale_img=5.0, num_videos_per_prompt=1, generator=None, latents=None, prompt_embeds=None, negative_prompt_embeds=None, output_type="np", return_dict=True, attention_kwargs=None, callback_on_step_end=None, callback_on_step_end_tensor_inputs=["latents"], max_sequence_length=512, offload=False, block_offload=False, **kwargs):
        if offload: self.text_encoder.to(self._execution_device)
        self._guidance_scale = guidance_scale
        device = self._execution_device
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt, guidance_scale > 1, num_videos_per_prompt, prompt_embeds, negative_prompt_embeds, max_sequence_length, device)
        
        if offload:
            self.text_encoder.to("cpu")
            if not block_offload: self.transformer.to(device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        latents, condition = self.prepare_latents(ref_imgs, batch_size * num_videos_per_prompt, self.vae.config.z_dim, height, width, num_frames, torch.float32, device, generator, latents)
        uncondition = torch.zeros_like(condition)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0])
                # Predicción con guía dual (Texto e Imagen)
                latent_model_input = torch.cat([latents, condition], dim=2).to(self.transformer.dtype)
                noise_pred = self.transformer(hidden_states=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds, attention_kwargs=attention_kwargs, return_dict=False)[0]
                noise_pred = noise_pred[:, :, :latents.shape[2]]

                if guidance_scale > 1:
                    noise_uncond_txt = self.transformer(hidden_states=latent_model_input, timestep=timestep, encoder_hidden_states=negative_prompt_embeds, attention_kwargs=attention_kwargs, return_dict=False)[0]
                    noise_uncond_txt = noise_uncond_txt[:, :, :latents.shape[2]]

                    noise_uncond_txt_img = self.transformer(hidden_states=torch.cat([latents, uncondition], dim=2).to(self.transformer.dtype), timestep=timestep, encoder_hidden_states=negative_prompt_embeds, attention_kwargs=attention_kwargs, return_dict=False)[0]
                    noise_uncond_txt_img = noise_uncond_txt_img[:, :, :latents.shape[2]]

                    noise_pred = noise_uncond_txt_img + guidance_scale_img * (noise_uncond_txt - noise_uncond_txt_img) + guidance_scale * (noise_pred - noise_uncond_txt)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                progress_bar.update()
                if XLA_AVAILABLE: xm.mark_step()

        if offload: self.transformer.to("cpu")

        if output_type != "latent":
            l_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            l_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents = (latents / l_std) + l_mean
            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        return WanPipelineOutput(frames=video) if return_dict else (video,)


class ReferenceToVideoPipeline:
    def __init__(self, model_path, device="cuda", weight_dtype=torch.bfloat16, use_usp=False, offload=False, low_vram=False):
        self.offload = offload or low_vram
        load_device = "cpu" if self.offload else device
        
        self.transformer = SkyReelsA2WanI2v3DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.bfloat16).to(load_device)
        self.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32).to(load_device)
        
        self.pipeline = WanSkyReelsA2WanT2VPipeline.from_pretrained(model_path, transformer=self.transformer, vae=self.vae, torch_dtype=weight_dtype).to(load_device)
        self.pipeline.scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=5.0)
        
        self.device, self.low_vram = device, low_vram
        if low_vram:
            from torchao.quantization import float8_weight_only, quantize_
            quantize_(self.pipeline.transformer, float8_weight_only(), device="cuda")
            self.pipeline.vae.enable_tiling()

        if self.offload:
            self.pipeline.vae.to(device)
            self.pipeline.transformer.to("cpu")
        else:
            self.pipeline.to(device)

    def generate_video(self, ref_imgs, prompt, duration, seed, resolution="720P"):
        from ..utils.util import get_height_width_from_image
        height, width = get_height_width_from_image(ref_imgs[0], resolution)
        ref_imgs = resize_ref_images(ref_imgs, (width, height))
        
        video_pt = self.pipeline(
            ref_imgs=ref_imgs, prompt=prompt, height=height, width=width,
            num_frames=duration * 24 + 1, guidance_scale=1.0, guidance_scale_img=1.0,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            output_type="pt", num_inference_steps=8, offload=self.offload, block_offload=self.low_vram
        ).frames

        final_images = []
        for frame in video_pt[0]:
            img_np = VaeImageProcessor.pt_to_numpy(frame.unsqueeze(0))
            img_pil = VaeImageProcessor.numpy_to_pil(img_np)[0]
            final_images.append(np.array(img_pil.convert("RGB")))
        
        torch.cuda.empty_cache()
        return final_images