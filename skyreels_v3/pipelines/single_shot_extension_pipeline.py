import gc
import logging
import os
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm

from ..modules import get_text_encoder, get_transformer, get_vae
from ..scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..utils.util import get_video_info

def split_m_n(m, n):
    """Divide la duración total m en fragmentos de tamaño n."""
    result = []
    while m >= n:
        result.append(n)
        m -= n
    if m > 0:
        result.append(m)
    return result

class SingleShotExtensionPipeline:
    """
    Pipeline para la extensión de video de una sola toma (single-shot).
    Genera extensiones de video de forma iterativa basándose en los últimos frames generados.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        weight_dtype=torch.bfloat16,
        use_usp=False,
        offload=False,
        low_vram=False,
    ):
        offload = offload or low_vram
        self.low_vram = low_vram
        self.device = device
        self.offload = offload
        load_device = "cpu" if offload else device

        # 1. Componentes base
        self.transformer = get_transformer(
            model_path,
            subfolder="transformer",
            device=load_device,
            weight_dtype=weight_dtype,
            low_vram=low_vram,
        )
        
        vae_model_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        self.vae = get_vae(vae_model_path, device=device, weight_dtype=torch.float32)
        
        self.text_encoder = get_text_encoder(
            model_path, device=load_device, weight_dtype=weight_dtype
        )
        
        self.video_processor = VideoProcessor(vae_scale_factor=16)
        self.sp_size = 1
        self.use_usp = use_usp

        # 2. Configuración USP (Ultra Sequence Parallel)
        if self.use_usp:
            import types
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.context_parallel_for_extension import (
                usp_attn_forward, usp_dit_forward,
            )

            for block in self.transformer.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            
            self.transformer.forward = types.MethodType(usp_dit_forward, self.transformer)
            self.sp_size = get_sequence_parallel_world_size()

        # 3. Scheduler y Stride
        self.scheduler = FlowUniPCMultistepScheduler()
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)

        # Offloading inicial
        self.vae.to(self.device)
        if self.offload:
            self.text_encoder.to("cpu")
            self.transformer.to("cpu")
        else:
            self.text_encoder.to(self.device)
            self.transformer.to(self.device)

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1.0

    def extend_video(
        self,
        raw_video: str,
        prompt: str,
        duration: int,
        seed: int,
        fps: int = 24,
        resolution: str = "720P",
    ):
        num_condition_frames = 25 # Frames de referencia para continuidad
        factor_num_frames = 6
        
        # Cargar video inicial
        prefix_video, _, height, width = get_video_info(
            raw_video, num_condition_frames, resolution
        )

        generatetime_list = split_m_n(duration, 5) # Divide la tarea en bloques
        output_video_frames = []
        
        current_prefix = prefix_video.to(self.device)
        padding_frames = 0

        for i, gen_time in enumerate(generatetime_list):
            latent_num_frames = factor_num_frames * gen_time
            
            # Codificar el prefijo actual (los últimos frames conocidos)
            prefix_latents = self.vae.encode(current_prefix)
            prefix_shape = prefix_latents.shape[2]
            
            # Ajuste de frames para alineación del VAE (múltiplos de 8)
            rest_frames = (latent_num_frames + prefix_shape) % 8
            if rest_frames > padding_frames:
                padding_frames = padding_frames + (8 - rest_frames)
                latent_num_frames = latent_num_frames - rest_frames + 8
            else:
                padding_frames = padding_frames - rest_frames
                latent_num_frames = latent_num_frames - rest_frames

            logging.info(f"Procesando bloque {i+1}/{len(generatetime_list)} | Frames latentes: {latent_num_frames}")

            # Generación por difusión
            video_frames = self.__call__(
                prompt=prompt,
                negative_prompt="",
                width=width,
                height=height,
                num_frames=latent_num_frames,
                num_inference_steps=8,
                guidance_scale=1.0,
                shift=8.0,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                block_offload=self.low_vram,
                latent_num_frames=latent_num_frames,
                condition=prefix_latents,
            )[0]

            # Guardar resultado (quitando el prefijo para no duplicar frames)
            output_video_frames.append(video_frames[num_condition_frames:])
            
            # Preparar el prefijo para el siguiente bloque (usando el final del video actual)
            last_frames = video_frames[-num_condition_frames:]
            current_prefix = torch.from_numpy(last_frames).unsqueeze(0).permute(0, 4, 1, 2, 3).float()
            current_prefix = current_prefix / (255.0 / 2.0) - 1.0 # Normalizar a [-1, 1]
            current_prefix = current_prefix.to(self.device)

        # Concatenar todos los bloques generados
        return np.concatenate(output_video_frames, axis=0)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        width: int = 544,
        height: int = 960,
        num_frames: int = 97,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        shift: float = 5.0,
        generator: Optional[torch.Generator] = None,
        block_offload: bool = False,
        **kwargs,
    ):
        self._guidance_scale = guidance_scale
        
        # 1. Encode Text
        if self.offload:
            self.text_encoder.to(self.device)
        
        context = self.text_encoder.encode(prompt).to(self.device)
        context_null = self.text_encoder.encode(negative_prompt).to(self.device) if self.do_classifier_free_guidance else None

        if self.offload:
            self.text_encoder.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        # 2. Shape Handling
        if "latent_num_frames" in kwargs:
            target_shape = (self.vae.vae.z_dim, kwargs["latent_num_frames"], height // 8, width // 8)
        else:
            target_shape = (self.vae.vae.z_dim, (num_frames - 1) // 4 + 1, height // 8, width // 8)

        # 3. Latents Initialization
        latents = torch.randn(*target_shape, device=self.device, generator=generator).unsqueeze(0)

        if self.offload and not block_offload:
            self.transformer.to(self.device)

        # 4. Diffusion Loop
        with torch.cuda.amp.autocast(dtype=self.transformer.dtype):
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, shift=shift)
            
            for t in tqdm(self.scheduler.timesteps, desc="Sampling"):
                # Concatenar condición (prefijo) si existe
                model_input = latents
                if "condition" in kwargs:
                    model_input = torch.cat([kwargs["condition"], latents], dim=2)

                # Preparar Timesteps para el Transformer
                t_vec = t.expand(1, model_input.shape[2]).to(self.device)
                if "condition" in kwargs:
                    t_vec = t_vec.clone()
                    t_vec[:, :kwargs["condition"].shape[2]] = 0 # Frames conocidos tienen t=0

                # Predicción de ruido
                if self.do_classifier_free_guidance:
                    noise_pred_cond = self.transformer(model_input, t=t_vec, context=context, block_offload=block_offload)[0]
                    noise_pred_uncond = self.transformer(model_input, t=t_vec, context=context_null, block_offload=block_offload)[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = self.transformer(model_input, t=t_vec, context=context, block_offload=block_offload)[0]

                # Recortar el ruido predicho para que coincida con los latentes (quitar parte del prefijo)
                if "condition" in kwargs:
                    noise_pred = noise_pred[:, -latents.shape[2]:]

                # Step
                latents = self.scheduler.step(noise_pred, t, latents, generator=generator)[0]

                if block_offload:
                    gc.collect()
                    torch.cuda.empty_cache()

        # 5. Decode
        if self.offload:
            self.transformer.cpu()
            torch.cuda.empty_cache()

        # Re-acoplar condición para decodificación suave
        decode_input = torch.cat([kwargs["condition"], latents], dim=2)[0] if "condition" in kwargs else latents[0]
        videos = self.vae.decode(decode_input)
        
        # Post-procesamiento
        videos = (videos / 2 + 0.5).clamp(0, 1)
        videos = [v.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8) * 255 for v in videos]

        if self.offload:
            gc.collect()
            torch.cuda.empty_cache()
            
        return videos