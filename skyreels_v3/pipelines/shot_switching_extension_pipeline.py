import gc
import logging
import os
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm

from ..config import SHOT_NUM_CONDITION_FRAMES_MAP
from ..modules import get_text_encoder, get_transformer, get_vae
from ..scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..utils.util import get_video_info

class ShotSwitchingExtensionPipeline:
    """
    Pipeline diseñado para la extensión de video mediante cambio de tomas (shot switching).
    Permite continuar un video existente basándose en frames de condición previos.
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
        """
        Inicializa los componentes del pipeline de extensión.

        Args:
            model_path (str): Ruta raíz de los checkpoints del modelo.
            device (str): Dispositivo de ejecución (ej. 'cuda').
            weight_dtype: Tipo de dato de los pesos (bfloat16 recomendado).
            use_usp: Activa el paralelismo de secuencia (Ultra Sequence Parallel).
            offload: Mueve modelos a CPU cuando no se usan para ahorrar VRAM.
            low_vram: Optimización agresiva de memoria.
        """
        self.offload = offload or low_vram
        self.low_vram = low_vram
        self.device = device
        load_device = "cpu" if self.offload else device
        
        # 1. Cargar Transformer (Específico para Shot Switching)
        self.transformer = get_transformer(
            model_path,
            subfolder="shot_transformer",
            device=load_device,
            weight_dtype=weight_dtype,
            low_vram=low_vram,
        )

        # 2. Cargar VAE (Wan 2.1)
        vae_model_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        self.vae = get_vae(vae_model_path, device=device, weight_dtype=torch.float32)

        # 3. Cargar Text Encoder (T5)
        self.text_encoder = get_text_encoder(
            model_path, device=load_device, weight_dtype=weight_dtype
        )

        self.video_processor = VideoProcessor(vae_scale_factor=16)
        self.sp_size = 1

        # Configuración de USP (Ultra Sequence Parallel)
        if use_usp:
            import types
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.context_parallel_for_extension import (
                usp_attn_forward,
                usp_dit_forward,
            )

            for block in self.transformer.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            
            self.transformer.forward = types.MethodType(usp_dit_forward, self.transformer)
            self.sp_size = get_sequence_parallel_world_size()

        # Scheduler y parámetros de arquitectura
        self.scheduler = FlowUniPCMultistepScheduler()
        self.vae_stride = (4, 8, 8)  # Temporal, Height, Width
        self.patch_size = (1, 2, 2)

        # Gestión de dispositivos inicial
        self.vae.to(self.device)
        if not self.offload:
            self.text_encoder.to(self.device)
            self.transformer.to(self.device)
        else:
            self.text_encoder.to("cpu")
            self.transformer.to("cpu")

    def extend_video(
        self,
        raw_video: str,
        prompt: str,
        duration: int,
        seed: int,
        fps: int = 24,
        resolution: str = "720P",
    ):
        """
        Función de alto nivel para extender un archivo de video.
        """
        if duration not in SHOT_NUM_CONDITION_FRAMES_MAP:
            raise ValueError(f"Duración {duration} no soportada por el mapa de frames de condición.")

        num_condition_frames = SHOT_NUM_CONDITION_FRAMES_MAP[duration]
        frames_num = duration * fps + 1

        # Obtener frames de prefijo y dimensiones
        prefix_video, _, height, width = get_video_info(
            raw_video, num_condition_frames, resolution
        )

        # Codificar video de prefijo a espacio latente
        prefix_video = prefix_video.to(self.device)
        prefix_latents = self.vae.encode(prefix_video)

        # Generar extensión
        video_frames = self.__call__(
            prompt=prompt,
            negative_prompt="",
            width=width,
            height=height,
            num_frames=frames_num,
            num_inference_steps=8,
            guidance_scale=1.0,
            shift=8.0,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            prefix_video=prefix_latents,
            block_offload=self.low_vram,
        )[0]

        logging.info(f"Video extendido generado con éxito. Shape: {video_frames.shape}")
        return video_frames

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1.0

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
        
        # 1. Procesar Texto
        if self.offload:
            self.text_encoder.to(self.device)
        
        context = self.text_encoder.encode(prompt).to(self.device)
        context_null = (
            self.text_encoder.encode(negative_prompt).to(self.device)
            if self.do_classifier_free_guidance else None
        )

        if self.offload:
            self.text_encoder.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        # 2. Preparar Latentes
        target_shape = (
            self.vae.vae.z_dim,
            (num_frames - 1) // self.vae_stride[0] + 1,
            height // self.vae_stride[1],
            width // self.vae_stride[2],
        )

        latents = torch.randn(
            *target_shape,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        ).unsqueeze(0) # Batch dimension

        prefix_video = kwargs["prefix_video"].to(self.device)
        context_frames = prefix_video.shape[2]
        total_frames = latents.shape[2] + context_frames

        # 3. Denoising Loop
        if self.offload and not block_offload:
            self.transformer.to(self.device)

        with torch.cuda.amp.autocast(dtype=self.transformer.dtype):
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, shift=shift)
            timesteps = self.scheduler.timesteps

            for t in tqdm(timesteps, desc="Extending video"):
                # Preparar entrada de tiempo con máscara para el prefijo (t=0 para frames conocidos)
                t_input = t.repeat(latents.shape[0]).unsqueeze(-1).repeat(1, total_frames)
                t_input[:, -context_frames:] = 0 

                if self.do_classifier_free_guidance:
                    noise_pred_cond = self.transformer(
                        [latents, prefix_video], t=t_input, context=context, block_offload=block_offload
                    )[0]
                    noise_pred_uncond = self.transformer(
                        [latents, prefix_video], t=t_input, context=context_null, block_offload=block_offload
                    )[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = self.transformer(
                        [latents, prefix_video], t=t_input, context=context, block_offload=block_offload
                    )
                    if isinstance(noise_pred, tuple):
                        noise_pred = noise_pred[0]

                # Paso del Scheduler
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False, generator=generator
                )[0]

                if block_offload:
                    gc.collect()
                    torch.cuda.empty_cache()

        if self.offload:
            self.transformer.cpu()
            torch.cuda.empty_cache()

        # 4. Decodificación Final
        videos = self.vae.decode(latents[0])
        videos = (videos / 2 + 0.5).clamp(0, 1)
        
        # Post-procesamiento a formato imagen/video
        final_videos = []
        for video in videos:
            # Reordenar: [C, F, H, W] -> [F, H, W, C]
            video = video.permute(1, 2, 3, 0) * 255
            final_videos.append(video.cpu().numpy().astype(np.uint8))

        if self.offload:
            gc.collect()
            torch.cuda.empty_cache()

        return final_videos