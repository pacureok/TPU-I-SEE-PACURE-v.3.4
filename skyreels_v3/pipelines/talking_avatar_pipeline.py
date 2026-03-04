import os
import gc
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from functools import partial
from contextlib import contextmanager

# Importaciones específicas para TPU (PyTorch XLA)
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    # Fallback si se ejecuta fuera de un entorno TPU
    xm = None

# --- Constantes de Resolución (Wan 2.1) ---
ASPECT_RATIO_960 = {"1.0": [960, 960], "0.75": [832, 1120], "1.33": [1120, 832]}
ASPECT_RATIO_627 = {"1.0": [627, 627], "0.75": [544, 736], "1.33": [736, 544]}

# --- Utilerías de Imagen para TPU ---

def resize_and_centercrop(image, size):
    """Ajusta la imagen al tamaño del bucket de Wan utilizando interpolación bilineal."""
    if isinstance(image, Image.Image):
        w, h = image.size
        target_h, target_w = size
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - target_w) / 2
        top = (new_h - target_h) / 2
        return image.crop((left, top, left + target_w, top + target_h))
    return F.interpolate(image, size=size, mode='bilinear', align_corners=False)

def process_video_samples(video):
    """Normaliza los tensores de video de [-1, 1] a [0, 1] para la salida final."""
    return video.clamp(-1, 1).add(1).div(2)

class TalkingAvatarPipelineTPU:
    """
    Pipeline de Avatar Parlante optimizado para TPU (Tensor Processing Unit).
    Utiliza PyTorch XLA y bfloat16 para máxima eficiencia.
    """
    
    def __init__(
        self, 
        model, 
        vae, 
        text_encoder, 
        clip, 
        device=None, 
        param_dtype=torch.bfloat16, # TPU prefiere bfloat16 sobre float16
        num_timesteps=1000
    ):
        # Inicializar dispositivo TPU
        if device is None:
            self.device = xm.xla_device() if xm else torch.device("cpu")
        else:
            self.device = device
            
        self.model = model.to(self.device)
        self.vae = vae.to(self.device)
        self.text_encoder = text_encoder.to(self.device)
        self.clip = clip.to(self.device)
        self.param_dtype = param_dtype
        self.num_timesteps = num_timesteps
        
        # Identificar si es el proceso maestro en ejecución distribuida
        self.is_master = xm.is_master_ordinal() if xm else True
        
        self.sample_neg_prompt = (
            "low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, jpeg artifacts, signature, watermark, blurry"
        )

    def _tpu_memory_clean(self):
        """Limpieza de memoria optimizada para el recolector de XLA."""
        gc.collect()
        # En TPU no existe empty_cache, se confía en el grafo y mark_step

    def add_noise(self, original_samples, noise, timesteps):
        """Implementación de Flow Matching optimizada para tensores XLA."""
        sigmas = (timesteps / self.num_timesteps).view(-1, 1, 1, 1, 1)
        return (1.0 - sigmas) * original_samples + sigmas * noise

    @torch.no_grad()
    def generate(
        self,
        input_data,
        size_bucket="720P",
        motion_frame=25,
        drop_frame=12,
        frame_num=81,
        sampling_steps=40,
        text_guide_scale=5.0,
        audio_guide_scale=4.0,
        seed=-1,
        max_frames_num=5000
    ):
        self._tpu_memory_clean()
        
        if seed == -1: seed = random.randint(0, 2**31)
        # El generador debe estar en la CPU para semillas deterministas en XLA a veces, 
        # o usar xm.get_rng_state
        generator = torch.Generator().manual_seed(seed)

        # 1. Configuración de Resolución
        bucket_config = ASPECT_RATIO_960 if size_bucket == "720P" else ASPECT_RATIO_627
        cond_image_pil = Image.open(input_data["cond_image"]).convert("RGB")
        
        src_w, src_h = cond_image_pil.size
        aspect_ratio = src_h / src_w
        closest_ratio = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x) - aspect_ratio))[0]
        target_h, target_w = bucket_config[closest_ratio]
        
        cond_image_input = resize_and_centercrop(cond_image_pil, (target_h, target_w))
        cond_image_tensor = torch.from_numpy(np.array(cond_image_input)).permute(2, 0, 1).float()
        cond_image_tensor = (cond_image_tensor / 255.0 - 0.5) * 2
        cond_image_tensor = cond_image_tensor.unsqueeze(0).unsqueeze(2).to(self.device, dtype=self.param_dtype)
        original_color_reference = cond_image_tensor.clone()

        # 2. Codificación de Prompts (XLA)
        context, context_null, connection_embedding = self.text_encoder.encode(
            [input_data["prompt"], self.sample_neg_prompt, "a person is talking"]
        )
        if xm: xm.mark_step() # Sincronizar grafo tras codificación

        # 3. Características de Audio
        audio_embs_full = torch.load(input_data["cond_audio"]["person1"]).to(self.device, dtype=self.param_dtype)
        video_length_real = min(max_frames_num, audio_embs_full.shape[0])
        
        audio_start_idx = 0
        is_first_clip = True
        gen_video_list = []
        indices = (torch.arange(5, device=self.device) - 2)
        
        lat_h, lat_w = target_h // 8, target_w // 8
        lat_t = (frame_num - 1) // 4 + 1

        # 4. Bucle Autorregresivo de Ventana Deslizante
        while True:
            audio_end_idx = audio_start_idx + frame_num
            if audio_end_idx >= video_length_real:
                audio_end_idx = video_length_real
                audio_start_idx = max(0, audio_end_idx - frame_num)

            # Segmentación de Audio
            curr_idx = torch.clamp(torch.arange(audio_start_idx, audio_end_idx, device=self.device).unsqueeze(1) + indices, 0, audio_embs_full.shape[0]-1)
            audio_embs = audio_embs_full[curr_idx.flatten()].view(frame_num, 5, -1).unsqueeze(0)

            # Preparación de Latentes (XLA)
            vae_input = torch.zeros(1, 3, frame_num, target_h, target_w, device=self.device, dtype=self.param_dtype)
            vae_input[:, :, :cond_image_tensor.shape[2]] = cond_image_tensor
            y = self.vae.encode(vae_input)
            
            ref_mask = torch.zeros(1, 1, lat_t, lat_h, lat_w, device=self.device, dtype=self.param_dtype)
            ref_mask[:, :, : (cond_image_tensor.shape[2] // 4 + 1)] = 1.0
            y = torch.cat([ref_mask, y], dim=1)

            # CLIP Visual Features
            clip_context = self.clip.visual(cond_image_tensor[:, :, :1])

            # Proceso de Denoising (Placeholder de Euler para TPU)
            latent = torch.randn(1, 16, lat_t, lat_h, lat_w, device=self.device, generator=generator).to(self.param_dtype)
            timesteps = torch.linspace(self.num_timesteps, 0, sampling_steps + 1, device=self.device)

            if not is_first_clip:
                m_noise = torch.randn_like(latent_motion_frames)
                add_lat = self.add_noise(latent_motion_frames, m_noise, timesteps[0])
                latent[:, :, :add_lat.shape[2]] = add_lat

            # Bucle de muestreo
            for i in range(len(timesteps) - 1):
                t = timesteps[i]
                # Simulación de inferencia del modelo
                # noise_pred = self.model(latent, t, context, audio_embs, y, clip_context)
                # dt = (timesteps[i] - timesteps[i+1]) / self.num_timesteps
                # latent = latent + noise_pred * dt
                if xm: xm.mark_step() # Muy importante en cada paso de sampling para evitar grafos gigantes

            # 5. Decodificación VAE
            videos = self.vae.decode(latent)
            if xm: xm.mark_step()
            
            if not (audio_end_idx == video_length_real):
                videos = videos[:, :, :-drop_frame]
            
            processed = process_video_samples(videos)
            
            if is_first_clip:
                gen_video_list.append(processed.cpu()) # Movemos a CPU para liberar memoria TPU
            else:
                gen_video_list.append(processed[:, :, motion_frame:].cpu())

            if audio_end_idx >= video_length_real: break
            
            # Actualizar estado autorregresivo
            is_first_clip = False
            latent_motion_frames = latent[:, :, -((motion_frame - 1) // 4 + 1):]
            cond_image_tensor = videos[:, :, -motion_frame:]
            audio_start_idx += (frame_num - motion_frame - drop_frame)
            
            self._tpu_memory_clean()

        # 6. Concatenación Final
        if self.is_master:
            final_v = torch.cat(gen_video_list, dim=2)[:, :, :video_length_real]
            return final_v[0].permute(1, 2, 3, 0).numpy()
        
        return None