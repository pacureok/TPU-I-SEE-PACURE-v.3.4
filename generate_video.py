# =================================================================
# 🇲🇽 TPU / I SEE PACURE v.3.4 - BY PACURE LABS
# Basado en SkyReels-V3 (Modificado para TPU v5e)
# Discord: pacureok | Comercial: 2% Royalty Agreement
# =================================================================

# 1. CARGAR EXTENSIONES PRIMERO (Prioridad Pacure)
import logging
from extension_loader import cargar_extensiones_pacure
logging.basicConfig(level=logging.INFO, format="%(asctime)s - PACURE_V3.4 - %(message)s")
cargar_extensiones_pacure()

# 2. INYECTOR DE COMPATIBILIDAD TPU/CUDA
import torch
import sys
import os
import random
import time
import imageio
import numpy as np
from deep_translator import GoogleTranslator

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend
    device = xm.xla_device()
    # Parcheamos llamadas CUDA de skyreels_v3 (attention.py, clip.py, etc.)
    torch.cuda.is_available = lambda: True
    torch.Tensor.cuda = lambda self, *args, **kwargs: self.to(device)
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self.to(device)
    IS_TPU = True
    logging.info("🚀 [TPU] Motor XLA vinculado. CUDA interceptado.")
except ImportError:
    IS_TPU = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"⚠️ [HOST] torch_xla no disponible. Usando: {device}")

# 3. IMPORTS DE SKYREELS
from skyreels_v3.modules import download_model
from skyreels_v3.pipelines import (
    SingleShotExtensionPipeline,
    ShotSwitchingExtensionPipeline
)

# --- FUNCIONES CORE DE PACURE LABS ---

def traducir_prompt(texto):
    """Soporte para prompts en español."""
    try:
        if any(c in texto for c in "áéíóúñÁÉÍÓÚÑ"):
            logging.info("🇲🇽 Detectado español. Traduciendo...")
            return GoogleTranslator(source='auto', target='en').translate(texto)
        return texto
    except Exception as e:
        return texto

def guardar_video_streaming(gen_frames, path, fps=24):
    """Guarda frame a frame para no explotar la RAM de la TPU."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with imageio.get_writer(path, fps=fps, quality=9, codec='libx264') as writer:
        for i, frame in enumerate(gen_frames):
            writer.append_data(np.array(frame).astype(np.uint8))
            if i % 10 == 0:
                logging.info(f"🎞️ Procesando frame {i} físicamente en disco...")

# --- FLUJO DE EJECUCIÓN ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PACURE LABS TPU v3.4")
    parser.add_argument("--task_type", type=str, required=True, choices=["single_shot_extension", "shot_switching_extension"])
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--resolution", type=str, default="720P")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--low_vram", action="store_true")
    
    import argparse
    args = parser.parse_args()

    print("\n" + "="*50)
    print("🔱 PACURE LABS - TPU I SEE PACURE v.3.4")
    print("🔱 Contacto Discord: pacureok")
    print("🔱 Uso Comercial: Requiere contrato (2% royalties)")
    print("="*50 + "\n")

    # Descarga e Inicialización
    model_id = "Skywork/SkyReels-V3-Video-Extension"
    local_model_path = download_model(model_id)
    
    prompt_final = traducir_prompt(args.prompt)
    if args.task_type == "shot_switching_extension" and "[CUT]" not in prompt_final:
        prompt_final = f"[ZOOM_IN_CUT] {prompt_final}"

    pipe_class = SingleShotExtensionPipeline if args.task_type == "single_shot_extension" else ShotSwitchingExtensionPipeline
    
    pipe = pipe_class(
        model_path=local_model_path,
        low_vram=args.low_vram,
        device=device
    )

    if args.seed is None: args.seed = random.randint(0, 1000000)
    
    logging.info("🎬 Renderizando video...")
    video_frames = pipe.extend_video(
        input_video=args.input_video,
        prompt=prompt_final,
        duration=args.duration,
        seed=args.seed,
        resolution=args.resolution
    )

    output_file = f"result/PACURE_LABS_{args.seed}.mp4"
    guardar_video_streaming(video_frames, output_file)
    logging.info(f"✨ ¡COMPLETO! Video en: {output_file}")