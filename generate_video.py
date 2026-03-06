# =================================================================
# 🇲🇽 TPU / I SEE PACURE v.3.4 - BY PACURE LABS
# Basado en SkyReels-V3 (Modificado para TPU v5e/v3-8)
# Discord: pacureok | Comercial: 2% Royalty Agreement
# =================================================================

import logging
import torch
import sys
import os
import random
import time
import imageio
import numpy as np
import argparse
import cv2
from PIL import Image
from deep_translator import GoogleTranslator

# 1. CARGAR EXTENSIONES PRIMERO (Prioridad Pacure)
from extension_loader import cargar_extensiones_pacure
logging.basicConfig(level=logging.INFO, format="%(asctime)s - PACURE_V3.4 - %(message)s")
cargar_extensiones_pacure()

# 2. INYECTOR DE COMPATIBILIDAD TPU/CUDA
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend
    device = xm.xla_device()
    # Parcheamos llamadas CUDA de skyreels_v3 para redirigirlas a TPU
    torch.cuda.is_available = lambda: True
    torch.Tensor.cuda = lambda self, *args, **kwargs: self.to(device)
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self.to(device)
    IS_TPU = True
    logging.info("🚀 [TPU] Motor XLA vinculado. CUDA interceptado exitosamente.")
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

def apply_watermark(frame_np, logo_path):
    """Aplica marca de agua en la esquina inferior derecha."""
    if not os.path.exists(logo_path):
        return frame_np
    
    try:
        # Cargar logo con OpenCV
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is None: return frame_np
        
        # Redimensionar logo proporcionalmente (ancho de 180px por defecto)
        target_width = 180
        scale = target_width / logo.shape[1]
        logo = cv2.resize(logo, (0, 0), fx=scale, fy=scale)
        
        h_f, w_f, _ = frame_np.shape
        h_l, w_l, _ = logo.shape
        
        # Margen de 25 píxeles
        offset_y = h_f - h_l - 25
        offset_x = w_f - w_l - 25
        
        # Si tiene canal alfa, hacer blend; si no, pegar directo
        if logo.shape[2] == 4:
            alpha_l = logo[:, :, 3] / 255.0
            for c in range(0, 3):
                frame_np[offset_y:offset_y+h_l, offset_x:offset_x+w_l, c] = (
                    alpha_l * logo[:, :, c] + (1.0 - alpha_l) * frame_np[offset_y:offset_y+h_l, offset_x:offset_x+w_l, c]
                )
        else:
            frame_np[offset_y:offset_y+h_l, offset_x:offset_x+w_l] = logo
            
    except Exception as e:
        logging.warning(f"No se pudo aplicar la marca de agua: {e}")
        
    return frame_np

def guardar_video_streaming(gen_frames, path, fps=24, logo_path="assets/logo.png"):
    """Guarda frame a frame aplicando marca de agua y optimizando RAM."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with imageio.get_writer(path, fps=fps, quality=9, codec='libx264') as writer:
        for i, frame in enumerate(gen_frames):
            # Convertir frame a array de numpy
            frame_np = np.array(frame).astype(np.uint8)
            
            # Aplicar marca de agua de Pacure Labs
            frame_np = apply_watermark(frame_np, logo_path)
            
            # Escribir frame
            writer.append_data(frame_np)
            
            if i % 10 == 0:
                logging.info(f"🎞️ Procesando frame {i} con marca de agua...")
                if IS_TPU: xm.mark_step() # Limpiar grafo XLA periódicamente

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
    
    args = parser.parse_args()

    print("\n" + "="*50)
    print("🔱 PACURE LABS - TPU I SEE PACURE v.3.4")
    print("🔱 Contacto Discord: pacureok")
    print("🔱 Uso Comercial: Requiere contrato (2% royalties)")
    print("="*50 + "\n")

    # Descarga e Inicialización de pesos de SkyReels
    model_id = "Skywork/SkyReels-V3-Video-Extension"
    logging.info(f"Cargando modelo: {model_id}")
    local_model_path = download_model(model_id)
    
    # Procesamiento de Prompt
    prompt_final = traducir_prompt(args.prompt)
    if args.task_type == "shot_switching_extension" and "[CUT]" not in prompt_final:
        prompt_final = f"[ZOOM_IN_CUT] {prompt_final}"

    # Selección de Pipeline
    pipe_class = SingleShotExtensionPipeline if args.task_type == "single_shot_extension" else ShotSwitchingExtensionPipeline
    
    pipe = pipe_class(
        model_path=local_model_path,
        low_vram=args.low_vram,
        device=device
    )

    if args.seed is None: 
        args.seed = random.randint(0, 1000000)
    
    logging.info(f"🎬 Iniciando renderizado (Seed: {args.seed})...")
    
    # Inferencia Real en TPU
    video_frames = pipe.extend_video(
        input_video=args.input_video,
        prompt=prompt_final,
        duration=args.duration,
        seed=args.seed,
        resolution=args.resolution
    )

    # Guardado Final con Marca de Agua
    output_file = f"result/PACURE_LABS_{args.seed}.mp4"
    logo_file = "assets/logo.png"
    
    guardar_video_streaming(video_frames, output_file, logo_path=logo_file)
    logging.info(f"✨ ¡GENERACIÓN EXITOSA! Video guardado en: {output_file}")