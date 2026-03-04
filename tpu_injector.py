# =================================================================
# 💉 TPU / I SEE PACURE v.3.4 - INYECTOR DE COMPATIBILIDAD XLA
# =================================================================
import torch
import sys

def activar_modo_tpu():
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        
        # 1. Redirigimos la detección de CUDA a la TPU
        torch.cuda.is_available = lambda: True
        torch.cuda.device_count = lambda: 1
        torch.cuda.current_device = lambda: 0
        torch.cuda.get_device_name = lambda x: "Google TPU v5e"
        
        # 2. Forzamos que .cuda() envíe los datos a la TPU
        # Esto arregla clip.py y attention.py sin tocarlos
        torch.Tensor.cuda = lambda self, *args, **kwargs: self.to(device)
        torch.nn.Module.cuda = lambda self, *args, **kwargs: self.to(device)
        
        # 3. Parche para tipos de datos (bfloat16 es mejor en TPU)
        # Muchos archivos de attention usan float16 por CUDA, lo forzamos a bfloat16
        original_to = torch.Tensor.to
        def smart_to(self, *args, **kwargs):
            if args and args[0] == torch.float16:
                return original_to(self, torch.bfloat16, **kwargs)
            return original_to(self, *args, **kwargs)
        torch.Tensor.to = smart_to

        print("🚀 [PACURE] Sistema CUDA redirigido a TPU exitosamente.")
    except ImportError:
        print("⚠️ [PACURE] torch_xla no instalado. Usando hardware por defecto.")

activar_modo_tpu()