# =================================================================
# 🧩 PACURE LABS - DYNAMIC EXTENSION LOADER
# =================================================================
import os
import importlib.util
import logging

def cargar_extensiones_pacure():
    # Buscamos la carpeta 'extencions' en el directorio actual
    ext_dir = os.path.join(os.getcwd(), "extencions")
    
    if not os.path.exists(ext_dir):
        os.makedirs(ext_dir)
        logging.info(f"📁 [PACURE LABS] Carpeta 'extencions' creada.")
        return

    archivos = [f for f in os.listdir(ext_dir) if f.endswith(".py")]
    
    if not archivos:
        logging.info("ℹ️ [PACURE LABS] No hay extensiones adicionales en /extencions.")
        return

    logging.info(f"🔌 [PACURE LABS] Cargando {len(archivos)} extensiones...")

    for archivo in archivos:
        nombre_ext = archivo[:-3]
        ruta_completa = os.path.join(ext_dir, archivo)
        
        try:
            spec = importlib.util.spec_from_file_location(nombre_ext, ruta_completa)
            modulo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modulo)
            logging.info(f"✅ Extensión '{nombre_ext}' inyectada con éxito.")
        except Exception as e:
            logging.error(f"❌ Error en extensión {archivo}: {e}")