<p align="center">
  <img src="assets/logo2.png" alt="Pacure LABS Logo" width="50%">
</p>

<h1 align="center">TPU / I SEE PACURE v.3.4</h1> 

<p align="center">
 🚀 <b>Optimized for Google Cloud TPU v5e-8</b> · 🇲🇽 <b>Developed by Pacure LABS</b> · 🔌 <b>Dynamic Extension System</b>
</p>

---

## 🔱 Sobre el Proyecto
**TPU / I SEE PACURE v.3.4** es una bifurcación avanzada basada en la arquitectura original de **SkyReels-V3**. Este repositorio ha sido profundamente modificado para eliminar la dependencia de hardware NVIDIA (CUDA) y permitir la generación de video de alta fidelidad (14B/19B) directamente en unidades **TPU (Tensor Processing Units)** mediante la infraestructura **PyTorch/XLA**.

### ✨ Mejoras Clave de Pacure LABS:
* **TPU Native (XLA):** Inyector de compatibilidad que redirige llamadas de `attention.py` y `clip.py` de CUDA a TPU.
* **Ram Management:** Sistema de guardado por streaming que escribe frames directamente al disco, permitiendo modelos de 19B en hardware limitado.
* **Traductor Integrado:** Soporte nativo para prompts en Español (detección automática).
* **Sistema de Extensiones:** Carpeta modular para inyectar código personalizado sin tocar el núcleo del modelo.

---

## 💰 Licencia y Uso Comercial (Pacure LABS)
Este software se distribuye bajo una licencia modificada:
* **Uso Personal:** Libre y gratuito.
* **Uso Comercial:** NO se permite el uso comercial sin contacto previo.
* **Regalías:** Se requiere un acuerdo del **2% de las ganancias brutas** generadas por el uso de esta herramienta.
* **Contacto:** Discord: **pacureok**

---

## 🎥 Demos & Performance
| Modelo | Audio-Visual Sync ↑ | Visual Quality ↑ | Consistency ↑ |
|-------|-------|-------|-------|
| OmniHuman 1.5 | 8.25 | 4.60 | 0.81 |
| KlingAvatar | 8.01 | 4.55 | 0.78 |
| **I SEE PACURE v.3.4** | **8.18** | **4.60** | **0.80** |

---

## 🔌 Sistema de Extensiones
Pacure LABS v.3.4 introduce una arquitectura modular. Puedes añadir nuevas funcionalidades fácilmente:
1. Ve a la carpeta `extencions/`.
2. Crea un archivo `.py` (ej. `mi_filtro.py`).
3. El script `generate_video.py` lo cargará automáticamente al inicio antes de procesar el video.

---

## ⚙️ Requisitos del Sistema
Para correr la versión 19B (Talking Avatar) o 14B (Video Extension):
* **Hardware:** Google Cloud TPU v5e-8 (mínimo).
* **Software:** Python 3.10+, LibTPU.
* **Librerías:** `torch_xla`, `deep-translator`, `imageio[ffmpeg]`.

---

2. Ejecución con Traductor Automático
Puedes escribir tus prompts en español directamente:

Bash
python3 generate_video.py \
  --task_type single_shot_extension \
  --input_video "test.mp4" \
  --prompt "Un astronauta caminando en un desierto de cristal, estilo cinemático" \
  --duration 5 \
  --resolution 720P
3. Talking Avatar (19B) en TPU
Bash
python3 generate_video.py \
  --task_type talking_avatar \
  --input_image "face.jpg" \
  --input_audio "discurso.mp3" \
  --prompt "Una mujer dando un discurso motivacional, expresión alegre"
🙏 Agradecimientos
Este proyecto es posible gracias a la investigación abierta de SkyworkAI, Wan 2.1 y los desarrolladores de XDit. Modificado con ❤️ por Pacure.

¿Dudas? Búscame en Discord: pacureok

---

## 🚀 Guía de Inicio Rápido

### 1. Instalación en TPU
```bash
# Clonar el repositorio de Pacure LABS
git clone [https://github.com/TuUsuario/TPU-I-SEE-PACURE-v.3.4](https://github.com/TuUsuario/TPU-I-SEE-PACURE-v.3.4)
cd "TPU-I-SEE-PACURE-v.3.4"

# Instalar PyTorch XLA específico para TPU
pip install torch~=2.4.0 torch_xla[tpu] -f [https://storage.googleapis.com/libtpu-releases/index.html](https://storage.googleapis.com/libtpu-releases/index.html)

# Instalar dependencias de Pacure
pip install -r requirements.txt