<div align="center">

# 🌙 Low-Light Image Denoising & 4× Super-Resolution ✨
### 🖼️ Fine-tuning Real-ESRGAN's RRDBNet for dark, noisy, low-resolution images

**Deep Learning Practice (DLP) · IIT Madras BS Degree Program**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Task](https://img.shields.io/badge/Task-Denoise_+_4×_SR-success?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-RRDBNet-blueviolet?style=for-the-badge)
![Leaderboard](https://img.shields.io/badge/Leaderboard_PSNR-38.42_dB-brightgreen?style=for-the-badge)

</div>

---

## 🎯 Problem Statement

Low-light photos suffer from two big problems 🌑:

- 📡 **Noise** — sensor noise and photon shot noise under poor illumination
- 🔍 **Low resolution** — dark-condition captures lose detail

**Task:** given a **low-resolution noisy** image, produce a **clean, 4× super-resolved** output that is sharp, detailed and visually appealing ✨

- 🧼 Remove noise while **preserving edges and textures**
- 📐 Upscale by **4×** in both height and width

---

## 📦 Dataset

| 📂 Split | 🌫️ Input (LR noisy) | 🏞️ Target (HR clean) | 🔢 Count |
|---|---|---|---|
| 🏋️ **Train** | `train/train/` | `train/gt/` | **1,105** pairs |
| 🧪 **Validation** | `val/val/` | `val/gt/` | **267** pairs |
| 🚀 **Test** | `test/` | 🔒 hidden | **60** images |

📏 Example: LR **160 × 256** → SR **640 × 1024**

---

## 📏 Evaluation Metric — PSNR

$$\text{PSNR} = 20 \cdot \log_{10}\left(\frac{255}{\sqrt{\text{MSE}}}\right) \; \text{dB}$$

📈 **Higher is better.**

---

## 🏆 Results

| 📊 Metric | 🎯 Score |
|---|---|
| Mean validation PSNR (10 val images) | **38.31 dB** |
| Validation range (min – max) | 37.13 – 39.66 dB |
| 🥇 **Leaderboard PSNR** | **38.42 dB** |
| Final training L1 loss | ~0.0101 |

---

## 🧠 Approach

| 🎛️ Component | ✅ Choice | 💭 Why |
|---|---|---|
| 🏗️ **Architecture** | **RRDBNet** (23 RRDB blocks, 16.7M params) | Strong SR backbone from Real-ESRGAN |
| 🎓 **Init** | `RealESRGAN_x4plus` pretrained weights | Transfer learning from natural-image SR |
| 🔧 **Fine-tuning** | On the 1,105 domain LR/HR pairs | Adapts the model to low-light noise |
| 📉 **Loss** | L1 (MAE) | Directly optimises for PSNR |
| ⚙️ **Optimizer** | Adam, `lr = 1e-4` | Standard for SR fine-tuning |
| 🌀 **Scheduler** | Cosine annealing → `1e-6` | Smooth LR decay |
| 🔁 **Epochs** | 45 | Loss plateaus ~0.010 |
| ✂️ **Patches** | 64×64 LR random crops (256×256 HR) | Memory efficient |
| 🔄 **Augmentation** | Random horizontal / vertical flips | More variety from small data |
| 🧱 **Inference** | Tile-based with overlap blending | Avoids CUDA out-of-memory |

### 🧬 Network layout

```text
Conv_first → 23 × RRDB → Conv_body → (global residual)
          → Upsample 2× + Conv → Upsample 2× + Conv → HR Conv → Conv_last
```

Each **RRDB** = 3 × **Residual Dense Blocks** with residual scaling (β = 0.2) 🔗

---

## 🛠️ Pipeline

```text
🌑 Low-light LR image (noisy)
   │
   ▼
📥 Load → RGB → normalise to [0, 1]
   │
   ▼
🧠 RRDBNet (fine-tuned, 4× SR)   ← tile-based inference
   │
   ▼
🌟 Clean high-resolution output (4H × 4W)
   │
   ▼
⚫ Grayscale → flatten → subsample [::8]
   │
   ▼
📤 submission.csv  (ID, pixel_0 … pixel_N)
```

---

## 🌟 Key Features

- 🧩 **RRDBNet written from scratch** — no `basicsr` dependency, so it works on Python 3.12 + latest torchvision
- 🎓 **Pretrained → fine-tuned** for the low-light domain
- 🧱 **Tile-based inference** with overlap blending to avoid boundary artifacts
- 🛡️ **Bicubic fallback** if any image fails during inference
- 💾 **Best-checkpoint saving** during training
- ☁️ **Optional HuggingFace Hub upload** of weights + config

---

## 💡 Key Learnings

- 🎯 **Transfer learning works** — starting from Real-ESRGAN weights gave a strong base even with just ~1.1K training pairs.
- 📉 **L1 loss is PSNR-friendly** — pixel-wise loss beats perceptual/GAN losses when the metric is PSNR.
- 🧠 **Fine-tuning on the real domain** matters more than architecture tweaks.
- 🧱 **Tiling** makes large-image inference possible on a single Kaggle GPU.

---

## 🔮 Future Work

- 🔄 **Test-time augmentation** (flip / rotate ensembles)
- 🧪 Try newer backbones (**SwinIR**, **HAT**, **Restormer**)
- 🌈 Add **SSIM / perceptual loss** alongside L1
- 📈 Evaluate on the **full validation set** (only 10 images used here)
- 🤝 Model ensembling / EMA weights

---

## 🚀 How to Run

1. 📥 Open the notebook on **Kaggle** and add the competition dataset
2. ⚙️ Settings → Accelerator → **GPU** 🎮
3. 🌐 Enable **Internet** (needed to download `RealESRGAN_x4plus.pth`, ~63 MB)
4. ▶️ **Run All** — outputs `submission.csv`, SR images and `best_model.pth`

```bash
pip install opencv-contrib-python requests Pillow huggingface_hub
```

---

## 🗂️ Repository Structure

```text
📦 DLP-LowLight-Denoising-SR-RRDBNet
 ┣ 📔 dlp-lowlight-denoise-sr-rrdbnet.ipynb
 ┗ 📄 README.md
```

---

<div align="center">

### 👨‍💻 Author
**Saini** · (CyberSoul)🎓

⭐ If this helped you, drop a star on the repo! ⭐

*Made with ❤️, ☕ and a lot of GPU hours* 🔥

</div>
