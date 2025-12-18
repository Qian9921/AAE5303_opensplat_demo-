# 🏆 AAE5303 3D Gaussian Splatting Assignment - Leaderboard

<div align="center">

![3DGS](https://img.shields.io/badge/3D_Gaussian-Splatting-blue?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-HKisland-orange?style=for-the-badge)

**Rendering Quality Evaluation for 3D Gaussian Splatting**

</div>

---

## 📋 Overview

This leaderboard evaluates student submissions based on **Rendering Quality Metrics** for the 3D Gaussian Splatting assignment. All three metrics are evaluated independently without weighted scoring.

---

## 📊 Evaluation Metrics

The leaderboard uses three standard academic metrics for novel view synthesis evaluation:

### 1. PSNR (Peak Signal-to-Noise Ratio) ↑

**Higher is better**

PSNR measures the pixel-level reconstruction accuracy between rendered images and ground truth.

$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right)$$

| Range | Interpretation |
|-------|----------------|
| > 30 dB | Excellent quality |
| 25-30 dB | Good quality |
| 20-25 dB | Acceptable quality |
| < 20 dB | Poor quality |

---

### 2. SSIM (Structural Similarity Index) ↑

**Higher is better** (Range: 0 to 1)

SSIM measures perceptual quality based on structural information, luminance, and contrast.

$$SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

| Range | Interpretation |
|-------|----------------|
| > 0.95 | Excellent quality |
| 0.90-0.95 | Good quality |
| 0.85-0.90 | Acceptable quality |
| < 0.85 | Poor quality |

---

### 3. LPIPS (Learned Perceptual Image Patch Similarity) ↓

**Lower is better** (Range: 0 to 1)

LPIPS uses deep neural network features to measure perceptual similarity, correlating better with human perception.

| Range | Interpretation |
|-------|----------------|
| < 0.05 | Excellent quality |
| 0.05-0.10 | Good quality |
| 0.10-0.15 | Acceptable quality |
| > 0.15 | Poor quality |

---

## 🏅 Leaderboard Format

The leaderboard will display results in the following format:

| Rank | Student ID | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Submission Date |
|:----:|:----------:|:------:|:------:|:-------:|:---------------:|
| 1 | - | - | - | - | - |
| 2 | - | - | - | - | - |
| 3 | - | - | - | - | - |
| ... | ... | ... | ... | ... | ... |

**Note**: Rankings are sorted by PSNR by default. All three metrics are displayed independently for comprehensive comparison.

---

## 📝 Submission Requirements

Please refer to the following documents:

1. **[Report Template](./REPORT_TEMPLATE.md)** - How to write your assignment report
2. **[Leaderboard Submission Guide](./LEADERBOARD_SUBMISSION_GUIDE.md)** - How to submit for leaderboard evaluation

---

## 🔗 Important Links

| Resource | Link | Status |
|----------|------|--------|
| **Leaderboard Website** | *To be announced* | 🔜 Coming Soon |
| **Baseline Results** | *To be announced* | 🔜 Coming Soon |
| **Dataset** | HKisland COLMAP | ✅ Available |

---

## ❓ FAQ

**Q: Are there weighted scores for the three metrics?**

A: No. All three metrics (PSNR, SSIM, LPIPS) are displayed independently. There is no combined score.

**Q: How is the ranking determined?**

A: Default ranking is by PSNR. The leaderboard website will allow sorting by any metric.

**Q: When will the leaderboard website be available?**

A: The submission portal and baseline results will be announced separately.

---

<div align="center">

**AAE5303 - Advanced Topics in Aerospace Engineering**

*Hong Kong Polytechnic University*

</div>
