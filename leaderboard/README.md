# 🏆 AAE5303 3D Gaussian Splatting - Leaderboard

## 📁 Evaluation Dataset

**AMtown02** sequence from MARS-LVIG / UAVScenes Dataset

| Resource | Link |
|----------|------|
| MARS-LVIG Dataset | https://mars.hku.hk/dataset.html |
| UAVScenes GitHub | https://github.com/sijieaaa/UAVScenes |

---

## 🎯 Baseline Results

A **baseline implementation** is provided to help students understand the complete workflow and benchmark their results.

### Baseline Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | AMtown02 (1,380 images) |
| **Training Iterations** | 300 |
| **Downscale Factor** | 4× (training resolution: 612×512) |
| **Training Device** | CPU only |
| **Training Time** | ~25 minutes |
| **Initial Points** | 8.3M (from LiDAR merge) |
| **Output Model Size** | 2.0 GB |

### Baseline Training Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Initial Loss** | 0.2164 | Combined L1 + SSIM loss at step 1 |
| **Final Loss** | 0.0888 | Loss at step 300 |
| **Minimum Loss** | 0.0454 | Best loss achieved during training |
| **Loss Reduction** | 58.9% | Improvement over 300 iterations |

### Baseline Test Set Evaluation

> **📢 Important Note on Evaluation**
>
> The baseline model has been trained, but **test set evaluation is pending**. The official baseline evaluation metrics (PSNR, SSIM, LPIPS) will be updated once:
>
> 1. ✅ **Test set is defined** - 138 images (10% of dataset, every ~10th image)
> 2. ⏳ **Rendering tool is set up** - To render test views from the trained model
> 3. ⏳ **Metrics are calculated** - Using the evaluation script provided
>
> **Current Status:** Test set indices are defined in [`../baseline/test_set_indices.json`](../baseline/test_set_indices.json).
>
> Students can proceed with training and will use the same test set for fair comparison.

#### Expected Baseline Performance (Estimated)

Based on training loss and typical 3DGS performance with similar configurations:

| Metric | Estimated Range | Note |
|--------|-----------------|------|
| **PSNR** | 18-21 dB | Low due to limited iterations (300) and 4× downscaling |
| **SSIM** | 0.70-0.75 | Moderate structural similarity |
| **LPIPS** | 0.20-0.30 | Perceptual quality affected by low resolution training |

> ⚠️ **These are rough estimates.** Actual evaluation will be performed and updated.
>
> Students are **strongly encouraged to surpass these estimates** by:
> - Training for more iterations (3,000-30,000)
> - Using smaller downscale factors
> - Leveraging GPU acceleration
> - Tuning hyperparameters

📂 **Baseline Details**: See [../baseline/README.md](../baseline/README.md) for complete implementation details, training logs, and configuration.

📊 **Evaluation Script**: [`../baseline/evaluate_baseline.py`](../baseline/evaluate_baseline.py) - Tool for calculating test set metrics

---

### How to Beat the Baseline

Students are encouraged to improve upon the baseline by:

✅ **Training Longer**: 300 → 3,000-30,000 iterations
- Expected improvement: +3-8 dB PSNR, +0.05-0.15 SSIM
- Training time: 4-10+ hours on CPU, or 30 min - 2 hours on GPU

✅ **GPU Acceleration**: 50-100× faster than CPU
- Enables practical training for 30,000 iterations
- Better quality in reasonable time

✅ **Better Initialization**: Higher quality point cloud
- Reduce LiDAR sampling rate (10× → 5× or 2×)
- Use structure-from-motion if available

✅ **Hyperparameter Tuning**: Optimize for your hardware
- Adjust `densify-grad-thresh` and `refine-every`
- Tune SSIM weight for perceptual quality
- Experiment with learning rates

✅ **Resolution Strategy**: Reduce downscaling if memory allows
- Baseline uses 4× downscaling for memory constraints
- Try 2× or 1× if you have sufficient RAM/VRAM
- Expected improvement: +2-5 dB PSNR per 2× resolution increase

---

## 📊 Evaluation Metrics

The leaderboard evaluates submissions using three standard rendering quality metrics. All metrics are computed between **rendered images** and **ground truth images** on a held-out test set.

### Test Set Definition

- **Total Images**: 1,380 (AMtown02 sequence)
- **Test Set Size**: 138 images (10% of dataset)
- **Sampling**: Every ~10th image for uniform temporal coverage
- **Test Indices**: Defined in `baseline/test_set_indices.json`
- **Purpose**: Fair comparison across all submissions

---

### 1. PSNR (Peak Signal-to-Noise Ratio) ↑

**Higher is better** | Unit: dB (decibels)

#### Definition

PSNR measures the ratio between the maximum possible power of a signal and the power of corrupting noise. It quantifies the pixel-level reconstruction accuracy.

#### Mathematical Formula

$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right) = 20 \cdot \log_{10}\left(\frac{MAX_I}{\sqrt{MSE}}\right)$$

where:
- $MAX_I$ = Maximum possible pixel value (255 for 8-bit images)
- $MSE$ = Mean Squared Error between rendered and ground truth images

$$MSE = \frac{1}{H \times W \times C}\sum_{i=1}^{H}\sum_{j=1}^{W}\sum_{k=1}^{C}[I_{rendered}(i,j,k) - I_{gt}(i,j,k)]^2$$

where:
- $H$ = Image height
- $W$ = Image width  
- $C$ = Number of channels (3 for RGB)

#### Reference Code

```python
import numpy as np

def calculate_psnr(rendered: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate PSNR between rendered and ground truth images.
    
    Args:
        rendered: Rendered image, shape (H, W, C), dtype uint8, range [0, 255]
        ground_truth: Ground truth image, shape (H, W, C), dtype uint8, range [0, 255]
    
    Returns:
        PSNR value in dB
    """
    # Convert to float64 for precision
    rendered = rendered.astype(np.float64)
    ground_truth = ground_truth.astype(np.float64)
    
    # Calculate MSE
    mse = np.mean((rendered - ground_truth) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR (MAX_I = 255 for 8-bit images)
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    
    return psnr
```

#### Using scikit-image (Recommended)

```python
from skimage.metrics import peak_signal_noise_ratio

# rendered and ground_truth: numpy arrays, shape (H, W, C), dtype uint8
psnr = peak_signal_noise_ratio(ground_truth, rendered, data_range=255)
```

---

### 2. SSIM (Structural Similarity Index) ↑

**Higher is better** | Range: 0 to 1

#### Definition

SSIM measures the perceptual quality of images by comparing structural information, luminance, and contrast. Unlike PSNR, SSIM is designed to better correlate with human visual perception.

#### Mathematical Formula

$$SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

where:
- $\mu_x$, $\mu_y$ = Local means of images $x$ and $y$
- $\sigma_x^2$, $\sigma_y^2$ = Local variances of images $x$ and $y$
- $\sigma_{xy}$ = Local covariance of images $x$ and $y$
- $C_1 = (K_1 \cdot L)^2$, $C_2 = (K_2 \cdot L)^2$ = Stability constants
- $L$ = Dynamic range of pixel values (255 for 8-bit images)
- $K_1 = 0.01$, $K_2 = 0.03$ (default constants)

The SSIM is computed locally using a sliding window (typically 11×11 Gaussian window), then averaged across the entire image.

#### Reference Code

```python
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

def calculate_ssim(rendered: np.ndarray, ground_truth: np.ndarray, 
                   window_size: int = 11, K1: float = 0.01, K2: float = 0.03) -> float:
    """
    Calculate SSIM between rendered and ground truth images.
    
    Args:
        rendered: Rendered image, shape (H, W, C), dtype uint8, range [0, 255]
        ground_truth: Ground truth image, shape (H, W, C), dtype uint8, range [0, 255]
        window_size: Size of the sliding window (default: 11)
        K1, K2: Stability constants
    
    Returns:
        SSIM value (0 to 1)
    """
    # Convert to float64
    rendered = rendered.astype(np.float64)
    ground_truth = ground_truth.astype(np.float64)
    
    L = 255.0  # Dynamic range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    # Calculate SSIM for each channel and average
    ssim_channels = []
    
    for c in range(rendered.shape[2]):
        img1 = rendered[:, :, c]
        img2 = ground_truth[:, :, c]
        
        # Local means
        mu1 = gaussian_filter(img1, sigma=1.5)
        mu2 = gaussian_filter(img2, sigma=1.5)
        
        # Local variances and covariance
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = gaussian_filter(img1 ** 2, sigma=1.5) - mu1_sq
        sigma2_sq = gaussian_filter(img2 ** 2, sigma=1.5) - mu2_sq
        sigma12 = gaussian_filter(img1 * img2, sigma=1.5) - mu1_mu2
        
        # SSIM formula
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / denominator
        ssim_channels.append(np.mean(ssim_map))
    
    return np.mean(ssim_channels)
```

#### Using scikit-image (Recommended)

```python
from skimage.metrics import structural_similarity

# rendered and ground_truth: numpy arrays, shape (H, W, C), dtype uint8
ssim = structural_similarity(ground_truth, rendered, channel_axis=2, data_range=255)
```

---

### 3. LPIPS (Learned Perceptual Image Patch Similarity) ↓

**Lower is better** | Range: 0 to 1

#### Definition

LPIPS uses deep neural network features (typically from VGG or AlexNet) to measure perceptual similarity. It computes the distance between learned feature representations, which correlates better with human perception than pixel-level metrics.

#### Mathematical Formula

$$LPIPS(x, y) = \sum_{l} \frac{1}{H_l W_l} \sum_{h,w} \left\| w_l \odot \left( \hat{\phi}_l^{x}(h,w) - \hat{\phi}_l^{y}(h,w) \right) \right\|_2^2$$

where:
- $\hat{\phi}_l^{x}$, $\hat{\phi}_l^{y}$ = Normalized feature maps from layer $l$ for images $x$ and $y$
- $w_l$ = Learned weights for layer $l$
- $H_l$, $W_l$ = Height and width of feature maps at layer $l$
- $\odot$ = Element-wise multiplication

#### Reference Code

```python
import numpy as np
import torch
import lpips

def calculate_lpips(rendered: np.ndarray, ground_truth: np.ndarray, 
                    net: str = 'vgg') -> float:
    """
    Calculate LPIPS between rendered and ground truth images.
    
    Args:
        rendered: Rendered image, shape (H, W, C), dtype uint8, range [0, 255]
        ground_truth: Ground truth image, shape (H, W, C), dtype uint8, range [0, 255]
        net: Network backbone ('vgg' or 'alex')
    
    Returns:
        LPIPS value (0 to 1, lower is better)
    """
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net=net)
    
    # Convert to tensor: (H, W, C) uint8 [0,255] -> (1, C, H, W) float [-1, 1]
    def to_tensor(img):
        img = img.astype(np.float32) / 255.0  # [0, 1]
        img = img * 2 - 1  # [-1, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return img
    
    rendered_tensor = to_tensor(rendered)
    gt_tensor = to_tensor(ground_truth)
    
    # Calculate LPIPS
    with torch.no_grad():
        lpips_value = lpips_model(rendered_tensor, gt_tensor)
    
    return lpips_value.item()
```

#### Using lpips Package (Recommended)

```bash
pip install lpips
```

```python
import lpips
import torch
import numpy as np

# Initialize model (do this once)
lpips_model = lpips.LPIPS(net='vgg')

def calculate_lpips(rendered: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Args:
        rendered: numpy array, shape (H, W, 3), dtype uint8, range [0, 255]
        ground_truth: numpy array, shape (H, W, 3), dtype uint8, range [0, 255]
    Returns:
        LPIPS value (lower is better)
    """
    # Normalize to [-1, 1] and convert to tensor
    rendered_tensor = torch.from_numpy(rendered).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
    gt_tensor = torch.from_numpy(ground_truth).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
    
    with torch.no_grad():
        lpips_value = lpips_model(rendered_tensor, gt_tensor)
    
    return lpips_value.item()
```

---

## 📦 Complete Evaluation Script

Use the provided script to compute all three metrics for your submission:

**Location**: [`../baseline/evaluate_baseline.py`](../baseline/evaluate_baseline.py)

```bash
# Step 1: Generate test set definition (already done for baseline)
python3 evaluate_baseline.py --test-set-only

# Step 2: Render test images from your trained model
# (Use your Gaussian Splatting viewer/renderer)

# Step 3: Calculate metrics
python3 evaluate_baseline.py \
    --rendered ./your_rendered_test_images/ \
    --gt ./ground_truth_test_images/ \
    --output your_submission.json
```

---

## 📄 Submission Format

Submit a JSON file with the following format:

```json
{
    "group_id": "Group_01",
    "group_name": "Team Alpha",
    "metrics": {
        "psnr": 25.67,
        "ssim": 0.8834,
        "lpips": 0.1052
    },
    "submission_date": "2024-12-25"
}
```

Template file: [submission_template.json](./submission_template.json)

---

## 🌐 Leaderboard Website

> **📢 The leaderboard submission website will be announced later.**
>
> Students will submit their JSON files through the web portal.
>
> The leaderboard will display rankings based on the three metrics, with special recognition for:
> - 🥇 **Best PSNR**
> - 🥈 **Best SSIM**
> - 🥉 **Best LPIPS**
> - 🏆 **Best Overall** (combined ranking)

---

## 📚 Additional Resources

- **Baseline Implementation**: [../baseline/README.md](../baseline/README.md)
- **Submission Guide**: [LEADERBOARD_SUBMISSION_GUIDE.md](./LEADERBOARD_SUBMISSION_GUIDE.md)
- **Evaluation Script**: [../baseline/evaluate_baseline.py](../baseline/evaluate_baseline.py)
- **Test Set Definition**: [../baseline/test_set_indices.json](../baseline/test_set_indices.json)
- **OpenSplat Documentation**: https://github.com/pierotofy/OpenSplat
- **UAVScenes Dataset**: https://github.com/sijieaaa/UAVScenes
- **3D Gaussian Splatting Paper**: Kerbl et al., SIGGRAPH 2023

---

<div align="center">

**AAE5303 - Robust Control Technology in Low-Altitude Aerial Vehicle**

*Department of Aeronautical and Aviation Engineering*

*The Hong Kong Polytechnic University*

December 2024

</div>
