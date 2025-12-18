# AAE5303 Assignment 2 - Report Template

## 📋 Report Structure

Your assignment report should follow this structure. The report should be submitted as a **PDF file** with a maximum of **10 pages** (excluding references and appendix).

---

## Required Sections

### 1. Title Page

```
AAE5303 Assignment 2: 3D Gaussian Splatting

Student Name: [Your Name]
Student ID: [Your Student ID]
Date: [Submission Date]
```

---

### 2. Abstract (0.5 page)

Provide a brief summary of:
- The objective of this assignment
- Your approach and methodology
- Key results achieved (PSNR, SSIM, LPIPS values)
- Main conclusions

---

### 3. Introduction (1 page)

#### 3.1 Background
- Brief introduction to 3D Gaussian Splatting
- Importance in novel view synthesis
- Comparison with other methods (e.g., NeRF)

#### 3.2 Objectives
- State the goals of this assignment
- What you aim to achieve

---

### 4. Methodology (2-3 pages)

#### 4.1 3D Gaussian Splatting Overview
- Explain the core algorithm
- Gaussian representation (position, covariance, opacity, spherical harmonics)
- Rendering equation
- Training process

#### 4.2 OpenSplat Framework
- Describe the framework used
- Key components and pipeline

#### 4.3 Training Configuration
- List all hyperparameters used
- Justify your choices (if different from default)

**Example Table:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Iterations | 30000 | Total training iterations |
| SH Degree | 3 | Spherical harmonics degree |
| SSIM Weight | 0.2 | Weight for SSIM loss |
| Learning Rate (means) | 0.00016 | Position learning rate |
| ... | ... | ... |

---

### 5. Experimental Setup (1 page)

#### 5.1 Dataset
- Describe the HKisland dataset
- Number of images, resolution
- Train/test split (if applicable)

#### 5.2 Hardware and Software
- Computing environment
- GPU/CPU specifications
- Software versions

#### 5.3 Evaluation Metrics
- Define PSNR, SSIM, and LPIPS
- Explain how they are computed

---

### 6. Results (2-3 pages)

#### 6.1 Training Analysis
- Training loss curve
- Convergence behavior
- Training time

**Include Figure:** Training loss over iterations

#### 6.2 Quantitative Results

**Required Table:**

| Metric | Value |
|--------|-------|
| **PSNR** | XX.XX dB |
| **SSIM** | 0.XXXX |
| **LPIPS** | 0.XXXX |

#### 6.3 Qualitative Results
- Include rendered image comparisons
- Show different viewpoints
- Highlight areas of good/poor reconstruction

**Include Figures:** 
- Rendered vs Ground Truth comparisons (at least 3 views)
- Close-up details if applicable

#### 6.4 Model Statistics
- Number of Gaussians
- Output file size
- Any other relevant statistics

---

### 7. Discussion (1 page)

#### 7.1 Analysis
- Interpret your results
- What worked well?
- What could be improved?

#### 7.2 Challenges
- Difficulties encountered
- How you addressed them

#### 7.3 Comparison with Baseline
- Compare with baseline results (when available)
- Explain any differences

---

### 8. Conclusion (0.5 page)

- Summarize key findings
- Lessons learned
- Suggestions for future improvement

---

### 9. References

Use proper academic citation format. Include at least:

1. Kerbl, B., et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." SIGGRAPH 2023.
2. OpenSplat GitHub repository
3. Any other relevant papers or resources

---

### 10. Appendix (Optional, not counted in page limit)

- Additional figures
- Complete training logs
- Code modifications (if any)

---

## 📝 Formatting Requirements

| Requirement | Specification |
|-------------|---------------|
| **Format** | PDF |
| **Page Limit** | 10 pages (excluding references/appendix) |
| **Font** | 11pt, Times New Roman or similar |
| **Margins** | 2.5 cm / 1 inch |
| **Figures** | High resolution, properly labeled |
| **Tables** | Clear headers, proper alignment |
| **Language** | English |

---

## ✅ Checklist Before Submission

- [ ] All required sections are included
- [ ] Page limit is respected
- [ ] Figures are clear and properly labeled
- [ ] Tables have clear headers
- [ ] PSNR, SSIM, LPIPS values are reported
- [ ] Training configuration is documented
- [ ] References are properly formatted
- [ ] PDF file is named correctly: `{StudentID}_Report.pdf`

---

## 📊 Required Figures

1. **Training Loss Curve** - Loss vs iterations
2. **Rendered Images** - At least 3 different viewpoints
3. **Comparison Images** - Rendered vs Ground Truth (side by side)
4. **Optional**: Any additional visualizations that support your analysis

---

## 📈 Required Tables

1. **Training Configuration** - All hyperparameters used
2. **Final Results** - PSNR, SSIM, LPIPS values
3. **Model Statistics** - Gaussian count, file size, training time

---

<div align="center">

**Good luck with your assignment!**

</div>

