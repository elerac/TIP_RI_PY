# MLRI+wei demosaicing in Python

This repository provides an unofficial Python implementation of the **MLRI+wei demosaicing** method proposed by Kiku et al., "[Beyond Color Difference: Residual Interpolation for Color Image Demosaicking](http://www.ok.sc.e.titech.ac.jp/res/DM/TIP_RI.pdf)," in the IEEE TIP 2016. The authors provide only a MATLAB implementation, so I re-implement it in Python using NumPy and OpenCV.

## Usage

```python
from tip_ri import demosaicing

img_bgr = demosaicing(img_raw, cv2.COLOR_BAYER_RGGB2BGR)
```

You can copy the `tip_ri.py` file into your project and import the demosaicing function. The interface is designed to be similar to OpenCV’s cv2.demosaicing.

## Benchmark

The evaluation was conducted on a 1500 × 1000 image with the RGGB Bayer pattern using the methods listed below. Experiments were performed on a MacBook Pro equipped with an M1 Max chip (10-core CPU, 32 GB RAM). PSNR was computed with respect to the original reference image, and runtime was averaged over 50 runs.

The benchmark compares the proposed MLRI+wei implementation against several widely used demosaicing baselines: Malvar (2004) and Menon (2007) from the colour-demosaicing library, as well as OpenCV’s built-in edge-aware and bilinear demosaicing methods.

### PSNR / Runtime Summary

| Method      | Implementation     | PSNR (dB) | Time (s) [mean ± std] |
| ----------- | ------------------ | --------- | --------------------- |
| MLRI+wei    | This repository    | **38.17** | 0.7629±0.0270         |
| Menon2007   | colour_demosaicing |   35.69   | 0.3383±0.0077         |
| Malvar2004  | colour_demosaicing |   34.54   | 0.1058±0.0061         |
| Edge-Aware  | OpenCV             |   30.77   | 0.0004±0.0002         |
| Bilinear    | OpenCV             |   30.57   | 0.0005±0.0006         |

### Input and CFA

| Input (BGR)         | Bayer CFA (RGB-colored) |
| ------------------- | ----------------------- |
| ![input][img_input] | ![cfa_rgb][img_cfa_rgb] |

### CFA Cropped Images (4x nearest-neighbor zoom)

| Crop Region          | Original Input              | RGB-colored CFA              |
| -------------------- | --------------------------- | ---------------------------- |
| crop1                | ![input1][img_input_crop1]  | ![cfa_crop1][img_cfa_crop1]  |
| crop2                | ![input2][img_input_crop2]  | ![cfa_crop2][img_cfa_crop2]  |

### Demosaiced Cropped Images (4x nearest-neighbor zoom)

<!-- markdownlint-disable MD060 -->

| Method     | Crop1                  | Crop2                  |
| :--------: | :--------------------: | :--------------------: |
| Original   | ![o1][orig_crop1]      | ![o2][orig_crop2]      |
| Bilinear   | ![obl][crop_opencv]    | ![obl2][crop_opencv2]  |
| Edge-aware | ![ea][crop_opencv_ea]  | ![ea2][crop_opencv_ea2] |
| Malvar2004 | ![cml][crop_cmalvar]   | ![cml2][crop_cmalvar2] |
| Menon2007  | ![cmn][crop_cmenon]    | ![cmn2][crop_cmenon2]  |
| MLRI+wei   | ![tip][crop_tip]       | ![tip2][crop_tip2]     |

<!-- markdownlint-restore -->

[img_input]: results/tshirts_input.png
[img_cfa_rgb]: results/tshirts_cfa_rgb.png
[img_input_crop1]: results/tshirts_input_crop1.png
[img_input_crop2]: results/tshirts_input_crop2.png
[img_cfa_crop1]: results/tshirts_cfa_rgb_crop1.png
[img_cfa_crop2]: results/tshirts_cfa_rgb_crop2.png
[orig_crop1]: results/tshirts_input_crop1.png
[orig_crop2]: results/tshirts_input_crop2.png
[crop_tip]: results/tshirts_demosaiced_tip_ri_mlri_crop1.png
[crop_tip2]: results/tshirts_demosaiced_tip_ri_mlri_crop2.png
[crop_opencv_ea]: results/tshirts_demosaiced_opencv_ea_crop1.png
[crop_opencv_ea2]: results/tshirts_demosaiced_opencv_ea_crop2.png
[crop_opencv]: results/tshirts_demosaiced_opencv_bilinear_crop1.png
[crop_opencv2]: results/tshirts_demosaiced_opencv_bilinear_crop2.png
[crop_cmalvar]: results/tshirts_demosaiced_colour_malvar2004_crop1.png
[crop_cmalvar2]: results/tshirts_demosaiced_colour_malvar2004_crop2.png
[crop_cmenon]: results/tshirts_demosaiced_colour_menon2007_crop1.png
[crop_cmenon2]: results/tshirts_demosaiced_colour_menon2007_crop2.png
