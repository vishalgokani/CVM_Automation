# CVM Automation

This repository contains tools for automating cervical vertebral maturation
(CVM) measurements. The current implemented pipeline post-processes existing
C2-C4 segmentation masks to measure inferior endplate doming and predict CVM
class.

## Repository Structure

```text
post-processing/   CVM measurements and overlay generation from existing masks
training/          Placeholder for future segmentation model training
inference/         Placeholder for future segmentation-mask inference
original/          Legacy/reference scripts
```

## Environment

Install a Python environment, then install the repository requirements:

```bat
pip install -r requirements.txt
```

## Post-Processing

Run the mask post-processing pipeline from Command Prompt:

```bat
python post-processing\postprocess_cvm.py --target-folder "<TARGET_FOLDER>"
```

`<TARGET_FOLDER>` should contain original `.bmp` images and a `masks` folder
with C2, C3, and C4 mask files. See `post-processing/README.md` for details.

