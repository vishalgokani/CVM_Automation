# CVM Post-Processing

This folder contains the CVM post-processing pipeline for masks that have
already been created by an upstream segmentation model.

The script expects:

```text
<TARGET_FOLDER>/
├── image_001.bmp
├── image_002.bmp
└── masks/
    ├── image_001_C2.bmp
    ├── image_001_C3.bmp
    ├── image_001_C4.bmp
    └── ...
```

Mask files may also be organized in `masks/C2`, `masks/C3`, and `masks/C4`.

Run from Command Prompt:

```bat
python post-processing\postprocess_cvm.py --target-folder "<TARGET_FOLDER>"
```

Optional quick review on the first few images:

```bat
python post-processing\postprocess_cvm.py --target-folder "<TARGET_FOLDER>" --limit 10
```

Outputs are written by default to:

```text
<TARGET_FOLDER>/postprocessing_overlays/
<TARGET_FOLDER>/cvm_postprocessing_measurements.csv
```

The doming ratios are:

- C2: C2 dome height / C3 posterior vertebral body height
- C3: C3 dome height / C3 posterior vertebral body height
- C4: C4 dome height / C4 posterior vertebral body height

A vertebra is marked as doming when its ratio is strictly greater than `0.10`.
The CVM prediction is `CVM1` for `000`, `CVM2` for `100`, `CVM3` for `110`,
and `CVM4-6` for `111`. Other patterns are marked `atypical`.
