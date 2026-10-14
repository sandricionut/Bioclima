# Bahna demo

The folder contains three independent, OS-agnostic examples:

```text
download_demo_data.py       download/extract the Bahna demo dataset
fusion_demo.py              point-cloud + raster multimodal fusion
forest_ebv_ecv_demo.py      forest EBV/ECV products and validation example
```

Run from any working directory with the Python environment where `uav_biodiv`
is installed:

```bash
python download_demo_data.py
python fusion_demo.py
python forest_ebv_ecv_demo.py
```

`forest_ebv_ecv_demo.py` produces CHM, NDVI, GNDVI, NDRE,
`forest_health.tif`, `lst_anomaly.tif`, ecosystem-extent metrics,
canopy-structure metrics, LST-anomaly ECV statistics, Random Forest habitat
classification, confusion matrix / overall accuracy, a text summary report
and a JSON metrics file.

The downloaded Bahna demo does not include an independently mapped habitat
reference. For software demonstration only, the forest example creates a
three-class reference from explicit CHM/NDVI thresholds. This generated map is
**not** an independent scientific validation dataset. Replace it with a field-
or expert-mapped habitat raster for scientific accuracy assessment.

`fusion_demo.py` uses a single-line `tqdm` elapsed-time indicator for long operations.
