# cheating-detection-deployment

Dataset Detail
==============

The dataset includes 2271 images (split in 3 folder, train, valid and test).
Human are annotated in YOLO Object Detection format.

The following pre-processing was applied to each image:

* Auto-orientation of pixel data (with EXIF-orientation stripping)

The following augmentation was applied to create 3 versions of each source image:

* 50% probability of horizontal flip
* Random rotation of between -5 and +5 degrees
* Random brigthness adjustment of between -25 and +25 percent
* Random exposure adjustment of between -10 and +10 percent

## Paper Reference

Please cite the following paper:
1. Cheating Detection Through CCTV using YOLOv7 - 2nd MIMSE (2023)
   (Production Process on Proceeding)
2. Pendeteksian Kecurangan Ujian Melalui CCTV Menggunakan Algoritma YOLOv5 - STAINS Proceeding (2024)
   ([https://doi.org/10.29407/stains.v3i1.4360](https://doi.org/10.29407/stains.v3i1.4360))
