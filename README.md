# camelyon17

Code adapted from [Camelyon17] (https://github.com/Camelyon17/camelyon17)

## Requirements and Setup

* torch
* torchvision
* openslide
* opencv
* matplotlib

## Usage

* Download the Camelyon17 dataset:
 + First download the label data `lesion_annotations.zip`, unzip it, and find `lesion_annotations/patient###_node_#.xml` (each # represents a digit).
 + From `centre_0` ... `centre_4`, only download the `patient###.zip` files that have labeled files in folder `lesion_annotations`, since otherwise you will to download a larger number of hugh files that cannot be used in training.

* Unzip downloaded data as `.tif` files: 
 + `python3 unzip_sh.py > unzip_all.sh`
 + `chmod +x unzip_all.sh`
 + `mkdir tif`
 + `./unzip_all.sh`

* Cut each huge `tif` file into small `png` patches, and create downsized `png` masks according the labels (tumor/normal) of the patches:
 + `python3 make_patch.py`

* (Optional) If you want to see the thumbnails of the huge tif files:
 + `python3 get_thumbnail.py`

* Make random train/dev/test splits:
 + `python3 make_manifest.py`

* Train:
 + `python3 train.py --train_epoch=10`
