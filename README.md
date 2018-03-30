# camelyon17

Code adapted from [Camelyon17] (https://github.com/Camelyon17/camelyon17)

## Requirements and Setup

* torch
* torchvision
* openslide
* opencv
* matplotlib

## Usage

### Download the Camelyon17 dataset:
 + First download the label data `lesion_annotations.zip`, unzip it, and you will find the label data `lesion_annotations/patient###_node_#.xml` (where each # represents a digit) for the image data files `centre_#/patient###_node_#.tif` you will download.
 + From folders `centre_0` ... `centre_4`, only download the `patient###.zip` files that have label files in the above folder `lesion_annotations`, since otherwise you will to download a much larger number of hugh image data files that cannot be used in training due to lacking of labels.

### Unzip downloaded data as `.tif` files: 
 + `python3 unzip_sh.py > unzip_all.sh`
 + `chmod +x unzip_all.sh`
 + `mkdir tif`
 + `./unzip_all.sh`

### Cut each huge `.tif` image file and put the results into a folder (named after the image file) containing small `.png` patches, and create downsized `.png` masks according the labels (tumor/normal) of the patches:
 + `python3 make_patch.py`

### (Optional) If you want to see the thumbnails of the huge tif files:
 + `python3 get_thumbnail.py`

### Make random train/dev/test splits:
 + `python3 make_manifest.py`

### Train:
 + `python3 train.py --train_epoch=10'
