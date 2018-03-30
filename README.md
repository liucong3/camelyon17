# camelyon17

Code adapted from [Camelyon17] (https://github.com/Camelyon17/camelyon17)

## Requirements and Setup

* torch
* torchvision
* openslide
* opencv
* matplotlib

## Usage

* Download the Camelyon17 dataset
First download the label data `lesion_annotations.zip`, unzip it, and find `lesion_annotations/patient###_node_#.xml` (each # represents a digit).
From `centre_0` ... `centre_4`, only download the `patient###.zip` files that have labeled files in folder `lesion_annotations`, since otherwise you need to download many more high files that cannot be used in training.


