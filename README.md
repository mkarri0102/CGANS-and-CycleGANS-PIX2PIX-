# CGANS-and-CycleGANS-PIX2PIX-Image Translation

## How to run
In this repo, we've implemented both Conditional GANS and Cycle GANS on multiple datasets.

The datasets we've used include:

 Markup :
   - Edges -> Shoes translation *(CGANS)
   - Satellite view -> Street View translation of maps *(CGANS)
   - Segmentation of road views *(CGANS)
   - Low Dynamic Resolution -> High Dynamic Resolution of photographs(CGANS)
   - Summer  -> Winter translation(Cycle GANs)


The datasets with * are implemented by the original paper. To implement your own dataset, please refer to one of the .json files as reference and create a similar config file for your dataset. For each dataset, dataloader.py has been written accordingly. Please create a folder named datasets in the current working directory and put your datasets in this folder. The path for train folder, test folder should be mentioned in the config files.

After modifying the dataloader.py and .json file as required by the dataset, you can run the CGANS.py file or CycleGANs.py file as required by the experiment. You can also refer to the notebooks, make the required changes in the notebook and run accordingly.

## Introduction

Image to Image translation is defined as the task of learning a mapping between the input image and output image. Since most problems in image processing. computer graphics and computer vision can be devised into a Image translation problem, having a model that can do the Image to Image translation without having to specify a task will be very useful to the Computer Vision and the Image processing Community. This can be achieved by the Generative Adversarial Networks. 

## Results

<img width="1147" alt="Screen Shot 2022-07-21 at 6 32 03 PM" src="https://user-images.githubusercontent.com/91915342/180343321-022fdf07-e817-4448-965f-67212fac7f1b.png">

<img width="1189" alt="Screen Shot 2022-07-21 at 6 27 42 PM" src="https://user-images.githubusercontent.com/91915342/180343371-85750e20-4044-4606-9900-d70a1b6c12eb.png">
