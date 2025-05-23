## KS-17 ACTREC:TMH



![alt_images](https://github.com/TMC-ComputationalPathology/Image_registration/blob/main/Logo.png)
 
## IHC-HE-ROIAlign - Image Registration(git page in progress ,not final yet,api part is to be improved and some text too.)
 _IHC-HE-ROIAlign is an image regisration approach to detect pixel-wise shift between region of interests in large sized hematoxylin and eosin stained (H&E) (WSIs) and their corresponding immunohistochemistry stained (IHC) (WSIs)at highest magnification . This approach enables precise identification of epithelial regions and thereafter capitalise aligned HE patches of desired size in AI model trainings._

![alt_images](https://github.com/TMC-ComputationalPathology/Image_registration/blob/8a9fef52c10d9c888a91d190880e62bfcf5c4cef/screenshots/Screenshot%20from%202024-09-12%2014-00-24.png)
 <sup>**_Shift of (1877.78) pixels in X direction , (925.3435) pixels in Y direction and rotation of 0.5 degress(clockwise) is registered.The  region used is of size (8192 x 8192)pixels_**</sup>
 ![alt_images](https://github.com/TMC-ComputationalPathology/Image_registration/blob/main/screenshots/Screenshot%20from%202024-09-12%2014-12-50.png)
 ![alt_images](https://github.com/TMC-ComputationalPathology/Image_registration/blob/main/screenshots/Screenshot%20from%202024-09-12%2014-02-16.png)







#### Registration: 
Registration is the process of establishing spatial relationships between images, enabling the alignment and transfer of essential information across subjects and atlases. It is a fundamental technique in various medical imaging applications. By identifying correspondences between images, we can transfer information, which serves as prior knowledge for tasks like segmentation. For instance, knowing the boundary of a specific anatomical structure in image A allows it to function as an atlas for locating similar boundaries in other images. If the correspondences between images A and B are identified, the boundary in image A can be transferred through these correspondences, providing an approximate starting point for finding the corresponding boundaries in image B.

#### Issues and Constraints with WSI images with different markers:
One of the major constraints of registering wsi images is the size(goes up-to lakhs x lakhs of pixels)and computational constraints that comes with it. However downsampling the image to smaller levels, registering it and then upsampling again leads to incorrect shift values by some margins when pixel wise accuracy  is desired . Moreover, performing registration repeatedly on region and then on patches in incremental iterations  also leads to wrong registered shifts because of the fact that most of the features or tissue information might present in the bigger region of HE but may or may not be lost in corresponding IHC patch. Examples of such  cases were washed out tissue , puffed up size of nucleus which leads to inefficient feature detections,descriptors and matching  while using well established algorithms like ORB , BF matcher.






#### IHC-HE-ROIAlign:
The solution proposed was to use the same algorithms but region wise than on whole image and at highest magnification without using scaling once the region of interests are marked and then crop the patches out of the regions.The motivation behind this is to find at least 50000 efficient features in bigger regions even if the information is lost on patch level. 

#### Software and Packages:
The Frequently used package in development of this work is Pathomation alias(pma-python) which is Comprehensive Image Viewing and Management Software for Digital Pathology and is preferably prerequisite to utilise this work efficiently .

### Fully Automated Run
>
     python rpauto.py --ihc_file_path "IHC_FILE.svs" --he_file_path "HE_FILE.svs" --X 56788 --Y 98675 --patch_size 256 --region_size 8192

The above command will take ihc file path ,he file path as per pma_pathomation along with top left coordinates of region of interest chosen on ihc image,desired patch size and targeted region size.The code will create a directory with structure as  file_name_region_coordinates which includes tiff files for ROI from both the images and two seperate subdirectories for registered IHC and HE patches.Since our main focus is to pick corrosponding coordinates of HE images , all the patches of the HE are saved with their coordinates_name.png.The final results will be generared as follow:

>
>
    
      ├──ihc_file_name_region_x_y/
                          ├── patches_he/
    		              ├── patch_x1_y1.png
    		              ├── patch_x2_y2.png
    		              └── ...
                          ├── patches_ihc/
    		              ├── patch_0_0.png
    		              ├── patch_0_256.png
    		              └── ...
      
      ├── he_ROI.tiff

      ├── ihc_ROI.tiff

      ├── registered_X.tiff

      ├── shifted_target.tiff

An example folder WSI_region_37200_33000 is added to show the structure of the folder.Since the intial coordinates used were 37200 and 33000 so the first he patch will be patch_37200_33000.png where as corrosponding ihc patch will be patch_0_0.png ,patch_37200_33256-patch_0_256 and so on so forth with increment of patch size of 256.
 

#### Run as API in Docker:
To run this script as an  API (with FASTAPI UI) in Docker Container ,the following steps needed to be taken:
1. Save the docker file,irapi.py,requirement.txt scripts in folder of your choice.
2. Build an image using the following command
>
     sudo docker build -t ihc_he_align_image .

3.To run in container use the following command.One can use image as well as container name of their own choice.
>
    sudo docker run --name ihc_he_align_container ihc_he_align_image 
        

        
    		
	


