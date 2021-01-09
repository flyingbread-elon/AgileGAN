## AgileGAN
We public our test code for artistic style image generation to reproduce our paper results. 
This codebase provides:
-test code
-test input samples in our paper.

## pretrain model
Please download the data and pretrain folder from: [data](https://drive.google.com/drive/folders/1sksjD4awYwSAgibix83hVtx1sm4KOekm?usp=sharing) and [pretrain](https://drive.google.com/drive/folders/118Fjhp-o5hrttK8M_sKSa4u6MrDJt70m?usp=sharing).

To follow anonymity requirement, download links direct to google drive under anonymous account: AnonymousSiggraph225@gmail.com .
## Requirements
- CUDA 10.1
- Python 3
- PyTorch tested on 1.6.0
- PIL
- skimage
- numpy
- cv2 tested on 4.2.0


## Test
We provide 5 styles in pretrain folder: cartoon, oil, wuxia, jackie, scarlett.
Please use following code to generate result.

`
python test.py --path examples/29899.jpg --style cartoon
`

