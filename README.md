# Toolkit
Toolkit for Deep Learning.
```
.
    |- DataProcess/
        |- data_aug.py
        |- data_config.py
        |- data_loader.py
        |- data_select.py
        |- divide_data.py
        |- img2video.py
        |- img_resize.py
        |- video_extract.py
    | - model/
        |- vgg.py
        |- resnet.py
        |- fcn8s.py
        |- unet.py
        |- pspnet.py
        |- deeplabv3+ (deeplabv3+.py, resnet_deeplab.py, aspp.py, decoder.py)
        |- DualGCN.py
        |- JL_DCF.py
        |- swin transformer (swin_transformer_seg.py, mlp_decoder.py)

    |- utils/
        |- nms
            |- nms_cpu.py
            |- ...
        |- bbox_iou.py
        |- bbox_iou_python.py
        |- BN_torch.py
        |- dataset.py
        |- loss.py
        |- seg_transform.py
        |- show_img.py
    |- README.md
    |- test.py
    |- train.py
```


## DataProcess

DataProcess/`data_aug.py` : augment data with transforms to aug file.

DataProcess/`data_config.py` : config (hyper-)parameters.

DataProcess/`data_loader.py` : load data to tensor of DataSet type.

DataProcess/`data_select.py` : select data in various ways.

DataProcess/`divide_data.py` : divide data to train and valid files.    

DataProcess/`img2video.py` : transform images set to video using opencv. 

DataProcess/`img_resize.py` : resize images to specific size using opencv. 

DataProcess/`video_extract.py` : extract each frame of video to images file. 


## model

model/`vgg.py` : ICLR(2015) [paper](https://arxiv.org/abs/1409.1556).

utils/`resnet.py` : CVPR(2016) [paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf).

model/`fcn8s.py` : CVPR(2015) [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf).

model/`unet.py` : MICCAI(2015) [paper](https://arxiv.org/abs/1505.04597).

model/`pspnet.py` : CVPR(2017) [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf).

model/`deeplabv3+` : ECCV(2018) [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf).

model/`DualGCN.py` : BMVC(2019) [paper](https://arxiv.org/abs/1909.06121).

model/`JL_DCF.py` : CVPR(2020) [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fu_JL-DCF_Joint_Learning_and_Densely-Cooperative_Fusion_Framework_for_RGB-D_Salient_CVPR_2020_paper.pdf).

model/`swin transformer` : arXiv(2021) [paper](https://arxiv.org/abs/2103.14030).


## utils

utils/nms/`nms_cpu.py` : remove useless bounding-box by nms(Non-maximum suppression).

utils/`bbox_iou.py` : calculate iou(Intersection-of-Union) between two bounding-box.

utils/`bbox_iou_python.py` : calculate iou in python version.

utils/`BN_torch.py` : implementation of BN(BatchNormalization) in PyTorch.

utils/`dataset.py` : implementation of dataset in segmentation.

utils/`loss.py` : implementation of loss function, including FocalLoss, BinaryDiceLoss.

utils/`seg_transform.py` : implementation of transform module in segmentation.

utils/`show_img.py` : visualize image in both plt(matplotlib.pyplot) and cv2(opencv).


(Updating...)