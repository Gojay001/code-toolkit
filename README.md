# Toolkit
Toolkit for coding, including deep learning, image processing, file processing.
```
.
    |- source-code/
        |- bn_details.py
        |- bn_run.py
        |- CrossEntropyLoss.py
        |- regularization.py
        |- SoftMax.py
    | - torch_model/
        |- vgg.py
        |- resnet.py
        |- densenet.py
        |- fcn8s.py
        |- unet.py
        |- pspnet.py
        |- deeplabv3+ (deeplabv3+.py, resnet_deeplab.py, aspp.py, decoder.py)
        |- DualGCN.py
        |- JL_DCF.py
        |- swin transformer (swin_transformer_seg.py, mlp_decoder.py)
    |- torch_template
        |- test_net.py
        |- test.py
        |- train.py
    |- utils/
        |- data
            |- data_config.py
            |- divide_data.py
            |- img_ops.py
            |- imgs2video.py
        |- nms
            |- nms_cpu.py
            |- ...
        |- bbox_iou.py
        |- bbox_iou_python.py
        |- BN_torch.py
        |- count_norm.py
        |- dataset.py
        |- loss.py
        |- seg_transform.py
        |- show_img.py
    |- README.md

```


## source_code

source-code/`bn_details.py` : implementation of BN(BatchNormalization) and analysis of its details.

source-code/`bn_run.py` : implementation of BN(BatchNormalization) and simulation of running.

source-code/`CrossEntropyLoss.py` : implementation of custom CrossEntropyLoss and BCELoss.

source-code/`regularization.py` : implementation of L1/L2 normalization, L1/L2 regularization and Dropout.

source-code/`SoftMax.py` : implementation of SoftMax function in various version.


## torch_model

model/`vgg.py` : ICLR(2015) [paper](https://arxiv.org/abs/1409.1556).

model/`resnet.py` : CVPR(2016) [paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf).

model/`densenet.py` : CVPR(2017) [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf).

model/`fcn8s.py` : CVPR(2015) [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf).

model/`unet.py` : MICCAI(2015) [paper](https://arxiv.org/abs/1505.04597).

model/`pspnet.py` : CVPR(2017) [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf).

model/`deeplabv3+` : ECCV(2018) [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf).

model/`DualGCN.py` : BMVC(2019) [paper](https://arxiv.org/abs/1909.06121).

model/`JL_DCF.py` : CVPR(2020) [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fu_JL-DCF_Joint_Learning_and_Densely-Cooperative_Fusion_Framework_for_RGB-D_Salient_CVPR_2020_paper.pdf).

model/`swin transformer` : arXiv(2021) [paper](https://arxiv.org/abs/2103.14030).


## utils

utils/data/`count_nrom.py` : count the mean and the standard deviation from datasets.

utils/data/`data_config.py` : config (hyper-)parameters in main.py.

utils/data/`dataset.py` : implementation of dataset in segmentation.

utils/data/`divide_data.py` : divide origin data into train and valid set.

utils/data/`img_ops.py` : resize, hist equalize and blur images using opencv.

utils/data/`imgs2video.py` : convert images to video and extract images from video.

utils/data/`seg_transform.py` : implementation of transform module in segmentation.

utils/nms/`nms_cpu.py` : remove useless bounding-box by nms(Non-maximum suppression).

utils/`bbox_iou.py` : calculate iou(Intersection-of-Union) between two bounding-box.

utils/`bbox_iou_python.py` : calculate iou in python version.

utils/`logging_util.py` : implementation of logging module with formatting.

utils/`loss.py` : implementation of loss function, including FocalLoss, BinaryDiceLoss.

utils/`show_img.py` : visualize image in both plt(matplotlib.pyplot) and cv2(opencv).


(Updating...)