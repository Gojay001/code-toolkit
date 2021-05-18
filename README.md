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