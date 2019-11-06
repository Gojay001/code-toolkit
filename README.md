# Toolkit
Toolkit for deep learning.
```
.
    |- DataProcess/
        |- data_aug.py
        |- data_config.py
        |- data_loader.py
        |- divide_data.py
        |- img_resize.py
        |- video_extract.py
    |- utils/
        |- bbox_iou.py
    |- README.md
    |- test.py
    |- train.py
```

## DataProcess

DataProcess/**data_aug.py** : augment data with transforms to aug file.

DataProcess/**data_config.py** : config (hyper-)parameters.

DataProcess/**data_loader.py** : load data to tensor of DataSet type.

DataProcess/**divide_data.py** : divide data to train and valid files.    

DataProcess/**img_resize.py** : resize images to specific size using opencv. 

DataProcess/**video_extract.py** : extract each frame of video to images file. 

utils/**bbox_iou.py** : calculate iou between two bounding-box.

(Updating...)