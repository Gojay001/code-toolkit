class DefaultConfigs(object):
    #=======================================
    #        1. String parameters
    #=======================================
    train_data = "../data/train/"
    valid_data = "../data/valid/"
    test_data = "../data/test/fuza100/"
    # model_path = "./models/resnet50.pth"
    model_path = "../checkpoints/resnet101_model-98.833.pth"
    checkpoint = "../checkpoints/"
    best_models = checkpoint + "resnet101_model"
    submit = "../submit/"
    submit_path = submit + "fuza100.json"
    gpus = "0"
    pretrained = False

    #=======================================
    #        2. Numeric parameters
    #=======================================
    epochs = 400
    batch_size = 16
    img_weight = 500
    img_height = 500
    num_classes = 6
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4

config = DefaultConfigs()
