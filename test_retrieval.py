import torch
import torchvision.models as models
import torch.nn as nn

# --- setup ---
# candidate images folder path
# 정상 이미지 path
# model name


# ----- loading model test ----
# given : model name
# when : setup retrieval
# then : load pre-trained model e.g. ResNet, VGG
pre_trained_model_name = "resnet"


def test_load_pretrained():
    if pre_trained_model_name == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model = torch.nn.Sequential(*list(model.children())[:-1])

    elif pre_trained_model_name == "vgg":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        flatten = nn.Flatten()
        model.add_module("Flatten", flatten)

    else:
        raise NotImplementedError

    model.eval()
    example_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        features = model(example_input)

    if pre_trained_model_name == "resnet":
        assert features.dim() == 4
    elif pre_trained_model_name == "vgg":
        assert features.dim() == 2


# def test_no_model_name():

#     assert 1


# # ---- test vectorize candidates ----
# # given : candidate images
# # when : after loading model
# # then : get list vectors of candidate images
# def test_vectorize_candidate():

#     assert 1


# def test_no_folder_dir():

#     assert 1


# def test_no_images_in_folder():

#     assert 1


# def test_not_jpg_format():

#     assert 1


# # ---- test inference model ----
# # given : input image
# # when : after vectorizing candidate images
# # then : get vector of input image
# def test_can_vectorize_input():

#     assert 1


# def test_not_jpg_input():

#     assert 1


# def test_no_input():

#     assert 1


# # ---- test similerity ----
# # given : input picture image
# # when : after crop, resize
# # then : Find the most similar image
# def test_can_find_image():

#     assert 1


# def test_no_higher_than_th():

#     assert 1
