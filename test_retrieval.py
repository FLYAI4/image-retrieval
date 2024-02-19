import os
from PIL import Image

import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms

import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from myexception import RetrievalException
from myerror import RetrievalErrorCode
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


wrong_model_name = "chanyoungNet"


def test_no_model_name():
    try:
        if wrong_model_name == "resnet":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model = torch.nn.Sequential(*list(model.children())[:-1])

        elif wrong_model_name == "vgg":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            flatten = nn.Flatten()
            model.add_module("Flatten", flatten)

        else:
            raise RetrievalException(**RetrievalErrorCode.NotImplementedError.value)

    except RetrievalException as e:
        assert e.log == "NotImplementedError, use 'vgg' or 'resnet'."


# # ---- test vectorize candidates ----
# 딕셔너리 저장 {'주소' : 벡터}
# # given : candidate images
# # when : after loading model
# # then : get list vectors of candidate images
image_dir = "./sample_image"
image_dict = dict()

# 이미지를 모델 입력에 맞게 전처리
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = torch.nn.Sequential(*list(model.children())[:-1])


def test_vectorize_candidate():

    # 디렉토리 내의 모든 파일에 대해 반복
    for file_path in sorted(os.listdir(image_dir)):
        # 파일의 확장자가 이미지인지 확인
        # print(file_path)
        if file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
            input_image = Image.open(os.path.join(image_dir, file_path))
            input_image = input_image.convert("RGB")
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)  # 배치 차원 추가
            model.eval()
            with torch.no_grad():
                feature_vector = model(input_batch)
                image_dict[file_path] = feature_vector
    # print(image_dict)
    # print(image_dict[file_path].size())

    assert image_dict[file_path].size() == torch.Size([1, 2048, 1, 1])


# def test_no_folder_dir():

#     assert 1


# def test_no_images_in_folder():

#     assert 1


# # ---- test inference model ----
# # given : input image
# # when : after vectorizing candidate images
# # then : get vector of input image
input_dir = "./sample_input"
input_img_name = "sample_test3.jpg"


def test_can_vectorize_input():
    input_image = Image.open(os.path.join(input_dir, input_img_name))
    input_image = input_image.convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # 배치 차원 추가
    model.eval()
    with torch.no_grad():
        feature_vector = model(input_batch)

    assert feature_vector.size() == torch.Size([1, 2048, 1, 1])


# def test_not_jpg_input():

#     assert 1


# def test_no_input():

#     assert 1


# ---- test similerity ----
# given : input picture image
# when : after crop, resize
# then : Find the most similar image
# def test_can_find_image():

#     assert 1


# def test_no_higher_than_th():

#     assert 1
