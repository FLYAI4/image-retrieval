import os
from PIL import Image

import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

from myexception import RetrievalException
from myerror import RetrievalErrorCode


def load_pretrained_model(model_name="resnet"):

    if model_name == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model = torch.nn.Sequential(*list(model.children())[:-1])

    elif model_name == "vgg":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        flatten = nn.Flatten()
        model.add_module("Flatten", flatten)

    else:
        raise RetrievalException(**RetrievalErrorCode.WrongModelError.value)

    return model


def load_vectorized_db_images(json_path=None):
    try:
        with open(json_path, 'r') as f:
            vectorized_db_dict = json.load(f)
    except Exception as e:
        raise RetrievalException(**RetrievalErrorCode.WrongJsonError.value, 
                                 error=e)
    return vectorized_db_dict


user_folder_path = "sample_input"


def preprocessing_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    input_image = image.convert("RGB")
    input_tensor = preprocess(input_image)
    # 배치 차원 추가
    input_batch = input_tensor.unsqueeze(0)

    return input_batch


# TODO : input image name 이 없을 때 처리
def vectorize_image(input_path=None, model=None):
    input_image = Image.open(os.path.join(user_folder_path, input_path))
    input_batch = preprocessing_image(input_image)
    model.eval()
    with torch.no_grad():
        feature_vector = model(input_batch)
    return feature_vector.squeeze()


# TODO : JSON Path 를 매개변수로 받는게 아니라 DB dict를 받아야함
def compute_similarity(feature_vector=None, json_path=None):
    db_dict = load_vectorized_db_images(json_path)
    input_feature = feature_vector.cpu().numpy()
    similarities = dict()

    for path, vec in db_dict.items():
        feature = np.array(vec)
        similarity = cosine_similarity(input_feature.reshape(1, -1),
                                       feature.reshape(1, -1))
        similarities[path] = similarity

    # 가장 유사한 feature의 인덱스 찾기
    most_similar_img = max(similarities, key=similarities.get)

    return most_similar_img
