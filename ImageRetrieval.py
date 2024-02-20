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


class RetrievalManager:
    def __init__(self, user_folder_path: str) -> None:
        self.user_folder_path = user_folder_path

    def load_pretrained_model(self, model_name="resnet"):

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

    def load_vectorized_db_images(self, json_path=None):
        try:
            with open(json_path, 'r') as f:
                vectorized_db_dict = json.load(f)
        except Exception as e:
            raise RetrievalException(**RetrievalErrorCode.WrongJsonError.value,
                                     error=e)
        return vectorized_db_dict

    def preprocessing_image(self, image):
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

    def vectorize_image(self, input_path=None, model=None):
        if os.path.exists(os.path.join(self.user_folder_path, input_path)):
            input_image = Image.open(os.path.join(self.user_folder_path,
                                                  input_path))
            input_batch = self.preprocessing_image(input_image)
            model.eval()
            with torch.no_grad():
                feature_vector = model(input_batch)
            return feature_vector.squeeze()
        else:
            raise RetrievalException(**RetrievalErrorCode.ImageNotFoundError.value)

    def compute_similarity(self, feature_vector=None, json_path=None):
        db_dict = self.load_vectorized_db_images(json_path)
        input_feature = feature_vector.cpu().numpy()
        similarities = dict()

        for path, vec in db_dict.items():
            feature = np.array(vec)
            similarity = cosine_similarity(input_feature.reshape(1, -1),
                                           feature.reshape(1, -1))
            similarities[path] = similarity

        # 가장 유사한 feature의 인덱스 찾기
        most_similar_img = max(similarities, key=similarities.get)
        # max 값을 받아서 th 이하이면 raise Exception
        similarity_level = max(similarities.values())
        threshold = 0.65
        print(f'Most similar img {most_similar_img} with Confidence Level {similarity_level}')

        if similarity_level < threshold:
            raise RetrievalException(**RetrievalErrorCode.LowConfidenceError.value)
        else:
            return most_similar_img
