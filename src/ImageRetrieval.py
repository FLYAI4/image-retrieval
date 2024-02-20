import os
from PIL import Image

import torch
import torchvision.transforms as transforms

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

from src.myexception import RetrievalException
from src.myerror import RetrievalErrorCode


class RetrievalManager:
    def __init__(self, user_folder_path: str) -> None:
        self.user_folder_path = user_folder_path

    def load_pretrained_model(self, model_name="resnet"):
        """
        사전 학습된 feature extractor 부분을 로드하는 함수
        Params:
            - model_name : which model to use "resnet" or "vgg"
        Return:
            - model object : without FC layer
        Description:
            - you can only select resnet or vgg pre-trained without fc layer
        """

        if model_name == "resnet":
            model = torch.load("./models/resnet50_model.pth")

        elif model_name == "vgg":
            model = torch.load("./models/vgg16_model.pth")

        else:
            raise RetrievalException(**RetrievalErrorCode.WrongModelError.value)

        return model

    def load_vectorized_db_images(self, json_path=None):
        """
        load Json file, which is custom dictionary of image : vector(feature)
        Params:
            - json_path : your json file path
        Return:
            - dict : {image_path : vectorized_image}
        Description:
            - you should make json dictionary first,
            please check vectorize_db_image.py
        """
        try:
            with open(json_path, 'r') as f:
                vectorized_db_dict = json.load(f)
        except Exception as e:
            raise RetrievalException(**RetrievalErrorCode.WrongJsonError.value,
                                     error=e)
        return vectorized_db_dict

    def preprocessing_image(self, image):
        """
        image preprocessing to input CNN model
        Params:
            - image : PIL Image object
        Return:
            - image tensor with batch size 1
        Description:
            - only one input image is possible
        """
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
        """
        computing image to vector(feature)
        Params:
            - input_path : your image file path
            - model : feature extractor (resnet or vgg)
        Return:
            - vector : extracted feature of user image input
        """
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
        """
        computing and extract most similar image with input feature
        Params:
            - feature_vector : vectorized user input image
            - json_path : json file path
        Return:
            - most similar image path
        Description:
            - onlt one image is possible, make json file first
        """
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
