import image_retrieval
from myexception import RetrievalException
import os
# from myerror import RetrievalErrorCode


# -----사전 학습 모델 로딩 테스트 ----
wrong_model_name = "chanyoungNet"


def test_wrong_model_name():
    try:
        image_retrieval.load_pretrained_model(model_name=wrong_model_name)
    except RetrievalException as e:
        assert e.log == "WrongModelError, use 'vgg' or 'resnet'."


# ----Json vector dict 로딩 테스트 ----
wrong_json_path = "./chanyoung.json"


def test_wrong_json_dir():
    try:
        image_retrieval.load_vectorized_db_images(json_path=wrong_json_path)
    except RetrievalException as e:
        assert e.log == "Something wrong with Json file or Json dir"


# ---- test inference model ----
# # given : input image
# # when : after vectorizing candidate images
# # then : get vector of input image
input_dir = "./sample_input"
input_img_name = "sample_test3.jpg"


# File not found check path
wrong_image_path = "chanyoung.jpg"


def test_wrong_input_to_vectorize():
    try:
        image_retrieval.vectorize_image(input_path=wrong_image_path)
    except RetrievalException as e:
        assert e.log == "File not found check path"


# ---- test similerity ----
# given : input picture image
# when : after crop, resize
# then : Find the most similar image
json_path = 'image_dict.json'
input_img_name = "sample6_2.jpg"
input_dir = "./sample_input"
db_dir = "./sample_image"


def test_can_find_image():
    # load model
    model = image_retrieval.load_pretrained_model(model_name="resnet")
    feature = image_retrieval.vectorize_image(input_path=input_img_name,
                                              model=model)
    most_similar = image_retrieval.compute_similarity(feature_vector=feature,
                                                      json_path=json_path)
    assert os.path.exists(os.path.join(db_dir, most_similar))

# TODO : th 홀드 값 작성
# def test_no_higher_than_th():

#     assert 1
