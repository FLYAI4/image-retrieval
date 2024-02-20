from enum import Enum


class RetrievalErrorCode(Enum):
    WrongModelError = {
        "code": 400,
        "message": "This model is not avaliable.",
        "log": "WrongModelError, use 'vgg' or 'resnet'."
    }
    WrongJsonError = {
        "code": 400,
        "message": "Something wrong with your Json, check your json path",
        "log": "Something wrong with Json file or Json dir"
    }
    FileNotFoundError = {
        "code": 404,
        "message": "File Not Found, check path",
        "log": "File not found check path"
    }
