from enum import Enum


class RetrievalErrorCode(Enum):
    NotImplementedError = {
        "code": 400,
        "message": "This model is not avaliable.",
        "log": "NotImplementedError, use 'vgg' or 'resnet'."
    }
