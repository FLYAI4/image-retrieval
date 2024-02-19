# --- setup ---
# candidate images folder path 
# 정상 이미지 path 
# 

# ----- loading model test ----
# given : model name
# when : setup retrieval
# then : load pre-trained model e.g. ResNet, VGG
def test_load_pretrained():

    assert 1


def test_no_model_name():

    assert 1


# ---- test vectorize candidates ----
# given : candidate images
# when : after loading model
# then : get list vectors of candidate images
def test_vectorize_candidate():

    assert 1


def test_no_folder_dir():

    assert 1


def test_no_images_in_folder():

    assert 1


def test_not_jpg_format():

    assert 1


# ---- test inference model ----
# given : input image
# when : after vectorizing candidate images
# then : get vector of input image
def test_can_vectorize_input():

    assert 1


def test_not_jpg_input():

    assert 1


def test_no_input():

    assert 1


# ---- test similerity ----
# given : input picture image
# when : after crop, resize
# then : Find the most similar image
def test_can_find_image():

    assert 1


def test_no_higher_than_th():

    assert 1
