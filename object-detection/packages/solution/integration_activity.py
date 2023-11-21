from typing import Tuple

def DT_TOKEN() -> str:
    dt_token = "dt1-3nT7FDbT7NLPrXykNJW6pwkBnqd6m6xEgPUtPkStjancVCg-43dzqWFnWd8KBa1yev1g3UKnzVxZkkTbfj33LGDLGg3dfoUaYQpezRtXj5WaBJudv9"
    return dt_token


def MODEL_NAME() -> str:
    # Change this to your model's name that you used to upload it on google colab.
    # if you didn't change it, it should be "yolov5n"
    return "yolov5n"


def NUMBER_FRAMES_SKIPPED() -> int:
    # Change this number to drop more frames
    # (must be a positive integer)
    return 3


def filter_by_classes(pred_class: int) -> bool:
    """
    Remember the class IDs:

        | Object    | ID    |
        | ---       | ---   |
        | Duckie    | 0     |
        | Cone      | 1     |
        | Truck     | 2     |
        | Bus       | 3     |


    Args:
        pred_class: the class of a prediction
    """
    #print("Pred class",pred_class)
    return pred_class == 0 


def filter_by_scores(score: float) -> bool:
    """
    Args:
        score: the confidence score of a prediction
    """
    #print("Score:",score)
    return score > 0.2


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    """
    Args:
        bbox: is the bounding box of a prediction, in xyxy format
                This means the shape of bbox is (leftmost x pixel, topmost y, rightmost x, bottommost y)
    """
    img_width=640
    img_height=480
    min_bb_size=1000
    left_right_buffer=50
    bottom_threshold = int(0.4*img_height)

    left_x = bbox[0]
    right_x = bbox[2]
    top_y = bbox[1]
    bottom_y = bbox[3]

    box_size = (right_x-left_x)*(bottom_y-top_y)
    #print(top_y,bottom_y)
    #print("size:",box_size,box_size > min_bb_size)

    return box_size > min_bb_size \
           and bottom_y > bottom_threshold \
           and left_x > left_right_buffer \
           and right_x < img_width - left_right_buffer
