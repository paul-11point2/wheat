import albumentations as A
import albumentations.augmentations.geometric.functional as F
import cv2
import numpy as np

class ResizeWithinBounds(A.DualTransform):
    """Rescale an image so that the aspect ratio is maintained.

    Args:
        max_height int: maximum height of the image after the transformation
        max_width int: maximum width of the image after the transformation
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        max_height: int = 2160,
        max_width: int = 4096,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1,
    ):
        super(ResizeWithinBounds, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_height = max_height
        self.max_width = max_width

    def apply(
        self, img: np.ndarray, max_height: int = 2160, max_width: int = 4096, interpolation: int = cv2.INTER_LINEAR, **params
    ) -> np.ndarray:
        scale_factor = min(max_height / img.shape[0], max_width / img.shape[1])
        return F.resize(img, height=int(img.shape[0] * scale_factor), width=int(img.shape[1] * scale_factor), interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]
        scale_x = self.max_width / width
        scale_y = self.max_height / height
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return ("max_height", "max_width", "interpolation")