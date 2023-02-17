import cv2
import math
import numpy as np
from PIL import Image


def euclidean_distance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def pad2square(img):
    h, w, _ = img.shape
    if h > w:
        img_pad = cv2.copyMakeBorder(img, 0, 0, (h - w) // 2, h - w - (h - w) // 2, cv2.BORDER_CONSTANT)
    else:
        img_pad = cv2.copyMakeBorder(img, (w - h) // 2, w - h - (w - h) // 2, 0, 0, cv2.BORDER_CONSTANT)
    return img_pad


def alignment_procedure(img, left_eye, right_eye):
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1

    a = euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = euclidean_distance(np.array(right_eye), np.array(left_eye))

    if b != 0 and c != 0:
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)
        angle = (angle * 180) / math.pi

        if direction == -1:
            angle = 90 - angle
        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    img = pad2square(img)
    img = cv2.resize(img, (112, 112))

    return img