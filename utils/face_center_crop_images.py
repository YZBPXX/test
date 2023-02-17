import cv2
import random


def compute_larger_face_bbox(tube_bbox, frame_shape, increase_area=0.5):
    increase_area = random.randint(5, 9) / 10

    xc, yc, w, h = tube_bbox
    tube_bbox = [xc - w // 2, yc - h // 2, xc + w // 2, yc + h // 2]
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top

    #Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    return left, top, right, bot


def process_image(bboxes, frame):
    bbox = compute_larger_face_bbox(bboxes, frame.shape)
    x1, y1, x2, y2 = bbox
    # cv2.imwrite('/tmp/_catalonia/tmp.jpg', frame[y1:y2, x1:x2])
    return frame[y1:y2, x1:x2]


def crop_and_mask_image(bboxes, frame):
    mask = frame.copy()
    xc, yc, w, h = bboxes
    tube_bbox = [xc - w // 2, yc - h // 2, xc + w // 2, yc + h // 2]
    left, top, right, bot = tube_bbox
    mask[top:bot, left:right] = 0

    bbox = compute_larger_face_bbox(bboxes, frame.shape)
    x1, y1, x2, y2 = bbox
    # cv2.imwrite('/tmp/_catalonia/tmp.jpg', frame[y1:y2, x1:x2])
    return frame[y1:y2, x1:x2], mask[y1:y2, x1:x2]


if __name__ == '__main__':
    image_path = '/data/storage1/public/bo.zhu/datasets/text2img/mj_facedet_230111/images/1047799451122683924.jpg'
    bbox = [488, 457, 196, 247]
    image_path = '/data/storage1/public/bo.zhu/datasets/text2img/mj_facedet_230111/images/1047798286842605628.jpg'
    bbox = [704,380,193,279]
    image_path = '/data/storage1/public/bo.zhu/datasets/text2img/mj_facedet_230111/images/1047798426580041768.jpg'
    bbox = [592, 295, 169, 202]
    image = cv2.imread(image_path)
    process_image(bbox, image)


