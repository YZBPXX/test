import cv2
import math
import onnxruntime
import numpy as np
from PIL import Image


class YoloFace:
    def __init__(self, device=0, pretrained='utils/yoloface/yolov5n-0.5.onnx'):
        device_ep = 'CUDAExecutionProvider'
        ep_flags = {}
        ep_flags['device_id'] = device
        self.session = onnxruntime.InferenceSession(pretrained, providers=[(device_ep, ep_flags)])

    @staticmethod
    def euclidean_distance(source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def alignment_procedure(self, img, left_eye, right_eye):
        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye

        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1

        a = self.euclidean_distance(np.array(left_eye), np.array(point_3rd))
        b = self.euclidean_distance(np.array(right_eye), np.array(point_3rd))
        c = self.euclidean_distance(np.array(right_eye), np.array(left_eye))

        if b != 0 and c != 0:
            cos_a = (b*b + c*c - a*a)/(2*b*c)
            angle = np.arccos(cos_a)
            angle = (angle * 180) / math.pi

            if direction == -1:
                angle = 90 - angle
            img = Image.fromarray(img)
            img = np.array(img.rotate(direction * angle))

        img = self.pad2square(img)
        img = cv2.resize(img, (112, 112))

        return img

    @staticmethod
    def py_cpu_nms(dets, conf=0.8, thresh=0.1):
        dets = dets[dets[..., 4] > conf]
        x1 = dets[:, 0] - dets[:, 2] // 2
        y1 = dets[:, 1] - dets[:, 3] // 2
        x2 = dets[:, 0] + dets[:, 2] // 2
        y2 = dets[:, 1] + dets[:, 3] // 2
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return dets[keep]

    @staticmethod
    def pad2square(img):
        h, w, _ = img.shape
        if h > w:
            img_pad = cv2.copyMakeBorder(img, 0, 0, (h - w) // 2, h - w - (h - w) // 2, cv2.BORDER_CONSTANT)
        else:
            img_pad = cv2.copyMakeBorder(img, (w - h) // 2, w - h - (w - h) // 2, 0, 0, cv2.BORDER_CONSTANT)
        return img_pad

    @staticmethod
    def pre_process(img):
        img = img[None, ...]
        img = img.transpose((0, 3, 1, 2))
        img = img.astype(np.float32)
        img /= 255.0
        return img

    def detect(self, img, conf=0.3, img_size=256):
        ori_h, ori_w, _ = img.shape
        img = self.pad2square(img)
        bg = cv2.resize(img, (img_size, img_size))
        img = self.pre_process(bg)
        y_onnx = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: img})[0]
        pred = self.py_cpu_nms(y_onnx, conf)

        scale = (max(ori_w, ori_h)) / img_size
        pred *= scale
        if ori_w >= ori_h:
            padding = (ori_w - ori_h) / 2
            pred[:, 1] -= padding
            pred[:, 4::2] -= padding
        else:
            padding = (ori_h - ori_w) / 2
            pred[:, 0] -= padding
            pred[:, 5::2] -= padding
        pred.astype(int)
        return pred

    def detect_and_align(self, ori_img, conf=0.3, img_size=256):
        ori_h, ori_w, _ = ori_img.shape
        img = self.pad2square(ori_img)
        bg = cv2.resize(img, (img_size, img_size))
        img = self.pre_process(bg)
        y_onnx = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: img})[0]
        pred = self.py_cpu_nms(y_onnx, conf=conf)

        scale = (max(ori_w, ori_h)) / img_size
        pred *= scale
        if ori_w >= ori_h:
            padding = (ori_w - ori_h) / 2
            pred[:, 1] -= padding
            pred[:, 4::2] -= padding
        else:
            padding = (ori_h - ori_w) / 2
            pred[:, 0] -= padding
            pred[:, 5::2] -= padding
        pred.astype(int)
        faces = []

        for i, p in enumerate(pred):
            box = p[:4].astype(int)
            xc, yc, w, h = box
            points = pred[i][5:15]
            points = points.reshape((5, 2))

            top = yc - h // 2
            left = xc - w // 2
            face = ori_img[yc - h // 2:yc + h // 2, xc - w // 2:xc + w // 2]

            left_eye = points[0]
            left_eye[0] -= top
            left_eye[1] -= left
            right_eye = points[1]
            right_eye[0] -= top
            right_eye[1] -= left
            try:
                face = self.alignment_procedure(face, left_eye, right_eye)
                faces.append(face)
            except Exception as e:
                print(e)
                continue
        return faces


if __name__ == '__main__':
    from tqdm import tqdm
    d = YoloFace(0, 'yolov5n-0.5.onnx')
    with open('/data/storage1/public/bo.zhu/datasets/text2img/train_0218.idx', 'r') as f:
        image_files = f.readlines()
        image_files = [file[:-1] for file in image_files]

    with open('/data/storage1/public/bo.zhu/datasets/text2img/train_face_0218.idx', 'w') as f:
        for image_file in tqdm(image_files):
            image = cv2.imread(image_file)
            pred = d.detect(image)
            if len(pred):
                f.write(image_file + '\n')

