import math
import torch
import numpy as np
import torch.nn.functional as F


class GridSampler:
    @staticmethod
    def euclidean_distance(source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def cal_angle(self, left_eye, right_eye):
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
            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = np.arccos(cos_a)
            # angle = (angle * 180) / math.pi
            if direction == -1:
                angle = math.pi / 2 - angle
            angle = direction * angle
        else:
            angle = 0

        return angle

    @staticmethod
    def crop_tensor(image_tensor, bbox):
        x1, y1, x2, y2 = bbox
        face_tensor = image_tensor[:, :, y1:y2, x1:x2]
        return face_tensor

    @staticmethod
    def pred2grid(pred):
        box = pred[:4].astype(int)
        xc, yc, w, h = box
        (x1, y1), (x2, y2) = (xc - w // 2, yc - h // 2), (xc + w // 2, yc + h // 2)

        points = pred[5:15].astype(int)
        points = points.reshape((5, 2))

        top = yc - h // 2
        left = xc - w // 2
        left_eye = points[0]
        left_eye[0] -= top
        left_eye[1] -= left
        right_eye = points[1]
        right_eye[0] -= top
        right_eye[1] -= left

        return [x1, y1, x2, y2], [left_eye, right_eye]

    @staticmethod
    def grid_samle(face_tensor, angle):
        b, c, h, w = face_tensor.shape
        theta = [
            # [np.cos(angle) * w / h, -np.sin(angle), 0.],
            # [np.sin(angle), np.cos(angle) * w / h, 0.]
            [np.cos(angle), -np.sin(angle), 0.],
            [np.sin(angle), np.cos(angle), 0.]
        ]
        theta = torch.Tensor(theta).unsqueeze(0)
        # grid is of size NxHxWx2
        grid = F.affine_grid(theta, face_tensor.size(), align_corners=False)
        x = F.grid_sample(face_tensor, grid, align_corners=False)
        return x

    def run_face(self, image_tensor, bbox, landmarks):
        face_tensor = self.crop_tensor(image_tensor, bbox)
        left_eye, right_eye = landmarks
        angle = self.cal_angle(left_eye, right_eye)
        aligned_face_tensor = self.grid_samle(face_tensor, angle)
        return aligned_face_tensor

    def run(self, image, pred):
        bbox, landmarks = self.pred2grid(pred)
        x = torch.from_numpy(original)
        x = x.type(torch.FloatTensor)
        x = torch.unsqueeze(x, 0)
        x = x.permute(0, 3, 1, 2)
        face_tesnor = self.grid_sampler.run(x, bbox, landmarks)

        face_tensor = self.crop_tensor(image_tensor, bbox)
        left_eye, right_eye = landmarks
        angle = self.cal_angle(left_eye, right_eye)
        aligned_face_tensor = self.grid_samle(face_tensor, angle)
        return aligned_face_tensor



