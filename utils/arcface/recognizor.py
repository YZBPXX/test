import onnxruntime
import numpy as np
import torch
from utils.arcface.pytorch_arcface import resnet_face18


class ArcFace_Onnx:
    def __init__(self, device=0, pretrained='/data/storage1/nas/bo.zhu/checkpoints/arcface/webface_r50.onnx'):
        device_ep = 'CUDAExecutionProvider'
        ep_flags = {}
        ep_flags['device_id'] = device
        self.session = onnxruntime.InferenceSession(pretrained, providers=[(device_ep, ep_flags)])

    @staticmethod
    def pre_process(img):
        img = img[None, ...]
        img = img.transpose((0, 3, 1, 2))
        img = img.astype(np.float32)
        img /= 255.0
        img -= 0.5
        img /= 0.5
        return img

    def extract(self, face):
        face = self.pre_process(face)
        y_onnx = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: face})[0]
        # print(y_onnx.shape)
        return y_onnx

    def extract_faces(self, faces):
        y_onnx = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: faces})[0]
        return y_onnx

    # def run(self, img, pred):


class ArcFace_Pytorch:
    def __init__(self, pretrained='/mnt/share/resnet18_110.pth'):
        # device_ep = 'CUDAExecutionProvider'
        # ep_flags = {}
        # ep_flags['device_id'] = device
        # self.session = onnxruntime.InferenceSession(pretrained, providers=[(device_ep, ep_flags)])
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        self.arcface_model = resnet_face18()
        self.arcface_model.load_state_dict(torch.load(pretrained))
        print("Loading pretrained pytorch ArcFace resnet18_110.pth")
        self.arcface_model.to(device)

    @staticmethod
    def pre_process(img):
        img = img[None, ...]
        img = img.transpose((0, 3, 1, 2))
        img = img.astype(np.float32)
        img /= 255.0
        img -= 0.5
        img /= 0.5
        return img

    def extract(self, face):
        face = self.pre_process(face)
        # y_onnx = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: face})[0]
        # print(y_onnx.shape)
        y_pytorch = self.arcface_model(face)
        return y_pytorch

    def extract_faces(self, faces):
        # y_onnx = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: faces})[0]
        y_pytorch = self.arcface_model(faces)
        return y_pytorch

