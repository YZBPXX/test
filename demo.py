import os
import cv2
import torch

from utils.yoloface.detector_align import YoloFace
from utils.arcface.recognizor import ArcFace_Onnx
from models.arcface_proj import ArcFaceProj
from models.pipeline import SelfStableDiffusionPipeline


if __name__ == '__main__':
    device_id = 0
    device = 'cuda:%d' % device_id
    sd_dir = 'model_path'
    proj = ArcFaceProj()
    proj_ckpt = torch.load(
        os.path.join(sd_dir + 'proj.ckpt')
    )
    # proj.load_state_dict(proj_ckpt)
    proj.load_state_dict(
        {k.replace('module.', ''): v for k, v in proj_ckpt.items()}
    )
    proj = proj.to(device)

    pipe_arc = SelfStableDiffusionPipeline.from_pretrained(sd_dir)
    pipe_arc = pipe_arc.to(device)
    detector = YoloFace(device=device_id)
    recognizer = ArcFace_Onnx(device=device_id)
    print('models loaded done')

    output_path = '/tmp/_catalonia/result/'
    os.makedirs(output_path, exist_ok=True)

    image_path = '/tmp/_catalonia/faces/20230222-114355.jpeg'
    image_path = '/tmp/_catalonia/faces/20230222-114324.jpeg'
    image_path = '/tmp/_catalonia/faces/cl.jpeg'
    seed = 666

    image = cv2.imread(image_path)
    cv2.imwrite(output_path + 'ori.jpg', image)
    faces = detector.detect_and_align(image)
    if not faces:
        print('no face detected')
        raise ValueError
    cv2.imwrite(output_path + 'face.jpg', faces[0])
    embedding = recognizer.extract(faces[0])
    embedding = torch.Tensor(embedding)
    embedding = embedding.to(device)
    hiddenstates = proj(embedding)
    hiddenstates = torch.cat([hiddenstates, hiddenstates], dim=0)
    generator = torch.Generator(device).manual_seed(seed)

    prompt = 'sai style, a man watercolor and pen illustration by michal sawtyruk --v 4'

    print('start')
    for i in range(5):
        res = pipe_arc(prompt, hiddenstates, num_inference_steps=20, generator=generator, guidance_scale=4)
        image = res[0] * 255
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path + '%d.jpg' % i, image)


