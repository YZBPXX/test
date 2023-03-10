import json
import cv2
import numpy as np
from PIL import Image
from torch.utils import data as data


class ImageData(data.Dataset):
    def __init__(self, files, trans, trans2, tokenizer):
        super(ImageData, self).__init__()
        self.files = files
        self.vae_transforms = trans
        self.yolo_transforms = trans2
        self.tokenizer = tokenizer

    @staticmethod
    def load_prompt(json_path):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        prompt = data['title']
        return prompt

    def __getitem__(self, data_index):
        try:
            # load image
            image = Image.open(self.files[data_index]).convert("RGB")
            detector_input = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            pixel_values = self.vae_transforms(image)
            detector_input = self.yolo_transforms(detector_input)

            # load prompt
            prompt_path = self.files[data_index].replace('.jpg', '.json')
            prompt = self.load_prompt(prompt_path)

            inputs = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length,
                padding="do_not_pad",
                truncation=True
            )
            input_ids = inputs.input_ids

        except Exception as e:
            print(self.files[data_index], e)
            return None

        return {
            "pixel_values": pixel_values,
            "detector_input": detector_input,
            "input_ids": input_ids,
        }

    def __len__(self):
        return len(self.files)

