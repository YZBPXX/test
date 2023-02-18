import json
import logging
import os
import io
import warnings
import cv2
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.nn import functional as F
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets.arrow_dataset import Dataset
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, EulerAncestralDiscreteScheduler
from torchvision import transforms
from datas.image_file_dataset import ImageData
from models.arcface_proj import ArcFaceProj
from utils.arcface.arcface_res50_model import Arcface
from utils.read_dir import get_file_from_dir
from utils.arcface.recognizor import ArcFace_Onnx
from utils.arcface.face_align import alignment_procedure
from utils.face_center_crop_images import process_image, crop_and_mask_image
from utils.yoloface.detector_align import YoloFace
from utils.face_region_grid_sample import GridSampler
from utils.read_dir import get_file_from_dir


torch.set_num_threads(4)
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore')
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class Trainer:
    def __init__(self, args):
        self.args = args
        logging_dir = os.path.join(self.args.output_dir, self.args.logging_dir)
        self.logger = get_logger(__name__)
        self.RANK = int(os.environ.get('RANK'))
        self.weight_dtype = torch.float32

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            logging_dir=logging_dir,
        )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("text2image-fine-tune", config=vars(self.args))
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)

        # Load models and create wrapper for stable diffusion
        self.tokenizer = CLIPTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path + "/text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")
        self.proj = ArcFaceProj()
        ckpt = torch.load(self.args.pretrained_model_name_or_path + 'proj.ckpt')
        self.proj.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
        self.detector = YoloFace(self.RANK % 8)
        self.recognizer = ArcFace_Onnx(self.RANK % 8)
        self.recognizer_pth = Arcface()
        # self.image_files = get_file_from_dir(
        #     '/data/storage1/public/bo.zhu/datasets/text2img/mj_yzb_0213/'
        # )
        with open('/data/storage1/public/bo.zhu/datasets/text2img/train_0218.idx', 'r') as f:
            image_files = f.readlines()
            self.image_files = [file[:-1] for file in image_files]

        # Freeze vae and text_encoder
        self.vae.requires_grad_(True)
        self.text_encoder.requires_grad_(False)
        self.recognizer_pth.requires_grad_(True)

        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.recognizer_pth.to(self.accelerator.device, dtype=self.weight_dtype)
        self.grid_sampler = GridSampler()

        # Initialize the optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        self.optimizer_proj = torch.optim.AdamW(
            self.proj.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )
        self.lr_scheduler_proj = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer_proj,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )

        # TODO (patil-suraj): load scheduler using args
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
        )
        self.betas = (
            torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float32) ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.vae_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.args.resolution, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.yolo_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.args.resolution, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
            ]
        )
        self.arcface_transform = transforms.Compose(
            [
                 transforms.ToPILImage(),
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(
                     brightness=0.1,
                     contrast=0.1,
                     saturation=0.1,
                 ),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ]
        )

        self.image_dataset = ImageData(self.image_files, self.vae_transforms, self.yolo_transforms, self.tokenizer)
        self.train_dataloader = iter(
            torch.utils.data.DataLoader(
                self.image_dataset, shuffle=True, collate_fn=self.collate_fn, batch_size=1
            )
        )
        self.unet, self.proj, self.optimizer, self.optimizer_proj, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.proj, self.optimizer, self.optimizer_proj, self.train_dataloader, self.lr_scheduler
        )

    def collate_fn(self, examples):
        examples = [example for example in examples if example is not None]
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        detector_input = torch.stack([example["detector_input"] for example in examples])
        detector_input = detector_input.to(memory_format=torch.contiguous_format).float()
        input_ids = [example["input_ids"] for example in examples]
        padded_tokens = self.tokenizer.pad({"input_ids": input_ids}, padding='max_length', return_tensors="pt", max_length=self.tokenizer.model_max_length)
        return {
            "pixel_values": pixel_values,
            "detector_input": detector_input,
            "input_ids": padded_tokens.input_ids,
        }

    def tokenize_captions(self, examples):
        captions = []
        for caption in examples['txt']:
            captions.append(caption)
        inputs = self.tokenizer(captions, max_length=self.tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids

    def sub_noise(
        self,
        noisy_latents: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        self.alphas_cumprod = self.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        timesteps = timesteps.to(noisy_latents.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(noisy_latents.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_latents.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        latents = (noisy_latents - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod
        return latents

    def train(self):
        # Train!
        total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        self.logger.info("***** Running training *****")
        # self.logger.info(f"  Num examples = {len(self.train_dataloader)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        for epoch in range(self.args.num_train_epochs):
            self.unet.train()
            self.proj.train()
            train_loss = 0.0

            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # Convert images to latent space
                    pixel_values = batch["pixel_values"].to(self.accelerator.device, dtype=self.weight_dtype)
                    latents = self.vae.encode(pixel_values.to(self.accelerator.device, dtype=self.weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215
                    detector_input = batch["detector_input"].numpy().permute(0, 2, 3, 1)

                    preds = [self.detector.detect_and_align(image) for image in detector_input]
                    inds_input = [True if len(pred) == 1 else False for pred in preds]
                    faces = [pred[0] for pred in preds if len(pred) == 1]
                    faces = np.array(faces)
                    faces = np.array([self.arcface_transform(face) for face in faces])
                    face_embeddings = torch.Tensor(self.recognizer.extract_faces(faces)).to(self.accelerator.device)
                    embeddings = self.proj(face_embeddings)

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)
                    timesteps = timesteps.long()
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    encoder_hidden_states = self.text_encoder(batch["input_ids"].to(self.accelerator.device))[0]
                    encoder_hidden_states[inds_input] = torch.cat([embeddings, encoder_hidden_states[inds_input]], dim=1)

                    # Predict the noise residual and compute loss
                    noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    latents = self.sub_noise(noisy_latents[inds_input], noise[inds_input], timesteps[inds_input])
                    latents = 1 / 0.18215 * latents
                    latents = self.vae.decode(latents).sample
                    latents = (latents / 2 + 0.5).clamp(0, 1)
                    images = latents.copy().detach().cpu().permute(0, 2, 3, 1).float().numpy()
                    images = (images * 255).round().astype("uint8")
                    images = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images]
                    all_preds = []
                    for image in images:
                        try:
                            preds = self.detector.detect(image)
                        except Exception as e:
                            all_preds.append(None)
                            continue
                        if preds:
                            all_preds.append(preds[0])
                        else:
                            all_preds.append(None)

                    inds_output = [False if pred is None else True for pred in all_preds]
                    # inds_cal_loss = [i & j for i, j in zip(inds_input, inds_output)]
                    embeddings_gt = face_embeddings[inds_input][inds_output]

                    embeddings = []
                    for face_ind, pred in enumerate(all_preds):
                        if pred is not None:
                            face = self.grid_sampler.run(latents[face_ind], pred)
                            face /= 255.0
                            face -= 0.5
                            face /= 0.5
                            embedding = torch.Tensor(self.recognizer.extract(face)).to(
                                memory_format=torch.contiguous_format).float().to(self.accelerator.device)
                        else:
                            embedding = embeddings_gt[face_ind].unsqueeze(0).to(self.accelerator.device)
                        embeddings.append(embedding)

                    # embeddings = torch.Tensor(self.recognizer.extract_faces(faces))
                    embeddings = torch.concat(embeddings, dim=0)
                    # print(source_embeddings.shape, embeddings.shape)
                    cos_sim = F.cosine_similarity(embeddings_gt, embeddings, dim=0)

                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") + cos_sim

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / self.args.gradient_accumulation_steps

                    # Backpropagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), self.args.max_grad_norm)
                        self.accelerator.clip_grad_norm_(self.proj.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()
                    self.optimizer_proj.step()
                    self.lr_scheduler.step()
                    self.lr_scheduler_proj.step()
                    self.optimizer.zero_grad()
                    self.optimizer_proj.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                logs = {"step_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= self.args.max_train_steps:
                    self.accelerator.end_training()
                    return

                if self.RANK == 0 and global_step % 5000 == 0 and global_step != 0:
                    pipeline = StableDiffusionPipeline(
                        text_encoder=self.text_encoder,
                        vae=self.vae,
                        unet=self.accelerator.unwrap_model(self.unet),
                        tokenizer=self.tokenizer,
                        scheduler=EulerAncestralDiscreteScheduler(
                            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
                        ),
                        safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                            "CompVis/stable-diffusion-safety-checker"),
                        feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
                    )
                    os.makedirs(self.args.output_dir + '/%d_%d/' % (1, global_step), exist_ok=True)
                    pipeline.save_pretrained(self.args.output_dir + '/%d_%d/' % (1, global_step))
                    torch.save(
                        self.proj.state_dict(),
                        self.args.output_dir + '/%d_%d/' % (1, global_step) + 'proj.ckpt',
                    )

        self.accelerator.end_training()

