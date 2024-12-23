import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms



class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True, **kwargs):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        ori_image = transforms.ToTensor()(image)
        image = self.image_transform(image)
        if self.is_train:
            return image
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                image_file = image_file.replace("test", "ground_truth").replace(".png", "_mask.png")
                target = Image.open(image_file)
                target = self.target_transform(target)
            return ori_image, image, target

    def __len__(self):
        return len(self.image_files)
    
    def jigsaw_puzzle(self, patch):
         return patch



class PlateDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True, is_visualize=False):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.470, 0.471, 0.471], [0.0559, 0.0543, 0.0516]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.jpg")
            )
        else:
            self.image_files = glob(
                os.path.join(root, category, "test", "*", "*.jpg")
            )
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train
        self.is_visualize = is_visualize
        self.get_segments()
        # self._compute_mean_stds()
        return
        

    def __getitem__(self, index):
        image = self.image_patches[index]
        ori_image = transforms.ToTensor()(image)
        image = self.image_transform(image)
        if self.is_train:
            return image
        else:
            target = self.target_patches[index]
            if target is not None:
                target = self.target_transform(target)
            else:
                target = torch.zeros([1, image.shape[-2], image.shape[-1]]) - 1.0
            return ori_image, image, target


    def __len__(self):
        return len(self.image_patches)
    
    
    def get_segments(self):
        self.image_patches, self.target_patches = [], []
        
        for image_file in self.image_files:
            image = Image.open(image_file)            
            image = image.crop((0, 0, 1200, 672))
            width, height = image.size

            target_file = image_file.replace("test", "ground_truth").replace(".jpg", "_mask.jpg")
            if os.path.exists(target_file):
                target = Image.open(target_file)
                target = target.crop((0, 0, width, height))
            else:
                if not self.is_visualize and not self.is_train:
                    continue
                else:
                    target = torch.zeros([1, width, height])
                    target = transforms.ToPILImage()(target)
                            
            patch_width, patch_height = (400, 224)
            patch_edge_max = max(patch_height, patch_width)
            for top in range(0, height, patch_height):
                for left in range(0, width, patch_width):
                    box_right = left + patch_width
                    box_bottom = top + patch_height
                    if box_right > width or box_bottom > height:
                        continue
                    
                    box = (left, top, box_right, box_bottom)
                    paste_left = (patch_edge_max - patch_width) // 2
                    paste_top = (patch_edge_max - patch_height) // 2
                    
                    # image
                    patch = image.crop(box)
                    padded_patch = Image.new("RGB", (patch_edge_max, patch_edge_max), (0, 0, 0))
                    padded_patch.paste(patch, (paste_left, paste_top))
                    
                    # target
                    patch_target = target.crop(box)
                    padded_patch_target = Image.new("L", (patch_edge_max, patch_edge_max), 0)
                    padded_patch_target.paste(patch_target, (paste_left, paste_top))
                    
                    self.image_patches.append(padded_patch)
                    self.target_patches.append(padded_patch_target)
                    
        self.img_width, self.img_height = width, height
        self.patch_width, self.patch_height = patch_width, patch_height
        self.patch_edge = patch_edge_max
        self.patch_cols, self.patch_rows = (width // patch_width),  (height // patch_height)
        self.patch_nums = self.patch_cols * self.patch_rows
        self.complete_jigsaw = Image.new('RGB', (self.img_width, self.img_height), color='white')
        return
    
    
    def _compute_mean_stds(self):
        image_transform = transforms.Compose([transforms.ToTensor(),])
        image_list = []
        for image_file in self.image_files:
            image = Image.open(image_file)            
            image = image.crop((0, 0, 1200, 672))
            image = image_transform(image)
            image = torch.reshape(image, shape=(3, -1))
            image_list.append(image)
        images = torch.cat(image_list, dim=-1)
        means = torch.mean(image, dim=-1)
        stds = torch.std(images, dim=-1)
        print(means, stds)
        return
        

    def meta_jigsaw_puzzle(self, type='RGB'):
        patch_current_num = 0
        complete_jigsaw = Image.new(type, (self.img_width, self.img_height))
        
        def jigsaw_puzzle(patch):       
            nonlocal patch_current_num
            paste_row, paste_col = patch_current_num // self.patch_cols, patch_current_num % self.patch_cols

            left = (self.patch_edge - self.patch_width) // 2
            top = (self.patch_edge - self.patch_height) // 2
            box_right = left + self.patch_width
            box_bottom = top + self.patch_height
            box = (left, top, box_right, box_bottom)
            cropped_patch = patch.crop(box)
            
            paste_x = paste_col * self.patch_width
            paste_y = paste_row * self.patch_height
            complete_jigsaw.paste(cropped_patch, (paste_x, paste_y))
            
            patch_current_num += 1
            if patch_current_num >= self.patch_nums:
                patch_current_num = 0
                return complete_jigsaw
            else:
                return None
        return jigsaw_puzzle