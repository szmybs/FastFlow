import argparse
import os

import torch
import time
import yaml
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import ndimage
from ignite.contrib import metrics
from torchvision import transforms
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor
import queue

import constants as const
import dataset
import fastflow
import utils
from plt_img import plt_density_map



def build_train_data_loader(args, config):
    if args.category in const.MVTEC_CATEGORIES:
        train_dataset = dataset.MVTecDataset(
            root=args.data,
            category=args.category,
            input_size=config["input_size"],
            is_train=True,
        )
    elif args.category in const.PLATE_CATEGORIES:
        train_dataset = dataset.PlateDataset(
            root=args.data,
            category=args.category,
            input_size=config["input_size"],
            is_train=True,
        )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)    
    return train_dataset, train_dataloader



def build_test_data_loader(args, config, batch_size=0):
    if args.category in const.MVTEC_CATEGORIES:
        test_dataset = dataset.MVTecDataset(
            root=args.data,
            category=args.category,
            input_size=config["input_size"],
            is_train=False,
        )
    elif args.category in const.PLATE_CATEGORIES:
        test_dataset = dataset.PlateDataset(
            root=args.data,
            category=args.category,
            input_size=config["input_size"],
            is_train=False,
        )
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size = const.BATCH_SIZE if batch_size <=0 else batch_size,
                                                shuffle = False,
                                                num_workers = 0,
                                                drop_last = False)
    return test_dataset, test_dataloader



def build_vis_data_loader(args, config, batch_size=0):
    if args.category in const.MVTEC_CATEGORIES:
        test_dataset = dataset.MVTecDataset(
            root=args.data,
            category=args.category,
            input_size=config["input_size"],
            is_train=False,
        )
    elif args.category in const.PLATE_CATEGORIES:
        test_dataset = dataset.PlateDataset(
            root=args.data,
            category=args.category,
            input_size=config["input_size"],
            is_train=False,
            is_visualize=True
        )
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size = const.BATCH_SIZE if batch_size <=0 else batch_size,
                                                shuffle = False,
                                                num_workers = 0,
                                                drop_last = False)
    return test_dataset, test_dataloader



def build_model(config):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model



def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )



def train_one_epoch(dataloader, model, optimizer, epoch, device):
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        # forward
        data = data.to(device)
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )



def eval_once(dataloader, model, device, **kwargs):
    model.eval()
    auroc_metric = metrics.ROC_AUC()
    for step, data in enumerate(dataloader):
        ori_img, data, targets = data
        # data, targets = data.cuda(), targets.cuda()
        data, targets = data.to(device), targets.to(device)
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs = outputs.flatten()
        targets = targets.flatten()
        auroc_metric.update((outputs, targets))
    auroc = auroc_metric.compute()
    print("AUROC: {}".format(auroc))
    
    if "threshold" in kwargs:
        threshold = kwargs["threshold"]
        total_targets, total_outputs = auroc_metric._targets.numpy(), auroc_metric._predictions.numpy()
        fpr, tpr, thresholds = roc_curve(total_targets, total_outputs)

        position = np.argwhere(thresholds <= threshold)
        position = np.min(position)
        print("True Positive Rate (TPR): {}".format(tpr[position]))
        print("False Positive Rate (FPR): {}".format(fpr[position]))
    
    # chosen_posit = np.argmax(tpr - fpr)
    # print("best threshold postion: {}".format(chosen_posit))
    # print("best threshold value: {}".format(thresholds[chosen_posit]))
    # print("best threshold value corresponding TPR: {}".format(tpr[chosen_posit]))
    # print("best threshold value corresponding FPR: {}".format(fpr[chosen_posit]))

    if "plt_roc_curve" in kwargs and kwargs["plt_roc_curve"]:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
    return
    


def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    optimizer = build_optimizer(model)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    _, train_dataloader = build_train_data_loader(args, config)
    _, test_dataloader = build_test_data_loader(args, config)
    model.to(device)

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch, device)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once(test_dataloader, model, device)
            
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )
    return



def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_dataloader = build_test_data_loader(args, config)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    eval_once(test_dataloader, model, device, plt_roc_curve=args.plt_roc_curve, threshold=args.threshold)
    return



def visualize(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    vis_dataset, vis_dataloader = build_vis_data_loader(args, config, batch_size=1)
    model.eval()
    
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    model.to(device)
        
    density_jigsaw_puzzle = vis_dataset.meta_jigsaw_puzzle('F')
    img_jigsaw_puzzle = vis_dataset.meta_jigsaw_puzzle('RGB')
    
    save_path = os.path.join(os.getcwd(), "results_imgs")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save_path = None
   
    threshold = args.threshold     
    for step, data in enumerate(vis_dataloader):
        ori_img, data, targets = data
        # data = data.cuda()
        data = data.to(device)
        with torch.no_grad():
            ret = model(data)
        
        if args.category in const.MVTEC_CATEGORIES:
            density = ret["anomaly_map"].cpu().detach().numpy()
            density = np.squeeze(density)
            img = ori_img.cpu().detach().squeeze()
            img = transforms.ToPILImage()(img)
            plt_density_map([img, density])
        
        elif args.category in const.PLATE_CATEGORIES:    
            density = ret["anomaly_map"].cpu().detach().numpy()
            density = np.squeeze(density)
            density = np.where(density<threshold, -1.0, density)            
            density = transforms.ToPILImage()(density)
            density = transforms.Resize(size=(vis_dataset.patch_edge, vis_dataset.patch_edge))(density)
            
            img = ori_img.cpu().detach().squeeze()
            img = transforms.ToPILImage()(img)

            jigsaw_density = density_jigsaw_puzzle(density)
            jigsaw_img = img_jigsaw_puzzle(img)
            
            if jigsaw_density is not None and jigsaw_img is not None:
                img = jigsaw_img
                density = transforms.ToTensor()(jigsaw_density).numpy().squeeze()
                density_mask = np.where(density<=-0.99, 0, 1)
                if save_path:
                    plt_density_map([img, density], mask=density_mask, save_path=os.path.join(save_path, str(step)+'.png'), only_save=args.only_save)
                else:
                    plt_density_map([img, density], mask=density_mask)
    
    _, test_dataloader = build_test_data_loader(args, config)
    eval_once(test_dataloader, model, device, plt_roc_curve=args.plt_roc_curve, threshold=args.threshold)    
    return



class PlateDefectLocalMachine(object):
    def __init__(self, model, input_size, threshold, device):
        self.model = model
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.470, 0.471, 0.471], [0.0559, 0.0543, 0.0516]),
            ]
        )
        self.threshold = threshold
        self.device = device
        return

    
    def processing(self, result_queue):
        self.segmentation()
        self.localization(result_queue)
        return
    
    
    def update_data(self, img_path):
        self.img_path = img_path
        print(f"新图片文件 updated: {self.img_path}")
        return


    def segmentation(self):
        with Image.open(self.img_path) as image:
            print(f"新图片文件 opened: {self.img_path}")
            # image = Image.open(self.img_path)            
            image = image.crop((0, 0, 1200, 672))
            width, height = image.size
            
            self.image_patches = []
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
                    self.image_patches.append(padded_patch)
                    
        self.img_width, self.img_height = width, height
        self.patch_width, self.patch_height = patch_width, patch_height
        self.patch_edge = patch_edge_max
        self.patch_cols, self.patch_rows = (width // patch_width),  (height // patch_height)
        self.patch_nums = self.patch_cols * self.patch_rows
        return
    
    
    def localization(self, result_queue):
        density_jigsaw_puzzle = self.meta_jigsaw_puzzle('F')
        img_jigsaw_puzzle = self.meta_jigsaw_puzzle('RGB')
        
        for data in self.image_patches:
            ori_image = transforms.ToTensor()(data)
            image = self.image_transform(data)
            if image.ndim == 3:
                image = image[None, ...]            
            
            image = image.to(self.device)
            with torch.no_grad():
                ret = self.model(image)
                    
            density = ret["anomaly_map"].cpu().detach().numpy()
            density = np.squeeze(density)
            density = np.where(density<self.threshold, -1.0, density)            
            density = transforms.ToPILImage()(density)
            density = transforms.Resize(size=(self.patch_edge, self.patch_edge))(density)
            
            img = ori_image.cpu().detach().squeeze()
            img = transforms.ToPILImage()(img)

            jigsaw_density = density_jigsaw_puzzle(density)
            jigsaw_img = img_jigsaw_puzzle(img)

            if jigsaw_density is not None and jigsaw_img is not None:
                img = jigsaw_img
                density = transforms.ToTensor()(jigsaw_density).numpy().squeeze()
                density_mask = np.where(density<=-0.99, 0, 1)
                
                result_queue.put(([img, density], density_mask, self.img_path))
                '''
                if self.save_path: 
                    plt_density_map(imgs=[img, density], \
                                    mask=density_mask, \
                                    save_path=os.path.join(self.save_path, f"densitymap_{os.path.basename(self.img_path)}"), \
                                    only_save=self.only_save)
                else:
                    plt_density_map([img, density], mask=density_mask)
                '''
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



def is_image(file_path):
    try:
        # 先检查文件大小，确保文件写入完成
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False
        
        img = Image.open(file_path)
        img.verify()  # 验证图像的完整性
        return True
    except Exception as e:
        print(f"验证失败: {file_path}, 错误信息: {e}")
        return False


def is_file_complete(file_path):
    try:
        time.sleep(0.5)
        initial_size = os.path.getsize(file_path)
        time.sleep(0.5)  # 稍等0.5秒钟，看看文件大小是否变化
        final_size = os.path.getsize(file_path)
        return initial_size == final_size
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
        return False
    except PermissionError:
        print(f"没有访问权限: {file_path}")
        return False
    except OSError as e:
        print(f"其他OS错误 [{e.errno}]: {file_path} - {e.strerror}")
        return False
    except Exception as e:
        print(f"未知错误: {file_path} - {str(e)}")
        return False
    

class ImageHandler(FileSystemEventHandler):
    def __init__(self, model, input_size, threshold, device, result_queue, executor):
        super().__init__()
        self.pdlm = PlateDefectLocalMachine(model, input_size, threshold, device)
        self.result_queue = result_queue
        self.executor = executor
        return
    
    def on_created(self, event):
        if event.is_directory:
            return
        file_path = event.src_path
        print(f"新图片文件 detected: {file_path}")

        while not is_file_complete(file_path):
            print(f"等待文件完成写入: {file_path}")
            time.sleep(1)

        self.pdlm.update_data(file_path)
        self.executor.submit(self.pdlm.processing, self.result_queue)
        '''
        if is_image(file_path):
            print(f"新图片文件 detected: {file_path}")
            self.pdlm.update_data(file_path)
            self.executor.submit(self.pdlm.processing, self.result_queue)
        else:
            print(file_path)
        '''
        return


def watchingdog(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    device = torch.device('cpu')
    model.to(device)

    save_path = os.path.join(os.getcwd(), "results_imgs")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(os.getcwd(), "results_txt")
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    result_queue = queue.Queue()  # 用于存放子线程结果的队列
    executor = ThreadPoolExecutor(max_workers=2) 
    
    WATCHED_FOLDER = os.path.join(args.data, args.category, "test", "watchdog")

    event_handler = ImageHandler(model, config["input_size"], args.threshold, device, result_queue, executor)
    observer = Observer()
    observer.schedule(event_handler, WATCHED_FOLDER, recursive=False)  # 不递归监控子文件夹
    observer.start()
    try:
        while True:
            if not result_queue.empty():
                imgs, density_mask, img_path = result_queue.get()
                print(f"开始处理新图片文件 started: {img_path}")
                labeled, num_features = ndimage.label(density_mask)
                region_sizes = []
                for i in range(1, num_features + 1):
                    size = np.sum(labeled == i)
                    region_sizes.append(size)
                # print(region_sizes)
                region_max_size = np.max(np.array(region_sizes))
                defect_flag = 1 if region_max_size >= args.post_threshold else 0  
                clf_txt = os.path.join(txt_path, f"densitymap_{os.path.basename(img_path)}.txt")
                with open(clf_txt, 'w') as f:  
                    f.write(str(defect_flag))  
                
                plt_density_map(imgs=imgs, \
                                mask=density_mask, \
                                save_path=os.path.join(save_path, f"densitymap_{os.path.basename(img_path)}"), \
                                only_save=args.only_save,
                                plt_duration=args.vis_duration)
            time.sleep(1)  # 持续运行，直到手动停止
    except KeyboardInterrupt:
        observer.stop()
    observer.join()



def parse_args():
    parser = argparse.ArgumentParser(description="Use FastFlow for Defect Localization")
    parser.add_argument(
        "--config", type=str, help="path to config file",
        default="configs/resnet18.yaml"
    )
    parser.add_argument("--data", type=str, help="path to data folder", default="data")
    parser.add_argument("--category", type=str, default="plate")
    parser.add_argument("--mode", default="watchdog")
    parser.add_argument(
        "--checkpoint", type=str, help="path to load checkpoint", default="checkpoints/500.pt"
    )
    parser.add_argument("--plt_roc_curve", type=bool, default=False)
    parser.add_argument("--threshold", type=float, default=-0.4)
    parser.add_argument("--only_save", type=bool, default=False)
    parser.add_argument("--post_threshold", type=int, default=2000)
    parser.add_argument("--vis_duration", type=int, default=20)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()    
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    elif args.mode == "vis":
        visualize(args)
    elif args.mode == "watchdog":
        watchingdog(args)
    os.system('pause')
