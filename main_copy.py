import argparse
import os

import matplotlib.colors
import matplotlib.gridspec
import torch
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from ignite.contrib import metrics
from torchvision import transforms

import constants as const
import dataset
import fastflow
import utils



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
    
    def plt_density_map(imgs, mask=None, save_path=None, only_save=False):        
        fig = plt.figure(figsize=[23, 4])
        gs = matplotlib.gridspec.GridSpec(1, 22)
        
        # ax0
        ax0 = fig.add_subplot(gs[0, 0:7])
        im0 = ax0.imshow(imgs[0])
        ax0.axis("off")
        
        norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=0.0)
        # ax1
        ax1 = fig.add_subplot(gs[0, 7:14])
        im1 = ax1.imshow(imgs[0])
        
        alpha = 0.25
        if mask is not None:
            alpha = mask * alpha
        im1 = ax1.imshow(imgs[1], cmap='jet', norm=norm, alpha=alpha)
        ax1.axis("off")
        
        # ax2
        norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=0.0)
        ax2 = fig.add_subplot(gs[0, 14:21])
        im2 = ax2.imshow(imgs[1], cmap='jet', norm=norm)
        ax2.axis("off")
        
        # coloar-bar
        cbar_ax = fig.add_subplot(gs[0, -1])  
        cbar = fig.colorbar(im2, cax=cbar_ax)
                
        if save_path is None:
            plt.show()
        else:
            if not only_save:
                plt.savefig(save_path, pad_inches=0.2, bbox_inches='tight')
                plt.show()
            else:
                plt.savefig(save_path, pad_inches=0.2, bbox_inches='tight')
                plt.close()                
        return
    
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
        


def parse_args():
    parser = argparse.ArgumentParser(description="Use FastFlow for Defect Localization")
    parser.add_argument(
        "--config", type=str, help="path to config file",
        default="configs/resnet18.yaml"
    )
    parser.add_argument("--data", type=str, help="path to data folder", default="data")
    parser.add_argument("--category", type=str, default="plate")
    parser.add_argument("--mode", default="vis")
    parser.add_argument(
        "--checkpoint", type=str, help="path to load checkpoint", default="checkpoints/500.pt"
    )
    parser.add_argument("--plt_roc_curve", type=bool, default=False)
    parser.add_argument("--threshold", type=float, default=-0.4)
    parser.add_argument("--only_save", type=bool, default=False)
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
    os.system('pause')
