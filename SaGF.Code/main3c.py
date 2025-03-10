import torch
from byol_pytorch import BYOL
from torchvision import models
import argparse
from models import *
from models.vit import ViT
from models.gen_vit import gen_vit
from models.inv_vit import inv_vit
from utils import progress_bar
from sklearn.metrics import roc_auc_score
import os
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
from torch.utils.data.dataset import ConcatDataset
from utilsfile.mask_utils import create_subgraph_mask2coords, create_rectangle_mask, create_rectangle_mask2coords, create_bond_mask2coords
from utilsfile.public_utils import setup_device
from skimage.feature import corner_harris
from skimage.color import rgb2gray
from skimage.restoration import denoise_tv_chambolle
from skimage.feature import corner_peaks
from utilsfile.harris import CornerDetection
import time
from warmup_scheduler import GradualWarmupScheduler
import copy


# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='add image augumentations')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', type=int, default='16')  #64
parser.add_argument('--weight_decay', default=1e-6, type=float, help='SGD weight decay')
parser.add_argument('--data_address', default='../data/Pretraining/', type=str)
parser.add_argument('--n_epochs', type=int, default='100')
parser.add_argument('--n_epochs_tafter', type=int, default='200')
parser.add_argument('--dim', type=int, default='128')
parser.add_argument('--imagesize', type=int, default='512')  #288
parser.add_argument('--patch', default='32', type=int)  #24
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--tau', type=float, default=0.99)
parser.add_argument('--downsample', type=float, default=0.5)
parser.add_argument('--cos', default='True', action='store_true', help='Train with cosine annealing scheduling')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
size = int(args.imagesize)

vit = ViT(
        image_size=int(args.imagesize),
        patch_size=args.patch,
        kernel_size=5,
        downsample=args.downsample,
        batch_size=args.bs,
        num_classes=args.num_classes,
        dim=args.dim,
        depth=9,
        heads=8,
        mlp_dim=args.dim,
        patch_stride=2,
        patch_pading=0,
        in_chans=3,
        dropout=0.2,  # 0.1
        emb_dropout=0.2,  # 0.1
        expansion_factor=2
    ).to(device)

gen_vit = gen_vit(
    image_size=int(args.imagesize),
    patch_size=args.patch//2,
    kernel_size=5,
    downsample=args.downsample, ####
    batch_size=args.bs,
    num_classes=args.num_classes,
    dim=args.dim,
    depth=7,
    heads=8,
    mlp_dim=args.dim,
    patch_stride=2,
    patch_pading=0,
    in_chans=3,####
    dropout=0.2,   # 0.1
    emb_dropout=0.2,   # 0.1
    expansion_factor=1
    )

# inv_vit = inv_vit(
#     image_size=int(args.imagesize),
#     patch_size=args.patch//2,
#     kernel_size=5,
#     downsample=args.downsample, ####
#     batch_size=args.bs,
#     num_classes=args.num_classes,
#     dim=args.dim,
#     depth=7,
#     heads=8,
#     mlp_dim=args.dim,
#     patch_stride=2,
#     patch_pading=0,
#     in_chans=3,####
#     dropout=0.2,   # 0.1
#     emb_dropout=0.2,   # 0.1
#     expansion_factor=1
#     )

learner = BYOL(
    vit,
    gen_vit,
    # inv_vit,
    image_size=args.imagesize,
    hidden_layer='to_cls_token',
    projection_size=args.dim,
    projection_hidden_size=4096
)
# learner.to(device)
opt = torch.optim.Adam(learner.parameters(), lr=args.lr)  #, weight_decay=args.weight_decay, betas=(0.5, 0.999)

scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(args.n_epochs / 2) + 1)
scheduler = GradualWarmupScheduler(opt, multiplier=2, total_epoch=int(args.n_epochs / 2) + 1,
                                        after_scheduler=scheduler_cosine)

# # Find total parameters and trainable parameters
# total_params = sum(p.numel() for p in learner.parameters())
# print(f'{total_params:,} total parameters.')
# total_trainable_params = sum(
#     p.numel() for p in learner.parameters() if p.requires_grad)
# print(f'{total_trainable_params:,} training parameters.')


transform_test = transforms.Compose([
    transforms.Resize((int(size), int(size))),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

##############kaishi

testset = torchvision.datasets.ImageFolder(root='../data/Finetuning3c4TSNE/test_scoffold/', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=int(args.bs), shuffle=True, num_workers=0)
transf = transforms.ToTensor()


train_after_dataset = torchvision.datasets.ImageFolder(root='../data/Finetuning3c4TSNE/train_scoffold/', transform=transform_test)
# for i in range(1):
#     temp = torchvision.datasets.ImageFolder(root='../data/hiv/train_scoffold/', transform=transform)
#     train_after_dataset = ConcatDataset([train_after_dataset, temp])

trainafterloader = torch.utils.data.DataLoader(train_after_dataset, batch_size=int(args.bs), shuffle=True, num_workers=0)
if args.cos:
    from warmup_scheduler import GradualWarmupScheduler

from utils import aug_rand, rot_rand
class ImagePairDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # self.transformaug = transformaug
        self.image_pairs = self.get_image_pairs()

    def get_image_pairs(self):
        # 假设您的图像对是按照一定的命名规则组织的，例如 "image1.jpg" 和 "image2.jpg" 是一对
        image_pairs = []
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.jpg'):  # 或者其他图像格式
                # base_name = filename.split('.')[0]
                pair1_path = os.path.join(self.image_dir, filename)
                pair2_path = os.path.join(self.image_dir, filename)  # 假设命名规则是这样的, base_name + '_pair.png'
                if os.path.exists(pair2_path):  # 确保第二个图像存在
                    image_pairs.append((pair1_path, pair2_path))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image1_path, image2_path = self.image_pairs[idx]
        image1 = Image.open(image1_path).convert('RGB')
        image1 = self.transform(image1)
        image2 = image1
        # img1_augment = aug_rand(image1)
        # img2_augment = aug_rand(image2)

        return image1, image2


# 定义文件datasetm Saving ..
# Tue Nov 14 00:27:36 2023 Epoch 35, test loss: 1.56787, acc: 88.67925, roc_auc_avg: 0.89025
# best acc=88
training_dir = args.data_address  # 训练集地址
transform_pre = transforms.Compose([
    transforms.Resize((int(size), int(size))),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

pair_dataset = ImagePairDataset(image_dir=training_dir, transform=transform_pre)

# 图像无监督预训练
train_dataloader = DataLoader(pair_dataset, shuffle=True, batch_size=int(args.bs), pin_memory=True, num_workers=0)

weights = torch.tensor([1.0, 0.33, 1.0])
criterion_ce = nn.CrossEntropyLoss(weight=weights).to(device)
# criterion_ce = nn.CrossEntropyLoss().to(device)
L1_loss = nn.L1Loss()

def train_after(epoch, vit4trainafter, net4trainafter, opt4net, opt4gen, scheduler2, scheduler3):
    print('\nEpoch: %d' % epoch)
    vit4trainafter.train()
    net4trainafter.train()
    # learner.gen_vit.required_grad = False

    # for param in learner.parameters():  #frezzee the conv layers
    #     if isinstance(param, nn.Conv2d):
    #         param.requires_grad = False

    train_loss = 0
    correct = 0
    total = 0
    accumulation = 2

    for batch_idx, (inputs, targets) in enumerate(trainafterloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # img = inputs.float()
        # images_denorm = img * torch.tensor(std).view(-1, 1, 1).to(device) + torch.tensor(mean).view(-1, 1, 1).to(device)
        # images_denorm = images_denorm.type_as(img)
        # augmented_img1 = unloader(images_denorm[0, :, :, :]).convert('RGB')
        # plt.imshow(augmented_img1, cmap="brg")
        # plt.show()

        x_pre, x, noise, x_sparsenoise_pool, gen_adj, sparsity_v = vit4trainafter(inputs)

        loss_sparse1 = L1_loss(noise, torch.tensor([0.0]).expand_as(noise).to(device))

        # nuclear_norms = [torch.linalg.norm(x[i, :, :], ord='nuc') for i in range(x.size(0))]
        # # 将核范数列表转换为张量
        # nuclear_norms_tensor = torch.tensor(nuclear_norms)
        # # 计算核范数的平均值
        # average_nuclear_norm = torch.mean(nuclear_norms_tensor)

        _, outputs = net4trainafter(inputs, x_pre, x, x_sparsenoise_pool, gen_adj, sparsity_v)

        # img1 = torch.empty(inputs.size()).cuda()
        # img2 = torch.empty(inputs.size()).cuda()
        # for i in range(inputs.size(0)):
        #     img1[i, :, :, :], img2[i, :, :, :] = augment_oneimage(inputs[i, :, :])
        # loss0 = learner(img1, img2)

        loss = criterion_ce(outputs, targets)  # + 0.1 * loss_sparse1 + 0.01 * average_nuclear_norm

        loss.backward()
        if ((batch_idx + 1) % accumulation) == 0:
            opt4net.step()
            opt4net.zero_grad()
            opt4gen.step()
            opt4gen.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    content = time.ctime() + 'opt4net' + f'Epoch {epoch}, lr: {opt4net.param_groups[0]["lr"]:.5f}, loss: {train_loss}'
    print(content)

    scheduler2.step()
    scheduler3.step()


from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
def test3c(epoch, vit4trainafter, net4trainafter):
    vit4trainafter.eval()
    net4trainafter.eval()
    test_loss = 0
    correct = 0
    total = 0

    # 用于收集所有样本的预测概率和真实标签
    all_targets = []
    all_probs = []  # 存储所有类别的概率矩阵
    features = []
    labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            x_pre, x, noise, x_sparsenoise_pool, gen_adj, sparsity_v = vit4trainafter(inputs)
            pred, outputs = net4trainafter(inputs, x_pre, x, x_sparsenoise_pool, gen_adj, sparsity_v)

            # 计算损失
            loss = criterion_ce(outputs, targets)
            test_loss += loss.item()

            # 获取预测类别
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 关键修改1：使用softmax获取三分类概率
            probs = torch.softmax(outputs, dim=1)  # 形状: (batch_size, 3)

            features.append(pred.cpu().numpy())
            labels.extend(targets.cpu().numpy())
            # 收集真实标签和预测概率
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())


        features = np.concatenate(features)
        # 使用t-SNE进行降维
        tsne = TSNE(n_components=2, verbose=1, perplexity=80, n_iter=600)
        tsne_results = tsne.fit_transform(features)
        # 可视化结果
        plt.figure(figsize=(10, 8))
        colors = ['red' if label == 0 else 'blue' if label == 1 else 'green' for label in labels]
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, s=8, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.xlabel('t-SNE feature 1', fontsize=20)
        plt.ylabel('t-SNE feature 2', fontsize=20)
        # plt.title('t-SNE visualization of MNIST features')
        plt.show()

    # 关键修改2：三分类ROC AUC计算
    try:
        # 转换为numpy数组
        all_targets_np = np.array(all_targets)
        all_probs_np = np.array(all_probs)

        # 计算多分类ROC AUC
        roc_auc = roc_auc_score(
            all_targets_np,
            all_probs_np,
            multi_class='ovr',  # One-vs-Rest策略
            average='macro'  # 可选'micro'/'weighted'
        )
    except ValueError as e:
        print(f"ROC AUC计算失败: {str(e)}")
        roc_auc = 0.0

    # 计算多分类混淆矩阵
    cm = confusion_matrix(all_targets, np.argmax(all_probs_np, axis=1))

    # 计算每个类别的敏感性和特异性
    sensitivity = {}
    specificity = {}
    for idx, class_name in enumerate(['Class0', 'Class1', 'Class2']):
        TP = cm[idx, idx]
        FN = np.sum(cm[idx, :]) - TP
        FP = np.sum(cm[:, idx]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        sensitivity[class_name] = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity[class_name] = TN / (TN + FP) if (TN + FP) > 0 else 0

    # 计算宏平均敏感性和特异性
    macro_sensitivity = np.mean(list(sensitivity.values()))
    macro_specificity = np.mean(list(specificity.values()))

    # 计算准确率
    acc = 100. * correct / total

    # 输出结果
    content = (f'Epoch {epoch}, test loss: {test_loss:.5f}, acc: {acc:.5f}, '
               f'roc_auc: {roc_auc:.5f}, '
               f'macro_sensitivity: {macro_sensitivity:.5f}, '
               f'macro_specificity: {macro_specificity:.5f}')
    print(content)

    return test_loss, acc, roc_auc, macro_sensitivity, macro_specificity



############ train using label $###################################
###################################################################
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def trainandevl():
    best_roc = 0
    best_acc = 0
    import copy
    vit4trainafter = copy.deepcopy(learner.gen_vit)
    # set_requires_grad(vit4trainafter, False)

    net4trainafter = copy.deepcopy(learner.online_encoder.net)
    opt4net = torch.optim.Adam(net4trainafter.parameters(),
                               lr=1.0*args.lr)  # , weight_decay=args.weight_decay, betas=(0.5, 0.999)
    opt4gen = torch.optim.Adam(vit4trainafter.parameters(),
                               lr=1.0*args.lr)  # , weight_decay=args.weight_decay, betas=(0.5, 0.999)

    scheduler_cosine2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt4net, int(args.n_epochs_tafter / 2) + 1)

    scheduler_cosine3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt4gen, int(args.n_epochs_tafter / 2) + 1)
    scheduler2 = GradualWarmupScheduler(opt4net, multiplier=2, total_epoch=int(args.n_epochs_tafter / 2) + 1,
                                        after_scheduler=scheduler_cosine2)
    scheduler3 = GradualWarmupScheduler(opt4gen, multiplier=2, total_epoch=int(args.n_epochs_tafter / 2) + 1,
                                        after_scheduler=scheduler_cosine3)

    for epoch in range(0, args.n_epochs_tafter):
        train_after(epoch, vit4trainafter, net4trainafter, opt4net, opt4gen, scheduler2, scheduler3)
        if epoch % 1 == 0:
            test_loss, acc, roc_auc, sensitivity, specificity = test3c(epoch, vit4trainafter, net4trainafter)
            # if roc_auc > best_roc:
            #     best_roc = roc_auc
            #     best_acc = acc
            #     best_model = copy.deepcopy(learner)
            if acc > best_acc:
                best_acc = acc
                # best_model = copy.deepcopy(learner)
                torch.save(learner.state_dict(), './best-net.pth')

    del net4trainafter, vit4trainafter
    return best_acc, best_roc  #, best_model

min_loss = 1e5
accumulation = 4   #4
best_acc_global = 0  # best test accuracy
best_roc_global = 0  # best test roc
biaozhi = 0
total_batch = 0
rank_k = int((args.imagesize / args.patch * args.downsample) ** 2) // 2
duration4rank = 0

if args.n_epochs == 0:
    # learner.load_state_dict(torch.load('improved-net.pth'), strict=False)  # -224-0.8056
    trainandevl()
learner.load_state_dict(torch.load('improved-net.pth'), strict=False)  # -224-0.8056
for interation in range(args.n_epochs):
    print('interation=%d'%interation)
    torch.cuda.synchronize()
    start = time.time()
    train_loss = 0
    learner.train()

    for i, data in enumerate(train_dataloader, 0):

        img1, img2 = data
        img1, img2 = img1.to(device), img2.to(device)  # 数据移至GPU

        loss = learner(img1, img2, None, rank_k)

        train_loss += loss.item()
        loss.backward()
        if ((i + 1) % accumulation) == 0:
            opt.step()
            opt.zero_grad()
        total_batch = i

    learner.update_moving_average()  # update moving average of target encoder
    train_loss = train_loss / (total_batch + 1)

    content = time.ctime() + ' ' + f'Epoch {interation}, Train loss: {train_loss:.4f}, lr: {opt.param_groups[0]["lr"]:.5f}'
    print(content)

    if train_loss <= min_loss:
        min_loss = train_loss
        biaozhi = 1
        duration4rank = 0
        torch.save(learner.state_dict(), './improved-net.pth')

    if train_loss > min_loss:
        duration4rank = duration4rank + 1
    if duration4rank >= 5 and rank_k >= 16:
        rank_k = rank_k - 1
        duration4rank = 0
    print('rank_k=%d' % rank_k)

    if interation >= 10 and (biaozhi == 1):  #interation%10 == 0 or
        learner.load_state_dict(torch.load('improved-net.pth'), strict=False)  # -224-0.8056
        test_loss, acc, roc_auc, sensitivity, specificity = test3c(0, learner.gen_vit, learner.online_encoder.net)

        biaozhi = 0
        best_acc, best_roc = trainandevl()

        # if best_roc > best_roc_global:
        #     best_roc_global = best_roc
        #     best_acc_global = best_acc
        #     # torch.save(learner.state_dict(), './improved-net.pth')

    scheduler.step()

    torch.cuda.synchronize()
    end = time.time()
    # print("cost time", end-start, "s")
