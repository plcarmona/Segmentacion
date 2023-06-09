import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import albumentations as A
from torchvision import transforms as T
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
import albumentations as A
import time
import torch.nn.functional as F
from tqdm.notebook import tqdm
from typing import Optional
from functools import partial
from typing import Optional
from functools import partial
from torch.nn.modules.loss import _Loss
from segmentation_models_pytorch.losses._functional import focal_loss_with_logits
from segmentation_models_pytorch.losses.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 7
IN_CHANNELS = [0,1,2]
ENCODER_NAME = 'resnet18'#'timm-regnetx_002'#trial.suggest_categorical('encoder',['resnet50','resnet18','timm-efficientnet-b1'])#'mobilenet_v2'  # 'mobilenet_v2', 'resnet50', 'resnet34'
ENCODER_WEIGHTS = 'imagenet'  # None, 'imagenet', 'ssl', 'swsl'
CHANS=len(IN_CHANNELS)

def getpaths(mode):
    if mode == "baches":
        IMAGE_PATH="../../shared_data/CUANTIFICACION/dataset/baches/images/"
        MASK_PATH="../../shared_data/CUANTIFICACION/dataset/baches/masks/"
    if mode == "corrugaciones":
        IMAGE_PATH="../../shared_data/CUANTIFICACION/dataset/corrugaciones//images/"
        MASK_PATH="../../shared_data/CUANTIFICACION/dataset/corrugaciones//masks/"
    if mode == "material":
        IMAGE_PATH="../../shared_data/CUANTIFICACION/dataset/material/images/"
        MASK_PATH="../../shared_data/CUANTIFICACION/dataset/material/masks/"
    if mode == "piedras":
        IMAGE_PATH="../../shared_data/CUANTIFICACION/dataset/piedras/images/"
        MASK_PATH="../../shared_data/CUANTIFICACION/dataset/piedras/masks/"
    if mode == "surcos":
        IMAGE_PATH="../../shared_data/CUANTIFICACION/dataset/surcos/images/"
        MASK_PATH="../../shared_data/CUANTIFICACION/dataset/surcos/masks/"
    return IMAGE_PATH, MASK_PATH
def create_df(mode):
    IMAGE_PATH, MASK_PATH = getpaths(mode)
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            name.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))
class OperationDataset(Dataset):
    def __init__(self, img_path, mask_path, X, IN_CHANNELS, transform=None):#, mean, std, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.IN_CHANNELS = IN_CHANNELS
        #self.mean = mean
        #self.std = std
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = np.load(self.img_path + self.X[idx] + '.npz')['arr_0'][:, :, self.IN_CHANNELS]
        mask = np.load(self.mask_path + self.X[idx] + '.npz')['arr_0']
        #mask = np.expand_dims(mask, axis = -1)
        #print(image.shape)
        #print(mask.shape)


        
        #if self.transform is not None:
            #mean=[image[0].mean(),image[1].mean(),image[2].mean()]
            #std=[image[0].std(),image[1].std(),image[2].std()]
            #t = T.Compose([T.ToTensor(),T.Normalize(mean,std)])
            #image = t(image).numpy()
            #image = np.moveaxis(image, 0, -1)
            #aug = self.transform(image=image, mask=mask)
            #image = aug['image']
            #mask = aug['mask']
        
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)  # (c, h, w)
        #image = torch.Tensor(image) / 255.0
        #mask = torch.from_numpy(mask).long()        
            
        return image, mask
    '''
def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy
'''

# Metric without background
def pixel_accuracy(output, mask):
    with torch.no_grad():
        ok_positions = torch.ge(mask, 1)
        output = torch.argmax(F.softmax(output[:, 1:, :, :], dim=1), dim=1) + 1
        
        output_select = torch.masked_select(output, ok_positions)
        mask_select = torch.masked_select(mask, ok_positions)
        
        correct = torch.eq(output_select, mask_select).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

'''
def mIoU(pred_mask, mask, smooth=1e-10, n_classes=N_CLASSES):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)
'''

# Metric without background
def mIoU(pred_mask, mask, smooth=1e-10, n_classes=N_CLASSES):
    with torch.no_grad():
        ok_positions = torch.ge(mask, 1)

        pred_mask = F.softmax(pred_mask[:, 1:, :, :], dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1) + 1

        pred_mask = torch.masked_select(pred_mask, ok_positions)
        mask = torch.masked_select(mask, ok_positions)

        pred_mask = pred_mask.contiguous()
        mask = mask.contiguous()

        iou_per_class = []
        for clas in range(1, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)
    
def RDS(mask):
    with torch.no_grad():
        Freq=mask[mask>=2].count_nonzero().item()/mask.count_nonzero().item()
        R=0
        for i in mask.unique():
            if i >= 2:
                ok = mask == i
                R += ok.count_nonzero().item()*i.item()/mask.count_nonzero().item()
        return R*Freq

def iter_mask(mode,trsh=0.1):
    IMG,MASK=getpaths(mode)
    files=os.listdir(MASK)
    r=[]
    for file in files:
        x=np.load(os.path.join(MASK,file))['arr_0']
        x=torch.from_numpy(x)
        x=RDS(x)[2]
        if x>trsh:
            r.append(file)
    return pd.DataFrame(r)

def get_randomimg(mode,n,seed=0):
    img,mask=getpaths(mode)
    x=os.listdir(img)
    #Select n random images from x
    #np.random.seed(seed)
    x=np.random.choice(x,n)
    x=pd.DataFrame(x,columns=["id"])
    x['id']=x['id'].str.replace(".npz","")
    return x

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    
    output= F.softmax(output[:, 1:, :, :], dim=1)
    output = torch.argmax(output, dim=1) + 1
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1).to(device)
    target = target.view(-1).to(device)
    
    output[target==ignore_index]=ignore_index
    
    metrica=torch.zeros(1).to(device)
    #total=mask.count_nonzero().item()
    for i in target.unique():
        ok = target == i

        intersection = (output[ok] == target[ok]).count_nonzero()
        
        error = ((output[ok]-target[ok]).abs().sum())
        union = output [ output == i ].count_nonzero() + target[ target == i ].count_nonzero() - intersection
        if error.item() == 0:
            error+=1
        if union.item() == 0:
            union+=1

        result=intersection/union/error
        #metric.append(result*target[ok].count_nonzero()/target.count_nonzero())
        metrica+=result.to(device)#*target[ok].to(device).count_nonzero()/target.to(device).count_nonzero()

    return 1-metrica

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, modelname, patch=False):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []; val_acc = []
    train_iou = []; train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1 ; not_improve=0
    model_name = modelname

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        #training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            #training phase
            image_tiles, mask_tiles = data
            
            image = image_tiles.to(device); mask = mask_tiles.to(device);
            #forward
            output = model(image)
            
            loss = criterion(output, mask)
            #evaluation metrics
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            #backward
            loss.backward()
            optimizer.step() #update weight          
            optimizer.zero_grad() #reset gradient
            
            #step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step() 
            
            running_loss += loss.item()
            
        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            #validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    #reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    image = image_tiles.to(device); mask = mask_tiles.to(device);
                    output = model(image)
                    #evaluation metrics
                    val_iou_score +=  mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    #loss
                    try:
                        loss =criterion(output,mask)#intersectionAndUnionGPU(output,mask,7) #criterion(output, mask)                                  
                    except:
                        loss =criterion(output.long(),mask.long())
                    test_loss += loss.item()
            
            #calculatio mean for each batch
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(val_loader))


            if min_loss > (test_loss/len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss/len(val_loader))))
                min_loss = (test_loss/len(val_loader))
                decrease += 1
                print('saving model...')
                #torch.save(model, model_name + 'mIoU-{:.3f}.pt'.format(val_iou_score/len(val_loader)))
                    

            if (test_loss/len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss/len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 7:
                    print('Loss not decrease for 7 times, Stop Training')
                    break
            
            #iou
            val_iou.append(val_iou_score/len(val_loader))
            train_iou.append(iou_score/len(train_loader))
            train_acc.append(accuracy/len(train_loader))
            val_acc.append(test_accuracy/ len(val_loader))
            print("Epoch:{}/{}..".format(e+1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss/len(val_loader)),
                  "Train mIoU:{:.3f}..".format(iou_score/len(train_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou_score/len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy/len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy/len(val_loader)),
                  "Time: {:.2f}m".format((time.time()-since)/60))
            
    if val_iou_score/len(val_loader) > 0.5:
        torch.save(model, model_name + 'mIoU-{:.3f}.pt'.format(val_iou_score/len(val_loader)))    
    
    history = {'train_loss' : train_losses, 'val_loss': test_losses,
               'train_miou' :train_iou, 'val_miou':val_iou,
               'train_acc' :train_acc, 'val_acc':val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    return history

t_train = A.Compose([A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(), A.Transpose(), 
                     A.GridDistortion(p=0.2)])#, A.RandomBrightnessContrast((0,0.5),(0,0.5)), A.GaussNoise()])

t_val = A.Compose([A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90()])


def getModel(model, ENCODER_NAME=ENCODER_NAME, ENCODER_WEIGHTS=ENCODER_WEIGHTS, N_CLASSES=N_CLASSES):
    if model == 'unet':
        return smp.Unet(encoder_name = ENCODER_NAME, encoder_weights = ENCODER_WEIGHTS, in_channels = CHANS, classes = N_CLASSES, activation = None)
    elif model == 'fpn':
        return smp.FPN(encoder_name = ENCODER_NAME, encoder_weights = ENCODER_WEIGHTS, in_channels = CHANS, classes = N_CLASSES, activation = None)
    elif model == 'linknet':
        return smp.Linknet(encoder_name = ENCODER_NAME, encoder_weights = ENCODER_WEIGHTS, in_channels = CHANS, classes = N_CLASSES, activation = None)
    elif model == 'unet++':
        return smp.UnetPlusPlus(encoder_name = ENCODER_NAME, encoder_weights = None, in_channels = CHANS, classes = N_CLASSES, activation = None)
    else:
        raise ValueError('model name is not correct')

class FocalLoss(_Loss):
    def __init__(
        self,
        mode: str,
        alpha: Optional[float] = None,
        gamma: Optional[float] = 2.0,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        """Compute Focal loss

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            alpha: Prior probability of having positive value in target.
            gamma: Power factor for dampening weight (focal strength).
            ignore_index: If not None, targets may contain values to be ignored.
                Target values equal to ignore_index will be ignored from loss computation.
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode you
                should use `reduction="sum"`.

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:

            num_classes = y_pred.size(1)
            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                cls_y_true = (y_true == cls).long()
                cls_y_pred = y_pred[:, cls, ...]

                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]
                if cls>=2:
                    aux=cls_y_pred!=cls_y_true
                    cls_y_pred=cls_y_pred[aux]
                    cls_y_true=cls_y_true[aux]
                    loss += self.focal_loss_fn(cls_y_pred, cls_y_true)*(cls_y_pred-cls_y_true).abs().mean()
                else:
                    loss += self.focal_loss_fn(cls_y_pred, cls_y_true)*0.5

        return loss

def getCriterion(CRITERION):
    if CRITERION == 'CEL':
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    elif CRITERION == 'DL':
        criterion = smp.losses.DiceLoss(mode='multiclass', ignore_index=0)
    #elif CRITERION == 'CEL+DL':  #definir bien
    #    criterion = smp.losses.JacardLoss('multiclass')
    #elif CRITERION == 'JC':
    #    criterion = smp.losses.JaccardLoss(mode='multiclass')  #RuntimeError: one_hot is only applicable to index tensor.
    elif CRITERION == 'LV':
        criterion = smp.losses.LovaszLoss(mode='multiclass', ignore_index=0)
    elif CRITERION == 'Focal':
        criterion = smp.losses.FocalLoss(mode='multiclass', ignore_index=0)
    elif CRITERION == 'Tversky':
        criterion = smp.losses.TverskyLoss(mode='multiclass',alpha=0.5,beta=0.5, ignore_index=0)
    elif CRITERION == 'Custom':
        criterion = FocalLoss(mode='multiclass',ignore_index=0)
    return criterion