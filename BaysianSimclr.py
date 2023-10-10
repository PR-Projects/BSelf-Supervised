import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer, required

import torchvision
from torchvision import datasets,transforms
import torchvision.models as models

import os
import random
import math
import json
import argparse
import copy
import time
import datetime
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path 
from sklearn.model_selection import ShuffleSplit
import warnings 
warnings.filterwarnings('ignore')

import wandb

def plotCurves(stats,results_dir=None):
    
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1,1,1)
    plt.plot(stats['train'], label='train_loss')
    plt.plot(stats['val'], label='valid_loss')
        
    textsize = 12
    marker=5
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('NLL')

    lgd = plt.legend(['train', 'validation'], markerscale=marker, 
                 prop={'size': textsize, 'weight': 'normal'})    
    
    plt.savefig(results_dir , bbox_extra_artists=(lgd,), bbox_inches='tight')
    

#### Data augmentation 
def get_transform(in_size,ds,s=1,aug=True):
    
    if ds == 'cifar10':
        
        mean_ds = [0.491, 0.482, 0.447]
        std_ds = [0.247, 0.243, 0.261] 
        
    elif ds == 'cifar100':
        
        mean_ds = [0.507, 0.486, 0.440]
        std_ds = [0.267, 0.256, 0.276] 
        
    elif ds == 'imagenet10' or ds == 'tinyimagenet':
        mean_ds = [0.485, 0.456, 0.406]
        std_ds = [0.229, 0.224, 0.225]
        
    elif ds == 'stl10':
        mean_ds = [0.485, 0.456, 0.406]
        std_ds = [0.229, 0.224, 0.225]
        
    
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    
    if aug:
        transform=transforms.Compose([transforms.RandomResizedCrop(size=in_size),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomApply([color_jitter], p=0.8),
                                  transforms.RandomGrayscale(p=0.2),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean_ds, std_ds)])
    else:
        transform=transforms.Compose([transforms.Resize(size=(in_size,in_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean_ds, std_ds)])     
    return(transform)



class pair_aug(object):
    def __init__(self,transform):
        self.transform = transform
        
    def __call__(self,img):
        if self.transform:
            img1= self.transform(img)
            img2= self.transform(img)
        return(img1,img2)

#### Dataloaders
def get_train_val_loader(dataset,val_size,batch_size,num_workers,seed):
    
    if val_size:
    
        split = ShuffleSplit(n_splits=1,test_size=val_size,random_state=seed)
    
        for train_idx, val_idx in split.split(range(len(dataset))):
        
            train_index= train_idx
            val_index = val_idx
        
        train_set = Subset(dataset,train_index)
        val_set = Subset(dataset,val_index)
    
        train_loader = DataLoader(train_set, batch_size=batch_size,num_workers=num_workers, drop_last=True, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size,num_workers=num_workers, drop_last=True, shuffle=True)
        return(train_loader,val_loader)
    else:
        
        dataloader = DataLoader(dataset, batch_size=batch_size,num_workers=num_workers, drop_last=True, shuffle=True)
        
        return(dataloader,_)

#### Building the model
class MLP(nn.Module): 
    def __init__(self,in_dim,mlp_hid_size,proj_size):
        super(MLP,self).__init__()
        self.head = nn.Sequential(nn.Linear(in_dim,mlp_hid_size),
                                 nn.BatchNorm1d(mlp_hid_size),
                                 nn.ReLU(),
                                 nn.Linear(mlp_hid_size,proj_size))
        
    def forward(self,x):
        x= self.head(x)
        return(x)
    
class network(nn.Module):
    
    def __init__(self,backbone,mid_dim,out_dim):  
        super(network,self).__init__()
        
        # we get representations from avg_pooling layer
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.projection = MLP(in_dim= backbone.fc.in_features,mlp_hid_size=mid_dim,proj_size=out_dim) 
  
    def forward(self,x):
        
        embedding = self.encoder(x)
        embedding = embedding.view(embedding.size()[0],-1)
        project = self.projection(embedding)        
        return(project)
    
##### NT_Xent loss
class NT_Xent(nn.Module):
    def __init__(self, temperature, device):
        super(NT_Xent, self).__init__()
        
        self.temperature = temperature
        self.device = device

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        
        self.batch_size= z_i.size()[0]
        
        self.mask = torch.ones((self.batch_size * 2, self.batch_size * 2), dtype=bool)
        self.mask = self.mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            self.mask[i, self.batch_size + i] = 0
            self.mask[self.batch_size + i, i] = 0
            
        z_i= F.normalize(z_i, dim=1)
        z_j= F.normalize(z_j, dim=1)
        
        p1 = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature
        
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.batch_size * 2, 1)
        negative_samples = sim[self.mask].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return(loss)

    
##### SGHMC optimizer

DEFAULT_DAMPENING = 0.0
class SGHM(Optimizer): 

    def __init__(self,params,
                 lr=required,
                 momentum=0.99, 
                 dampening=0.,
                 weight_decay=0.,
                 N_train =0.,
                 temp= 1.0,
                 addnoise=True,
                 epoch_noise=True):
            
        if weight_decay <0.0:
            
            raise ValueError("Invalid weight_decay value:{}".format(weight_decay))
            
        if lr is not required and lr < 0.0:
            
            raise ValueError("Invalid leraning rate:{}".format(lr))
            
        if momentum < 0.0:
            
            raise ValueError("Invalid momentum value: {}".format(momentum))
            
        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay = weight_decay,
                        N_train = N_train,
                        temp = temp,
                        addnoise=addnoise,
                        epoch_noise=epoch_noise)
        
        super(SGHM, self).__init__(params, defaults)
        
    def step(self,closure=None):
            
        """a single optimization step"""
            
        loss = None
            
        if closure is not None:
                
            with torch.enable_grad():
                loss = closure()
            
        for group in self.param_groups:
                
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            N_train = group['N_train']
            temp = group['temp']
            epoch_noise= group['epoch_noise']
            
            for p in group['params']:
                    
                if p.grad is None:
                        
                    continue
                        
                d_p = p.grad
                    
                if weight_decay!=0:

                    d_p.add_(p, alpha= weight_decay)

                d_p.mul_(-(1/2)* group['lr'])
                    
                if momentum != 0:
                    param_state = self.state[p]
                    
                    if 'momentum_buffer' not in param_state:
                        
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach() 
                        
                    else:
                        
                        buf = param_state['momentum_buffer']

                        buf.mul_(momentum*(group['lr']/N_train)**0.5).add_(d_p, alpha=1 - dampening)

                    d_p = buf

                if group['addnoise'] and group['epoch_noise']:
                    
                    noise = torch.randn_like(p.data).mul_((temp * group['lr']*(1-momentum)/N_train)**0.5)
                    
                    p.data.add_(d_p +noise)

                    if torch.isnan(p.data).any(): exit('Nan param')
                    
                    if torch.isinf(p.data).any(): exit('inf param')
                            
                else:

                    p.data.add_(d_p)        
        return(loss)
    
#### Cyclical step learning 

min_v = 0 

def update_lr(lr0,batch_idx,cycle_batch_length,n_sam_per_cycle,optimizer):
            
    is_end_of_cycle = False
        
    prop = batch_idx % cycle_batch_length
    
    pfriction = prop/cycle_batch_length
    
    lr = lr0 * (min_v +(1.0-min_v)*0.5*(np.cos(np.pi * pfriction)+1.0))
            
    if prop >= cycle_batch_length-n_sam_per_cycle:

        is_end_of_cycle = True

    for param_group in optimizer.param_groups:

        param_group['lr'] = lr


def get_lr(optimizer):
    
    for group in optimizer.param_groups:
        
        return(group['lr'])

def update_lin(lr,optimizer):
    
    for param_group in optimizer.param_groups:

        param_group['lr'] = lr    
    

#### Main training

parser = argparse.ArgumentParser(description='SimCLR pretraining')

parser.add_argument('--seed',type=int,default=42,
                    help='The seed for experiments!')

parser.add_argument('--exp',type=int,default=1009,
                        help='ID of this expriment!')

parser.add_argument('--ds',type=str,default='stl10',choices=('cifar10','cifar100','imagenet10','stl10','tinyimagenet'),
                        help='Dataset for pretraining!')

parser.add_argument('--num_epochs',type=int, default=200,
                       help='The number of epoch for pretraining')

parser.add_argument('--model_type',type=str, default='', choices=('','b','c','cb'),
                       help='If we want to train simclr, bsimclr, csimclr, cbsimclr!')

# model
parser.add_argument('--model_depth',type=str,choices=('res18','res50','res100'),default='res18',
                        help='Model to use as feature extractor!')

parser.add_argument('--mlp_hid_size',type=int, choices=(128,512,2048), default=512,
                       help='The mid size in MLP')

parser.add_argument('--proj_size',type=int, choices=(128,256,2048), default=128,
                       help='The size of projection map')

parser.add_argument('--prtr',type=bool, default=True,
                       help='If we want to use pretrained resnet on ImageNet or not!')

parser.add_argument('--temp_loss',type = float, default=0.1, 
                        help = 'Temprature for simCLR loss!')

# optimizer 
parser.add_argument('--optimizer',type=str, default='sgd', choices=('adam','sgd','sghm'),
                       help='The optimizer to use')

parser.add_argument('--base_target_ema',type=float, choices=(0,1), default=0.996,
                       help='The size of projection map')

parser.add_argument('--temp',type = float, default=0.1, 
                        help = 'Temprature for cold posterior in sghm opt!')


# data
parser.add_argument('--in_size',type=int, default=224,
                       help='The input size of images')

parser.add_argument('--s',type=float,default=1.0,choices=(0.5,1.0),
                        help='The strength of color distortion! (s=0.5 for cifar10 and 100)')

parser.add_argument('--aug',type=bool,default=True,
                        help='If we want to have data augmentation or not!')

# dataloader
parser.add_argument('--batch_size',type=int, default=256,
                       help='The number of batch size for training')

parser.add_argument('--val_size',type=float, default=0.05,
                       help='The validation size')

parser.add_argument('--num_workers',type=int, default=14,
                       help='num_workers')

# lr & lr scheduler
parser.add_argument('--lr',type=float, default=1e-1,
                       help='learning rate')

parser.add_argument('--lr_sch',type = str, default='fixed',choices=('cyc','fixed','lin'), 
                        help = 'If we want to use a fixed or cyc leraning schedule')

parser.add_argument('--lr_dec',type=float,default=3.0,
                       help='lr decay for linear schedule!')

parser.add_argument('--cycle_length',type=int, default=50,
                       help='Number of epochs in each cycle of cyclic lr schedule or save checkpoints!')

# regularizer
parser.add_argument('--wd',type=float,default=0,choices=(0,1,0.1,0.075,0.05,0.01,25),
                       help='weight decay')

parser.add_argument('--clip_grad',type = bool, default=False, 
                        help = 'If we want to clip grad or not!')

# inject noise & saving cechpoints
parser.add_argument('--epoch-noise',type =int, default=0, 
                        help = 'The epoch that we want to inject Gaussian noise, (set 0 if we do not want to inject noise)!')

parser.add_argument('--save_sample',type =bool, default =True, 
                        help = 'If we want to save samples or not!')

parser.add_argument('--epoch-st',type =int, default=40, 
                        help = 'The epoch that we want to start saving checkpints!')

parser.add_argument('--n_sam_cycle',type=int, default=1,
                       help='Number of samples in each cycle')

parser.add_argument('--N_samples',type =int, default=4, 
                        help = 'Total number of sample weights that we want to take!')

parser.add_argument('--scale',type =bool, default =False, 
                        help = 'If we want to scale the loss or not!')

                        
args = parser.parse_args() 


def main(args):
    
    seed =args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    
    
    ## setting device 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    
    ## making directory to save checkpoints 
    save_dir = Path('/dhc/home/netstore-old/Baysian/3D/Self_Supervised')
    save_dir_epoch = save_dir / 'simclr_ckpts' / args.ds
    save_dir_mcmc = save_dir / 'simclr_ckpts'/'samples'/ args.ds / f'{args.optimizer}_{args.exp}' 
    os.makedirs(save_dir_mcmc, exist_ok=True)
    
    ## saving version packages
    os.system(f'conda env export > {save_dir}/yml/{args.optimizer}_{args.exp}_env.yml')
    
    
    ## saving hyperparameters
    HPS = vars(args)    
    with open(save_dir /'simclr_params'/'pretrain'/args.ds/f'{args.exp}_{args.model_type}param.json','w') as file:    
        json.dump(HPS,file,indent=4)
        
    ## getting dataset and dataloaders
    if args.ds == 'cifar10':
        dataset = datasets.CIFAR10('./data',train=True,transform=pair_aug(get_transform(args.in_size,args.ds,args.s,args.aug)),\
                                   download=True)
        
    elif args.ds == 'cifar100':
        dataset = datasets.CIFAR100('./data',train=True,transform=pair_aug(get_transform(args.in_size,args.ds,args.s)),\
                                    download=True)
        
    elif args.ds == 'imagenet10':
        path_imagenet = save_dir/'core'/'data'/'imagenette2-320'/'train'     
        dataset = datasets.ImageFolder(path_imagenet,transform=pair_aug(get_transform(args.in_size,args.ds,args.s,args.aug)))
           
    elif args.ds == 'stl10':
        dataset = datasets.STL10('./data', split='unlabeled',transform=pair_aug(get_transform(args.in_size,args.ds,args.s)),\
                                 download = True)
        
    elif args.ds == 'tinyimagenet':
        path_tinyimagenet = save_dir/'core'/'data'/'tiny-imagenet-200'/'train'
        dataset = datasets.ImageFolder(path_tinyimagenet,transform=pair_aug(get_transform(args.in_size,args.ds,args.s)))
         
    train_loader,val_loader = get_train_val_loader(dataset,val_size=args.val_size,batch_size=args.batch_size,\
                                                   num_workers=args.num_workers,seed=args.seed)
    
    ## Setting parameters for cyclic learning rate schedule
    N_train = len(train_loader.dataset)
    n_batch = len(train_loader)
    cycle_batch_length = args.cycle_length * n_batch
    batch_idx = 0

    
    ## Building model 
    if args.model_depth=="res18":
        backbone = models.resnet18(pretrained=args.prtr, progress=True)
        
    elif args.model_depth=='res50':
        backbone = models.resnet50(pretrained=args.prtr, progress=True)
    
    online_network = network(backbone,args.mlp_hid_size,args.proj_size).to(device)
    
    
    ## getting optimizer
    if args.optimizer=='adam':
        
        optimizer = optim.Adam(online_network.parameters(),args.lr,weight_decay=args.wd)
        
    elif args.optimizer=='sgd':
        
        optimizer = optim.SGD(online_network.parameters(),lr=args.lr,weight_decay=args.wd/N_train,momentum=0.9)
        
        
    elif args.optimizer=='sghm':
        
        optimizer = SGHM(params=online_network.parameters(),lr=args.lr,weight_decay=args.wd/N_train,momentum=0.9,\
                         temp=args.temp,addnoise=1,dampening=DEFAULT_DAMPENING,N_train=N_train)
        
        
    ## getting loss
    criterion = NT_Xent(args.temp_loss,device)
        
    
    history = {'train':[], 'val':[]}
    best_val = float('inf')
    weight_set_samples = []
    sampled_epochs = []
    mt = 0 
    
    print(f'training is started')
    for epoch in range(args.num_epochs):
        
        tic = time.time()
        
        for phase in ['train','val']:
            
            if phase=='train':
                #print(f'############################')
                #print(f'# Training phase #')
                #print(f'############################')
                online_network.train()
                dataloader = train_loader
                
            else:
                
                #print(f'##############################')
                #print(f'# Validation phase #')
                #print(f'##############################')
                online_network.eval()
                dataloader= val_loader
                
            total_loss = 0
            
            for (img1,img2),_ in dataloader:
                
                img1= img1.to(device)
                img2= img2.to(device)
                
                proj1 = online_network(img1)
                proj2 = online_network(img2) 
                
                loss = Train(online_network,criterion,optimizer,proj1,proj2,phase,N_train,batch_idx,cycle_batch_length,epoch)
                             
                total_loss += loss.item()
                
                if phase=='train':    
                    batch_idx+=1
                            
            history[phase].append(total_loss/len(dataloader.dataset))
            
            if phase == 'train':
                metrics = {"loss_train": history[phase][-1],"epoch":epoch}
                
            elif phase == "val":
                metrics = {"loss_val": history[phase][-1],"epoch_val":epoch}
                
            wandb.log(metrics) 
            
            if args.save_sample:
            
                if epoch>=args.epoch_st and (epoch%args.cycle_length)+1>(args.cycle_length-args.n_sam_cycle) and phase=='train':

                    sampled_epochs.append(epoch)
                    if use_cuda:
                        online_network.cpu()
                    torch.save(online_network.state_dict(),os.path.join(save_dir_mcmc,f'model_{mt}.pt'))
                    mt +=1
                    online_network.cuda()        
                    print(f'sample {mt} from {args.N_samples} was taken!')
                       
        toc = time.time()
        runtime_epoch = toc - tic
        lr_epoch = optimizer.param_groups[0]['lr']
        
        print('Epoch: %d Train_Loss: %0.4f, Val_Loss: %0.4f, time:%0.4f seconds, lr:%0.7f'%(epoch,history['train'][epoch],\
                                                                                  history['val'][epoch],
                                                                                  runtime_epoch,lr_epoch))                                                                  
        ### save best model    
        if history['val'][epoch] < best_val:
            
            best_val = history['val'][epoch]
            check_without_progress = 0 
            torch.save({'epoch':epoch+1,
                        'model':online_network.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'Loss':history['val'][epoch]},os.path.join(save_dir_epoch,f'{args.exp}_best_{args.model_type}sclr.pt'))
               
    ### save last model
    torch.save({'epoch':epoch+1,
                'model':online_network.state_dict(),
                'optimizer':optimizer.state_dict(),
                'Loss':history['val'][epoch]},os.path.join(save_dir_epoch,f'{args.exp}_last_{args.model_type}sclr.pt'))
    print(f'save last model')
    
    ### save markov chain samples
    if args.save_sample: 
        torch.save(sampled_epochs,save_dir /'simclr_ckpts'/'samples'/ args.ds /f'{args.optimizer}_{args.exp}_epochs.pt') 
    
    ### plot learning curve
    plotCurves(history,save_dir / 'simclr_lr_curves'/ 'pretrain' / args.ds / f'exp_{args.exp}_{args.model_type}loss.png')    
                               
def Train(online_network,criterion,optimizer,proj1,proj2,phase,N_train,batch_idx,cycle_batch_length,epoch):
          
    loss = criterion(proj1,proj2)
    
    del proj1,proj2
    
    if phase=='train':
            
        # Update network
        if args.scale:
            loss = loss * N_train
            
        optimizer.zero_grad()
        
        # Update lr 
        if args.lr_sch == 'cyc':
            
            update_lr(args.lr,batch_idx,cycle_batch_length,args.n_sam_cycle,optimizer)
            
        if args.lr_sch == 'lin':
            
            lr = args.lr*np.exp(-args.lr_dec*min(1.0,(batch_idx*args.batch_size)/float(N_train)))

            update_lin(lr,optimizer)
        
        # Inject noise to parameter update
        if args.lr_sch == 'cyc' and args.epoch_noise:
            
            if (epoch%args.cycle_length)+1 > args.epoch_noise:
                
                optimizer.param_groups[0]['epoch_noise'] = True                       
            else:         
                optimizer.param_groups[0]['epoch_noise'] = False
                
                
        elif args.lr_sch in ['lin','fixed'] and args.epoch_noise:
            
            if epoch >= args.epoch_noise:
                
                optimizer.param_groups[0]['epoch_noise'] = True 
                
            else:
                optimizer.param_groups[0]['epoch_noise'] = False
            
        loss.backward() 
        optimizer.step()
                    
        if args.scale:
            loss = loss/N_train
            
    return(loss)


#### Running the code
if __name__=='__main__':
    
    main(args)
    


