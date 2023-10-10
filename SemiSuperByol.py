import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import lr_scheduler
#### pakckages from totrchvison
import torchvision
from torchvision import datasets,transforms
import torchvision.models as models

##### python libraries
import os, glob
import random
import math
import json
import argparse
import time
import datetime
import collections
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path 
from sklearn.model_selection import ShuffleSplit
try:
    import cPickle as pickle
except:
    import pickle
import warnings 
warnings.filterwarnings('ignore')


import wandb

def plotCurves(loss,acc,exp,ckt,split,i,ds,save_dir=None,opt='sghm'):
    
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)

    plt.plot(loss['train'], label='train_loss')
    plt.plot(loss['val'], label='valid_loss')
        
    textsize = 12
    marker=5
    
    plt.xlabel('Epochs')
    plt.ylabel(f'Loss')
    plt.title(f'NLL on {split}% data')
    
    lgd = plt.legend(['train', 'validation'], markerscale=marker, 
                 prop={'size': textsize, 'weight': 'normal'}) 
    
    
    ax = plt.gca()
    
    plt.subplot(1,2,2)
    
    plt.plot(acc['train'], label='train')
    plt.plot(acc['val'], label='validation')
    
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.title(f'Accuracy on {split}% data')
    
    lgd = plt.legend(['train', 'validation'], markerscale=marker, 
                 prop={'size': textsize, 'weight': 'normal'})
    
    crv_dir = save_dir / f'{ds}_exp_{exp}_opt_{opt}_fine_{ckt}_semi_{split}%_model_{i}_loss.png'
    plt.savefig(crv_dir, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    plt.show()

#####
def get_accuracy(gt, pred):
    assert len(gt)==len(pred)
    right = 0
    for i in range(len(gt)):
        if gt[i]==pred[i]:
             right += 1.0
    return right/len(gt)
    

#### Preprocessing for test_set
def get_transform(ds,in_size=None):
    
    if ds == 'cifar10':
        mean_ds = [0.491, 0.482, 0.447]
        std_ds = [0.247, 0.243, 0.261]
        
    elif ds == 'cifar100':
        mean_ds = [0.507, 0.486, 0.440]
        std_ds = [0.267, 0.256, 0.276] 
        
    elif ds == 'stl10':
        mean_ds = [0.485, 0.456, 0.406]
        std_ds = [0.229, 0.224, 0.225]
        
    elif ds == 'imagenet10':
        mean_ds = [0.485, 0.456, 0.406]
        std_ds = [0.229, 0.224, 0.225]
        
        
    transform=transforms.Compose([transforms.Resize(size=(in_size,in_size)),
                                  transforms.ToTensor()])
    return(transform)

                                  
#### Pretrain model
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
    
    def __init__(self,net,backbone,mid_dim,out_dim):  
        super(network,self).__init__()
        
        self.net = net
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.projection = MLP(in_dim= backbone.fc.in_features,mlp_hid_size=mid_dim,proj_size=out_dim) 
        self.prediction = MLP(in_dim= out_dim,mlp_hid_size=mid_dim,proj_size=out_dim)
                
    def forward(self,x):
        
        embedding = self.encoder(x)
        
        embedding = embedding.view(embedding.size()[0],-1)
        project = self.projection(embedding)
        
        if self.net=='target':
            return(project)
        predict = self.prediction(project)            
        return(predict)

#### Building finetune model
class finetune_net(nn.Module):
    
    def __init__(self,model,in_dim,num_classes=0):
        super(finetune_net,self).__init__()
        
        self.model = model
        self.linear = nn.Linear(in_dim,num_classes)
        
    def forward(self,x):
        
        embeding = self.model(x)
        embeding = embeding.view(embeding.size()[0],-1)
        logits = self.linear(embeding) 
        
        return(logits)
    

##### Update learning rate 
def update_lr(optimizer,mult):

    for param_group in optimizer.param_groups:
        param_group['lr'] *= mult
        

### Single finetune
def single_finetune(exp,model,optimizer,criterion,train_loader,val_loader,epochs,\
                    save_dir_fine,ckt,i,device,train_s,ds,last,opt):
   
    crv_dir = save_dir_fine.parents[2] / 'lr_curves' / 'finetune' / ds
    
    hist = {'train':[], 'val':[]}
    accur= {'train':[], 'val':[]}    
    best_loss = 1000
    
    for epoch in range(epochs):
        
        tic = time.time()
            
        for phase in ['train','val']:
            if phase=='train':
                model.train()
                data_loader = train_loader
                    
            else:
                model.eval()
                data_loader = val_loader
                                                   
            total_loss = 0
            correct = 0
            
            for batch_idx,(img,lable) in enumerate(data_loader):
                
                img = img.to(device)
                lable = lable.to(device)
                
                logits = model(img)
                loss = criterion(logits,lable)
                
                if phase=='train':
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                total_loss += loss.item() 
                pred = logits.argmax(1)
                    
                correct += (pred==lable).sum().item()
                        
            error = total_loss/len(data_loader)
            acc = correct/len(data_loader.dataset)  
            
            hist[phase].append(error)
            accur[phase].append(acc)
            
            if phase == 'train':
                metrics = {"loss_train": error,"acc_train":acc,"epoch":epoch}
                
            elif phase == "val":
                metrics = {"loss_val": error,"acc_val":acc,"epoch_val":epoch}
                       
        toc = time.time()
        time_epoch = toc - tic
                
        print('Epoch: %d Train_Loss: %0.4f, Val_Loss: %0.4f,  Train_Acc: %0.4f, Val_Acc: %0.4f , time:%0.4f seconds'%\
              (epoch,hist['train'][epoch],hist['val'][epoch],accur['train'][epoch],accur['val'][epoch],time_epoch))
                
        ## saving checkpoints when val loss is minimum
        is_best = bool(hist['val'][epoch] < best_loss)
        best_loss = hist['val'][epoch] if is_best else best_loss
        loss_val = hist['val'][epoch]
        
        if is_best:
            
            checkpoints = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': loss_val}
            torch.save(checkpoints,os.path.join(save_dir_fine,f'{exp}_{train_s}%_{ckt}_{i}_best_model.pt'))
                
    ## saving last checkpoint
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': hist['val'][epoch]},\
                os.path.join(save_dir_fine,f'{exp}_{train_s}%_{ckt}_{i}_last_model.pt'))
    
    if last:
        plotCurves(hist,accur,exp,ckt,train_s,i,ds,crv_dir,opt=opt)


#### Testing on test set 
def inference(model,test_loader,criterion,device):
    
    model.eval()    
    test_loss =0     
    correct = 0
    out_pr = []
    gt_list = []
    logits = []
    
    with torch.no_grad():
        
        for batch_idx,(img,lable) in enumerate(test_loader):
                
            img = img.to(device)
            lable = lable.to(device)    
            output = model(img)
            out_pr.append(F.softmax(output,dim=1).data.cpu().numpy())
            logits.append(output.data.cpu().numpy())
            gt_list += list(lable.data.cpu().numpy())
            loss = criterion(output,lable)
            test_loss += loss.item()            
            pred = output.argmax(1)
            correct += (pred==lable).sum().item()
            
        error = test_loss / len(test_loader)        
        acc = correct / len(test_loader.dataset)
        
        out_pr = np.concatenate(out_pr)
        logits = np.concatenate(logits)
        
    return(out_pr,logits,gt_list,error,acc)


# Evaluation Metrics
def evaluation_metrics(pr_tot,logits,gt_list,device,ig_sam=0): 
    
    pr_tot = torch.from_numpy(pr_tot[ig_sam:])
    
    logits = torch.from_numpy(logits[ig_sam:])
    
    tot_ens = pr_tot.size()[0]
    
    mean_logits = torch.mean(logits,dim=0).to(device)
    
    mean_pr = torch.mean(pr_tot,dim=0).to(device)
            
    _, pred_label = torch.max(mean_pr, dim=1)
        
    pred_list = list(pred_label.data)
    
    acc = get_accuracy(gt_list, pred_list)
    
    gt = torch.tensor(gt_list).to(device)
    
    nll = nn.CrossEntropyLoss()(mean_logits,gt)
    
    print(f'########## Total Accuracy and NLL for {tot_ens} Ens ###########')    
    print(f'\n total nll is:{nll:0.4f} and total accuracy is: {acc:0.4f}')    
    print(f'#############################################')
    return(nll,acc)
    
#### Fine Tunning
parser = argparse.ArgumentParser(description='Baysian Byol finetunning')

parser.add_argument('--seed',type=int,default=42,
                    help='the seed for experiments!')

parser.add_argument('--exp',type=int,default=477,
                        help='ID of this expriment!')

parser.add_argument('--ds_pr',type=str,default='cifar10',choices=('cifar10','cifar100','stl10','tinyimagenet','imagenet10'),
                        help='Dataset that we pretrained the model on that!')

parser.add_argument('--ds_ft',type=str,default='cifar10', 
                    choices=('cifar100','cifar10','stl10','imagenet10','tinyimagenet'),
                        help='Dataset for semisupervised learning or transfer learning!')

# Fine-tunning parameters
parser.add_argument('--fine',type=bool, default=False,
                       help='if we want to do finetunning or not!')

parser.add_argument('--ckt',type=str, default='all',choices=('best','last','all'),
                       help='if we want to use best, or last or MCMC checkpoints (all)')

parser.add_argument('--burn_in',type=int,default=0,
                        help='The burn in epochs!')

parser.add_argument('--ckt_sp',type=int,default=None,
                        help='The specific checkpoint that we want to look at!')

parser.add_argument('--ckt_inf',type=str,default='best',choices=('best','last'),
                       help='if we want to use best, or last model for inference')

# Optimizer & lr
parser.add_argument('--opt',type=str,default='sghm',choices=('adam','sgd','sghm'),
                       help='optimizer which was used in pretraining!')

parser.add_argument('--nes',type=bool, default=True,
                       help='if we want to have nestrov for sgd or not')

parser.add_argument('--wd',type=float, default=0,
                       help='Weight decay for finetunning!')

parser.add_argument('--num_epochs',type=int, default=50,
                       help='the number of epoch for pretraining')

parser.add_argument('--lr',type=float, default=2e-4,
                       help='learning rate')

# Architecure
parser.add_argument('--model_type',type=str, default='cb', choices=('','b','c','cb'),
                       help='If we want to train byol, bbyol, cbyol, cbbyol!')

parser.add_argument('--model_depth',type=str,choices=('res18','res50','res100'),default='res18',
                        help='model to use (causion with mlp_hidden_size, projection_size)')

parser.add_argument('--mlp_hid_size',type=int, choices=(512,4096), default=512,
                       help='the mid size in MLP')

parser.add_argument('--proj_size',type=int, choices=(128,256), default=128,
                       help='the size of projection map')

parser.add_argument('--num_classes',type=int, default=10,
                       help='the number of output classes in test set and semisupervised setting!')

# Test set and dataloaders
parser.add_argument('--num_workers',type=int, default=4,
                       help='num_workers')

parser.add_argument('--batch_size',type=int, default=80,
                       help='the number of batch size for training')

parser.add_argument('--exp_split',nargs="+", default=[100],
                       help='the split of train set that we want to consider')

# Evaluation parameters  
parser.add_argument('--pr_ens',type=bool, default=False,
                       help='if we want to save pr ensemble or not (when we are not in weight & biases)!')

parser.add_argument('--eval',type=bool, default=True,
                       help='if we want to evaluate our model or not!')

parser.add_argument('--ig_sam',type=int, default=1,
                       help='if we want to ignore samples from begining!')

parser.add_argument('--write_exp',type=bool, default=True,
                       help='if we want to write results in a csv file!')

args = parser.parse_args()

def semi_supervised(args):
    
    print(f'Baysian Semi-supervised evaluation')

    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    
    
    ## setting device 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ## the directory to load checkpoints 
    save_dir  = Path('/netstore-old/Baysian/3D/Self_Supervised')
    load_dir  = save_dir / 'ckpts' 
    split_dir = save_dir / 'split_data' 
    exp_dir = save_dir / 'samples' / args.ds_pr 
    param_dir = save_dir / 'params' / 'finetune' / args.ds_ft / f'exp_{args.exp}_opt_{args.opt}_pr_{args.ds_pr}'
    
    print(f'making a directory for results!')
    result_dir_exp = save_dir / 'results' / f'{args.ds_ft}_{args.exp_split[0]}'/ f'{args.exp}_pr_{args.ds_pr}'

    
    os.makedirs(param_dir, exist_ok=True)
    os.makedirs(result_dir_exp,exist_ok=True)
    
    # making a dic from hyperparameters
    config = vars(args) 
    
    # loadig hyperparameters pretrained model
    with open(save_dir /'params'/'pretrain'/args.ds_pr/f'{args.exp}_{args.model_type}param.json') as file:    
        HPP = json.load(file)
        in_size = HPP['in_size']

    # loading test_set
    if config['ds_ft']=='cifar10':                          
        testset = datasets.CIFAR10('./data',train=False,\
                                   transform=get_transform(config['ds_ft'],in_size),download=True)
    
    elif config['ds_ft']=='cifar100':
        testset = datasets.CIFAR100('./data',train=False,\
                                    transform=get_transform(config['ds_ft'],in_size),download=True)
        
    elif config['ds_ft']=='stl10':
        testset = datasets.STL10('./data',split='test',\
                                    transform=get_transform(config['ds_ft'],in_size),download=True)
        
    elif config['ds_ft']=='imagenet10':
        path_imagenet = save_dir/'core'/'data'/'imagenette2-320'/'val'
        testset = datasets.ImageFolder(path_imagenet,transform=get_transform(config['ds_ft'],in_size))
        
    elif config['ds_ft']=='tinyimagenet':
        path_tinyimagenet = save_dir/'core'/'data'/'tiny-imagenet-200'/'val'
        testset = datasets.ImageFolder(path_tinyimagenet,transform=get_transform(config['ds_ft'],in_size))
                                                                                  
    test_loader = DataLoader(testset,batch_size=100,num_workers=config['num_workers'],drop_last=False,shuffle=False)
    
    # defining loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # loading ckt list
    ckt_list = os.listdir(os.path.join(save_dir,'samples/%s/%s_%d'%(config['ds_pr'],config['opt'],config['exp'])))
    ckt_epochs = torch.load(os.path.join(save_dir,'samples/%s/%s_%d_epochs.pt'%\
                                         (config['ds_pr'],config['opt'],config['exp'])),map_location=device)
    n_ckts_tot = config['ckt_sp'] if config['ckt_sp'] else len(ckt_list) 
    
    n_ckts = n_ckts_tot - config['burn_in']
   
    for train_split in config['exp_split']:
        
        print("\n--------------------")
        print("\nfine tunning on : {} % of {}".format(train_split,config['ds_ft']))
        print("--------------------\n")
        
        # saving hyperparameters
        with open(param_dir / f'semi_{train_split}%_param.json','w') as file:
            json.dump(config,file,indent=4)
        
        with open(os.path.join(split_dir,'train_set_%s_%d_%d%%.pickle'%(config['ds_ft'],in_size,train_split)),'rb') as f:
            train_set = pickle.load(f)

        with open(os.path.join(split_dir,'val_set_%s_%d_%d%%.pickle'%(config['ds_ft'],in_size,train_split)),'rb') as f:
            val_set = pickle.load(f) 
            
        train_loader=DataLoader(train_set,batch_size=config['batch_size'],num_workers=config['num_workers'],\
                                drop_last=False,shuffle=True)
        val_loader=DataLoader(val_set,batch_size=config['batch_size'],num_workers=config['num_workers'],\
                              drop_last=False,shuffle=True)
        
        pr_tot = []
        log_tot= []
        
        error_cycles = []  
        acc_cycles = []
        
        if config['fine']:
        
            for i in range(config['burn_in'],n_ckts_tot):
            
                print("\n ##################################################")
                print(f'{i+1}th model form epoch {ckt_epochs[i]} is loaded!')
                print("#####################################################")
            
                sam_dir = os.path.join(exp_dir ,'%s_%d'%(config['opt'],config['exp']))
            
                state_dict = torch.load(glob.glob(os.path.join(sam_dir,f'*_{i}.pt'))[0],map_location=device)
                
                if args.model_depth == 'res18':
                    
                    backbone = models.resnet18(pretrained=False, progress=True)
                    fv_size = 512    # the size of embedding vector in resnet18 after avg layer
                    
                elif args.model_depth == 'res50':
                    
                    backbone = models.resnet50(pretrained=False, progress=True)
                    fv_size = 2048  # the size of embedding vector in resnet50 after avg pooling layer
                    
                                  
                net = network('online',backbone,config['mlp_hid_size'],config['proj_size']).to(device)
            
                net.load_state_dict(state_dict)
            
                encoder = nn.Sequential(*list(net.children())[:-2])
                model = finetune_net(encoder,fv_size,config['num_classes']).to(device)
                            
                optimizer = optim.SGD(model.parameters(),lr=config['lr'],\
                                      nesterov=config['nes'],weight_decay=config['wd'],momentum=0.9)
            
                last = True if i==n_ckts_tot-1 else False
                
                save_dir_fine = save_dir / 'ckpts' / 'finetune' / config['ds_ft']
                
                single_finetune(config['exp'],model,optimizer,criterion,train_loader,val_loader,\
                                config['num_epochs'],save_dir_fine,config['ckt'],i,device,train_split,\
                                config['ds_ft'],last,config['opt'])
                             
                print(f'\n making inference on test set for {i}th checkpoint!')
                ckt_inf = torch.load(os.path.join(save_dir_fine,'%s_%d%%_%s_%d_%s_model.pt'%
                                                  (config['exp'],train_split,config['ckt'],i,\
                                                   config['ckt_inf'])),map_location=device)
                                                                                                                                 
                model.load_state_dict(ckt_inf['model'])
                best_epoch = ckt_inf['epoch']                                               
        
                out_pr, logits, gt_list, error, acc = inference(model,test_loader,criterion,device)
            
                error_cycles.append(error)
                acc_cycles.append(acc)
                
                pr_tot.append(out_pr)
                log_tot.append(logits)
            
            pr_tot = np.stack(pr_tot,axis=0)
            log_tot = np.stack(log_tot,axis=0)
        
            error_cycles = np.array(error_cycles)
            acc_cycles = np.array(acc_cycles)
        
            # save gt
            np.save(os.path.join(result_dir_exp,f'gts.npy'),gt_list)
        
        # saving the results 
        if config['pr_ens']:
            
            np.save(os.path.join(result_dir_exp,'burn_%d_pr_ens.npy'%(config['burn_in'])),pr_tot)
            np.save(os.path.join(result_dir_exp,'burn_%d_logits_ens.npy'%(config['burn_in'])),log_tot)
            
            np.save(os.path.join(result_dir_exp,'burn_%d_error_cycl.npy'%(config['burn_in'])),error_cycles)
            np.save(os.path.join(result_dir_exp,'burn_%d_acc_cycl.npy'%(config['burn_in'])),acc_cycles)
        
        # evaluating on test set
        if config['eval']:
            
            if not config['pr_ens']:
                
                pr_tot = np.load(os.path.join(result_dir_exp,'burn_%d_pr_ens.npy'%(config['burn_in'])))
                log_tot = np.load(os.path.join(result_dir_exp,'burn_%d_logits_ens.npy'%(config['burn_in'])))
                gt_list = np.load(os.path.join(result_dir_exp,f'gts.npy'))
            
            nll_tot,acc_tot = evaluation_metrics(pr_tot,log_tot,gt_list,device,ig_sam=config['ig_sam'])
        
        if config['write_exp']:
            
            data = {
                'exp': [config['exp']],
                'opt':[config['opt']],
                'ds' :[config['ds_ft']],
                'ds_pr' :[config['ds_pr']],
                'split':[config['exp_split'][0]],
                'lr' :[config['lr']],
                'wd' :[config['wd']],
                'b_size':[config['batch_size']],
                'bur':[config['burn_in']],
                'ig_sam':[config['ig_sam']],
                'n_ensembel': [n_ckts],
                'NLL':[round(nll_tot.item(),4)],
                'ACC':[round(acc_tot,4)]}
        
            # ./results
            csv_path = Path(os.path.join(save_dir,'results/%s_%d/run_sweeps_test.csv'%(config['ds_ft'],config['exp_split'][0])))

            if os.path.exists(csv_path):

                sweeps_df = pd.read_csv(csv_path)
                sweeps_df = sweeps_df.append(
                pd.DataFrame.from_dict(data), ignore_index=True).set_index('exp')

            else:
            
                csv_path.parent.mkdir(parents=True, exist_ok=True) 
                sweeps_df = pd.DataFrame.from_dict(data).set_index('exp')

            # save experiment metadata csv file
            sweeps_df.to_csv(csv_path)
        
    print(f'\n Fine tunning was finished!\n')
    
##### Running finte tunning

if __name__=='__main__':
    
    semi_supervised(args)
         
        
        
        
        
                 
        
        
    
    
    

    
    
    
    
    
    
