import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from utils import *
from model.mymodel import MYModel
from model.SupConLoss import SupConLoss
from data.SGCC.dataset import SGCC
from sklearn import metrics
import time
import warnings
import yaml

warnings.filterwarnings('ignore')


def main_work(gpu, ngpus, args):
    '''
    args:
        gpu: 使用的gpu，如果是None代表使用cpu
    '''
    
    if args['train']['train_mode'] == 'dp':
        device = torch.device('cuda:0')
    elif gpu is not None:
        device = torch.device('cuda', gpu)
    else:
        device = torch.device('cpu')

    if args['train']['train_mode'] == 'ddp':
        dist.init_process_group(backend="nccl", init_method="env://", world_size=ngpus, rank=gpu)
    
    train_dataset = SGCC(args['train']['train_test_rate'], gpu, train=True, distributed=args['train']['train_mode'] == 'ddp', 
                            dropout=args['train']['dropout'], lttb=args['train']['lttb'])
    test_dataset = SGCC(args['train']['train_test_rate'], gpu, train=False)

    if args['train']['train_mode'] == 'ddp':
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        warnings.warn('You have set distributed training. '
                      'This will disable data rebalancing! ')
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['train']['batch_size'], shuffle=train_sampler is None,
        num_workers=args['train']['workers'], pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,batch_size=args['train']['batch_size'], shuffle=False,
        num_workers=args['train']['workers'], pin_memory=True)

    model = MYModel(args['model']['layer_num'], args['model']['hidden_nc'], 
                    args['model']['kernel_size'], args['model']['feature_dim'], 
                    args['model']['project_dim'], args['model']['classify_dim']).to(device)

    # distribute training
    if args['train']['train_mode'] == 'ddp':
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    elif args['train']['train_mode'] == 'dp':
        model = torch.nn.DataParallel(model)

    optimizor = torch.optim.Adam(model.parameters(), lr=args['train']['lr'])
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor(args['train']['loss_weight']).to(device))
    supcon_loss_fn = SupConLoss(temperature=args['train']['tau'])

    for e in range(args['train']['epoch']):
        if args['train']['train_mode'] != 'ddp' or gpu == 0:
            print('\n############## Epoch ', e+1, ' ###############')

        if args['train']['train_mode'] == 'ddp':
            train_sampler.set_epoch(e) 

        ## train
        model.train()
        for batch_input in train_loader:
            y = batch_input[0].squeeze().to(device)
            x = batch_input[1].to(device)

            bsz = x.size(0)
            xx = batch_input[2].to(device)                      # [B, 2, 2, H, W]
            xx = torch.concat(torch.unbind(xx, dim=1), dim=0)   # [2B, 2, H, W]
            out, supcon_out = model(x, xx)
            o1, o2 = torch.split(supcon_out, [bsz, bsz], dim=0)   # o1=o2=[B, PF]
            supcon_out = torch.cat([o1.unsqueeze(1), o2.unsqueeze(1)], dim=1) ## [B, 2, PF]

            loss = loss_fn(out, y) + args['train']['lambda'] * supcon_loss_fn(supcon_out, y)

            model.zero_grad()
            loss.backward()
            optimizor.step()

        ## test
        if args['train']['train_mode'] != 'ddp' or gpu == 0:
            auc, f1, recall, loss, confusion = test(model, test_loader, loss_fn, device)
            msg = 'Test Loss: {0:>2.4},  Test AUC: {1:>1.4},  Test F1-Score: {2:1.4},  Test Recall: {3:1.4}\n'
            print(msg.format(loss, auc, f1, recall))
            print("Confusion Matrix...")
            print(confusion)

        if args['train']['train_mode'] == 'ddp':
            torch.distributed.barrier()

        
def test(model, dataloader, loss_fn, device):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for batch_input in dataloader:
            y = batch_input[0].squeeze().to(device)
            x = batch_input[1].to(device)

            out = model(x)

            loss_total += loss_fn(out, y)
            true = y.data.cpu().numpy()
            predic = torch.argmax(out.data, 1).cpu().numpy()
            predict_all = np.append(predict_all, predic)
            labels_all = np.append(labels_all, true)

    auc = metrics.roc_auc_score(labels_all, predict_all)
    f1 = metrics.f1_score(labels_all, predict_all)
    recall = metrics.recall_score(labels_all, predict_all)
    confusion = metrics.confusion_matrix(labels_all, predict_all)

    return auc, f1, recall, loss_total / len(dataloader), confusion


if __name__ == '__main__':

    with open("model/model_config.yaml", 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f)

    start_time = time.time()
    if args['train']['train_mode'] == 'ddp':
        print('distribute training!')
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '6789'
        free_gpus = get_free_gpu(num=args['ngpus'])
        os.environ["CUDA_VISIBLE_DEVICES"] = free_gpus
        nprocs = torch.cuda.device_count()
        mp.spawn(main_work, args=(nprocs, args), nprocs=nprocs)

    elif args['train']['train_mode'] == 'dp':
        print('data parallel training!')
        free_gpus = get_free_gpu(num=args['ngpus'])
        os.environ["CUDA_VISIBLE_DEVICES"] = free_gpus
        main_work(args['train']['use_gpu'], None)

    elif args['train']['train_mode'] == 'gpu':
        print('single GPU training, use GPU ', args['train']['use_gpu'], ' !')
        main_work(args['train']['use_gpu'], None, args)

    else:
        print('cpu training!')
        main_work(None, None, args)
    
    use_seconds = int(time.time() - start_time)
    m, s = divmod(use_seconds, 60)
    h, m = divmod(m, 60)
    print('Time to run the loop: ', "%02d:%02d:%02d" % (h, m, s))