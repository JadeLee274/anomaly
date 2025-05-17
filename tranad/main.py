from typing import *
import os
import pickle
from tqdm import tqdm
from time import time
from pprint import pprint
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from model.tranad import *
from utils.constants import *
from utils.compute import *
from utils.pot import *
from utils.diagnosis import *
OUTPUT_FOLDER = 'processed_data'
DATA_FOLDER = 'data'


def convert_to_windows(data, model):
    windows = []
    window_size = model.n_windows
    for i, g in enumerate(data):
        if i >= window_size:
            w = data[i - window_size: i]
        else:
            w = torch.cat([data[0].repeat(window_size - i, 1), data[0: i]])
        windows.append(
            w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1)
        )
    return torch.stack(windows)


def cut_array(
        percentage: float,
        arr: np.ndarray):
    print(f'Slicing dataset to {int(percentage * 100)}%')
    mid = round(arr.shape[0] / 2)
    window = round(arr.shape[0] * percentage * 0.5)
    return arr[mid - window : mid + window, :]


def load_dataset(dataset) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    folder = os.path.join(OUTPUT_FOLDER, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed data not found')
    
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'SMD':
            file = 'machine-1-1_' + file
        elif dataset == 'SMAP':
            file = 'P-1_' + file
        elif dataset == 'MSL':
            file = 'C-1_' + file
        elif dataset == 'UCR':
            file = '136_' + file
        elif dataset == 'NAB':
            file = 'ec2_request_latency_system_failure_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    
    if args.less:
        loader[0] = cut_array(0.2, loader[0])
    
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]

    return train_loader, test_loader, labels


def save_model(
    model: nn.Module,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    epoch: int,
    accuracy_list: List[float],
) -> None:
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.pth'
    torch.save(
        obj={
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy_list': accuracy_list,
        },
        f=file_path,
    )


def load_model(
    modelname: str,
    dims: int,
):
    import model.tranad
    model_class = getattr(model.tranad, modelname)
    tranad_model = model_class(dims).double()
    optimizer = optim.AdamW(
        params=tranad_model.parameters(),
        lr=tranad_model.lr,
        weight_decay=1e-5,
    )
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.9)
    file_name = f'checkpoints/{args.model}_{args.dataset}/model.pth'
    
    if os.path.exists(file_name) and (not args.retrain or args.test):
        print(f'Loading pre-trained model: {tranad_model.name}')
        checkpoint = torch.load(file_name)
        tranad_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f'Creating new model: {tranad_model.name}')
        epoch = -1
        accuracy_list = []
    
    return tranad_model, optimizer, scheduler, epoch, accuracy_list


def backprop(
    epoch: int,
    model: nn.Module,
    data, 
    dataO,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    training: bool = True,
):
    criterion = nn.MSELoss(reduction='none')
    feats = dataO.shape[1]
    data_x = torch.DoubleTensor(data)
    dataset = TensorDataset(data_x, data_x)
    bs = model.batch_size if training else len(data)
    dataloader = DataLoader(dataset, batch_size=bs)
    n = epoch + 1
    loss1_list = []

    if training:
        for d, _ in dataloader:
            local_bs = d.shape[0]
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, local_bs, feats)
            z = model(window, elem)
            loss1 = criterion(z, elem) if not isinstance(z, tuple) else (1/n) * criterion(z[0], elem) + (1  - 1/n) * criterion(z[1], elem)
            if isinstance(z, tuple):
                z = z[1]
            loss1_list.append(torch.mean(loss1).item())
            loss = torch.mean(loss1)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step()
        tqdm.write(f'Epoch {epoch + 1} | Loss1 = {np.mean(loss1_list)}')
        return np.mean(loss1_list), optimizer.param_groups[0]['lr']

    else:
        for d, _ in dataloader:
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, bs, feats)
            z = model(window, elem)
            if isinstance(z, tuple):
                z = z[1]
        loss = criterion(z, elem)[0]
        return loss.detach().numpy(), z.detach().numpy()[0]
            

if __name__ == '__main__':
    train_loader, test_loader, labels = load_dataset(args.dataset)
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

    # Prepare Data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

    # Training phase
    if not args.test:
        print(f'Training {args.model} on {args.dataset}')
        num_epochs = 5
        e = epoch + 1
        start = time()
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            lossT, lr = backprop(
                epoch=e,
                model=model,
                data=trainD,
                dataO=trainO,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            accuracy_list.append((lossT, lr))
        print(f'Training Time: {(time() - start):.4f}')
        save_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=e,
            accuracy_list=accuracy_list,
        )

    # Testing phase
    torch.zero_grad = True
    model.eval()
    print(f'Testing {args.model} on {args.dataset}')
    loss, y_pred = backprop(
        epoch=0,
        model=model,
        data=testD,
        dataO=testO,
        optimizer=optimizer,
        scheduler=scheduler,
        training=False,
    )
    
    # Scores
    df = pd.DataFrame()
    lossT, _ = backprop(
        epoch=0,
        model=model,
        data=trainD,
        dataO=trainO,
        optimizer=optimizer,
        scheduler=scheduler,
        training=False,
    )
    for i in range(loss.shape[1]):
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        result, pred = pot_eval(lt, l, ls)
        preds.append(pred)
        df = df._append(result, ignore_index=True)
    
    lossT_final = np.mean(lossT, axis=1)
    loss_final = np.mean(loss, axis=1)
    labels_final = (np.sum(labels, axis=1) >= 1) + 0
    
    result, _ = pot_eval(lossT_final, loss_final, labels_final)
    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))
    print(df)
    pprint(result)
