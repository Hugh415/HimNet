import warnings
warnings.filterwarnings("ignore", message="It is not recommended to directly access")
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from model import HimNet
from dataset import scaffold_split
from dataset import MoleculeDataset
import datetime
from utils import generate_random_seed, set_seed, save_dataset_split, load_dataset_split
from train import train, train_reg
from eval import eval, eval_reg

def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of FH-GNN')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for the prediction layer')
    parser.add_argument('--dataset', type=str, default = 'bbbp', help='[bbbp, bace, sider, clintox,tox21, toxcast, esol,freesolv,lipophilicity,lmc,metstab]')
    parser.add_argument('--data_dir', type=str, default='./data/', help = "the path of input CSV file")
    parser.add_argument('--save_dir', type=str, default='./model_checkpoints', help = "the path to save output model")
    parser.add_argument('--depth', type=int, default=7, help = "the depth of molecule encoder")
    parser.add_argument('--seed', type=int, default=88, help = "seed for splitting the dataset")
    parser.add_argument('--runseed', type=int, default=None, help = "seed for minibatch selection, random initialization")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--load_split', action='store_true', help='load existing dataset split instead of creating new one')
    parser.add_argument('--split_dir', type=str, default='./dataset_splits', help='directory for saving/loading dataset splits')
    
    args = parser.parse_args()

    if args.runseed is None:
        args.runseed = generate_random_seed()

    set_seed(args.runseed)
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset in ['tox21', 'bace', 'bbbp', 'sider', 'clintox', 'metstab']:
        task_type = 'cls'
    else:
        task_type = 'reg'

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == 'esol':
        num_tasks = 1
    elif args.dataset == 'freesolv':
        num_tasks = 1
    elif args.dataset == 'lipophilicity':
        num_tasks = 1
    elif args.dataset == 'lmc':
        num_tasks = 1
    elif args.dataset == 'metstab':
        num_tasks = 2
    elif args.dataset == 'malaria':
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    print('process data')
    dataset = MoleculeDataset(os.path.join(args.data_dir, args.dataset), dataset=args.dataset)

    # set up criterion
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # 
    if args.load_split:
        print(f"Trying to load a dataset split，seed = {args.seed}")
        train_dataset, valid_dataset, test_dataset = load_dataset_split(
            args.dataset, args.seed, args.split_dir
        )
        # Dataset segmentation
        if train_dataset is None:
            print("Unable to load existing split, create new split")
            smiles_list = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'processed/smiles.csv'), header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = scaffold_split(
                dataset, smiles_list, null_value=0, 
                frac_train=0.8, frac_valid=0.1, frac_test=0.1, 
                seed=args.seed
            )
            
            save_dataset_split(train_dataset, valid_dataset, test_dataset, 
                              args.dataset, args.seed, args.split_dir)
    else:
        print(f"Create a new dataset split using seed {args.seed}")
        smiles_list = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'processed/smiles.csv'), header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, 
            frac_train=0.8, frac_valid=0.1, frac_test=0.1, 
            seed=args.seed
        )
        
        save_dataset_split(train_dataset, valid_dataset, test_dataset, 
                          args.dataset, args.seed, args.split_dir)
    
    print(f"Split complete - training set: {len(train_dataset)}, validation set: {len(valid_dataset)}, test set: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

    model = HimNet(data_name=args.dataset, atom_fdim=89, bond_fdim=98,fp_fdim=6338, 
                  hidden_size=512, depth=args.depth, device=device, out_dim=num_tasks,)
    model.to(device)

    #set up optimizer
    model_param_group = []
    model_param_group.append({"params": model.parameters(), "lr":args.lr})
    optimizer = optim.Adam(model_param_group)
    print(optimizer)

    model_save_path = os.path.join(args.save_dir, args.dataset + '.pth')

    best_result_path = os.path.join('./best_result/', args.dataset + '.txt')
    if task_type == 'cls':
        best_auc = 0
        train_auc_list, test_auc_list = [], []
        for epoch in range(1, args.epochs+1):
            print('====epoch:',epoch)
            train(model, device, train_loader, optimizer, criterion)
            
            print('====Evaluation')
            if args.eval_train:
                train_auc, train_loss = eval(args, model, device, train_loader, criterion)
            else:
                print('omit the training accuracy computation')
                train_auc = 0
            val_auc, val_loss = eval(args, model, device, val_loader, criterion)
            test_auc, test_loss = eval(args, model, device, test_loader, criterion)
            test_auc_list.append(float('{:.4f}'.format(test_auc)))
            train_auc_list.append(float('{:.4f}'.format(train_auc)))

            if best_auc < test_auc:
                best_auc = test_auc
                torch.save(model.state_dict(), model_save_path)

                with open(best_result_path, 'w') as f:
                    f.write(f"Current dataset：{args.dataset}\n")
                    current_time = datetime.datetime.now()
                    f.write(f"Save time：{current_time}\n")
                    f.write(f"train set AUC: {train_auc} val set AUC: {val_auc} test set AUC: {test_auc}\n")
                    f.write(f"The number of rounds epoch for the current experiment is：{epoch}\n")
                    f.write(f"Data Set Segmentation Seeds: {args.seed}\n")
                    f.write(f"Training random seeds: {args.runseed}\n")
                    print(f"The current epoch with the best performance has been saved to {best_result_path}")
            
            print("train_auc: %f val_auc: %f test_auc: %f" %(train_auc, val_auc, test_auc))

    elif task_type == 'reg':
        train_list, test_list = [], []
        best_rmse = 100
        
        for epoch in range(1, args.epochs+1):
            print('====epoch:',epoch)
            
            train_reg(args, model, device, train_loader, optimizer)

            print('====Evaluation')
            if args.eval_train:
                train_result = eval_reg(args, model, device, train_loader)
                train_mse, train_mae, train_rmse = train_result[0:3]
            else:
                print('omit the training accuracy computation')
                train_mse, train_mae, train_rmse = 0, 0, 0
                
            val_result = eval_reg(args, model, device, val_loader)
            val_mse, val_mae, val_rmse = val_result[0:3]
            
            test_result = eval_reg(args, model, device, test_loader)
            test_mse, test_mae, test_rmse = test_result[0:3]
            
            test_list.append(float('{:.6f}'.format(test_rmse)))
            train_list.append(float('{:.6f}'.format(train_rmse)))
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                torch.save(model.state_dict(), model_save_path)

                with open(best_result_path, 'w') as f:
                    f.write(f"Current dataset：{args.dataset}\n")
                    current_time = datetime.datetime.now()
                    f.write(f"Save time：{current_time}\n")
                    f.write(f"train set RMSE: {train_rmse} val set RMSE: {val_rmse} test set RMSE: {test_rmse}\n")
                    f.write(f"Data Set Segmentation Seeds: {args.seed}\n")
                    f.write(f"Training random seeds: {args.runseed}\n")

                    f.write(f"The number of rounds epoch for the current experiment is：{epoch}\n")
                    print(f"The current epoch with the best performance has been saved to {best_result_path}")
                    
            print("Average - train_rmse: %f val_rmse: %f test_rmse: %f" %(train_rmse, val_rmse, test_rmse))


    with open(best_result_path, 'a') as f:
        f.write(f"\nExperimental randomized seeds: {args.runseed}\n")
        f.write(f"Training is complete. ✅\n")

if __name__ == "__main__":
    main()