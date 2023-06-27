import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import get_configure, mkdir_p, init_trial_path, \
    split_dataset, collate_molgraphs, load_model, predict, init_featurizer, load_dataset
from Cal import Meter
from dgllife.utils import EarlyStopping
import os
from shutil import copyfile, copytree
# TODO 添加模型的参数以及调用
# TODO 解决pycharm的报错
# 可视化
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve



def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        if len(smiles) == 1:
            # Avoid potential issues with batch normalization
            continue

        labels, masks = labels.to(args['device']), masks.to(args['device'])
        logits = predict(args, model, bg)
        # Mask non-existing labels
        train_loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_meter.update(logits, labels, masks)
        if batch_id % args['print_every'] == 0:
            print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), train_loss.item()))
        # 可视化
        writer.add_scalar(tag='train_loss', scalar_value=train_loss, global_step=epoch)

    if args['metric'] == 'all_classification_results':
        train_result = train_meter.compute_metric(args['metric'])
        metric_value = [x for sublist in train_result for x in sublist]
        train_metric_name = ['train_' + name for name in metric_name]
        train_score = dict(zip(train_metric_name, metric_value))
        print(train_score)

        # 可视化部分
        writer.add_scalar(tag='train_roc_auc_score', scalar_value=train_score['train_roc_auc_score'], global_step=epoch)
        writer.add_scalar(tag='train_pr_auc_score', scalar_value=train_score['train_pr_auc_score'], global_step=epoch)
        writer.add_scalar(tag='train_mcc_score', scalar_value=train_score['train_recall_score'], global_step=epoch)
        writer.add_scalar(tag='train_acc_score', scalar_value=train_score['train_acc_score'], global_step=epoch)
        writer.add_scalar(tag='train_f1_score', scalar_value=train_score['train_f1_score'], global_step=epoch)
        writer.add_scalar(tag='train_mcc_score', scalar_value=train_score['train_mcc_score'], global_step=epoch)
        writer.add_scalar(tag='train_balanced_acc_score', scalar_value=train_score['train_balanced_acc_score'], global_step=epoch)
        writer.close()

    else:
        train_score = np.mean(train_meter.compute_metric(args['metric']))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric'], train_score))


def run_an_eval_epoch(args, epoch, model, data_loader, loss_criterion):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            logits = predict(args, model, bg)
            labels, masks = labels.to(args['device']), masks.to(args['device'])
            val_loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
            eval_meter.update(logits, labels, masks)
        # 可视化
        writer.add_scalar(tag='val_loss', scalar_value=val_loss, global_step=epoch)
    if args['metric'] == 'all_classification_results':
        val_result = eval_meter.compute_metric(args['metric'])
        metric_value = [x for sublist in val_result for x in sublist]
        val_metric_name = ['val_' + name for name in metric_name]
        val_score = dict(zip(val_metric_name, metric_value))
        val_score.update({'val_loss': val_loss.item()})
        val_score.update({'model': args['model']})
        # 可视化
        writer.add_scalar(tag='val_roc_auc_score', scalar_value=val_score['val_roc_auc_score'], global_step=epoch)
        writer.add_scalar(tag='val_pr_auc_score', scalar_value=val_score['val_pr_auc_score'], global_step=epoch)
        writer.add_scalar(tag='val_mcc_score', scalar_value=val_score['val_recall_score'], global_step=epoch)
        writer.add_scalar(tag='val_acc_score', scalar_value=val_score['val_acc_score'], global_step=epoch)
        writer.add_scalar(tag='val_f1_score', scalar_value=val_score['val_f1_score'], global_step=epoch)
        writer.add_scalar(tag='val_mcc_score', scalar_value=val_score['val_mcc_score'], global_step=epoch)
        writer.add_scalar(tag='val_balanced_acc_score', scalar_value=val_score['val_balanced_acc_score'],
                          global_step=epoch)
        writer.close()

    else:
        val_score = np.mean(eval_meter.compute_metric(args['metric']))
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric'],
            val_score, args['metric'], val_score))

    return val_score


def run_a_test_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.cpu()
            logits = predict(args, model, bg).cpu()
            eval_meter.update(logits, labels, masks)
    test_result = eval_meter.compute_metric(args['metric'])
    metric_value = [x for sublist in test_result for x in sublist]
    test_metric_name = ['test_' + name for name in metric_name]
    test_score = dict(zip(test_metric_name, metric_value))
    test_score.update({'model': args['model']})

    # 保存tpr，fpr，方便日后作图
    fpr, tpr, thresholds = roc_curve(labels, logits)
    fpr = fpr.tolist()
    tpr = tpr.tolist()
    dict_tfpr ={args['model']: (fpr, tpr)}

    return test_score, dict_tfpr




def main(args, exp_config, train_set, val_set, test_set):
    # Record settings
    exp_config.update({
        'model': args['model'],
        'n_tasks': args['n_tasks'],
        'atom_featurizer_type': args['atom_featurizer_type'],
        'bond_featurizer_type': args['bond_featurizer_type']
    })
    if args['atom_featurizer_type'] != 'pre_train':
        exp_config['in_node_feats'] = args['node_featurizer'].feat_size()
    if args['edge_featurizer'] is not None and args['bond_featurizer_type'] != 'pre_train':
        exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()


    # 加载数据
    train_loader = DataLoader(dataset=train_set, batch_size=exp_config['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_set, batch_size=exp_config['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader = DataLoader(dataset=test_set,
                             batch_size=len(test_set),
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    # 加载模型参数
    model = load_model(exp_config)
    model = model.to(args['device'])
    # 设置优化器和早停
    loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=exp_config['lr'],
                     weight_decay=exp_config['weight_decay'])
    if args['early_stop'] == 'loss':
        mode = 'lower'
    else:
        mode = 'higher'
    stopper = EarlyStopping(patience=exp_config['patience'],
                            mode=mode,
                            filename=args['trial_path'] + '/model.pth')

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)

        # Validation
        val_score = run_an_eval_epoch(args, epoch, model, val_loader, loss_criterion)
        if epoch >= 60:
            early_stop = stopper.step(val_score['val_'+args['early_stop']], model)
            print('epoch {:d}/{:d}, validation {} {}, best validation {} {:.4f}'.format(
                epoch + 1, args['num_epochs'], args['metric'],
                val_score, args['early_stop'], stopper.best_score))

            if early_stop:
                break
    stopper.load_checkpoint(model)
    # Test
    test_score, tfpr = run_a_test_epoch(args, model, test_loader)
    print('test {} {}'.format(args['metric'], test_score))

    return val_score, test_score, tfpr


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Multi-label Binary Classification')
    parser.add_argument('-c', '--csv-path', type=str, required=True,
                        help='Path to a csv file for loading a dataset')
    parser.add_argument('-sc', '--smiles-column', type=str, required=True,
                        help='Header for the SMILES column in the CSV file')
    parser.add_argument('-t', '--task-names', default=None, type=str,
                        help='Header for the tasks to model. If None, we will model '
                             'all the columns except for the smiles_column in the CSV file. '
                             'The header cannot contain spaces'
                             '(default: None)')
    parser.add_argument('-s', '--split',
                        choices=['scaffold_decompose', 'scaffold_smiles', 'random', 'custom'],
                        default='scaffold_smiles',
                        help='Dataset splitting method (default: scaffold_smiles). For scaffold '
                             'split based on rdkit.Chem.AllChem.MurckoDecompose, '
                             'use scaffold_decompose. For scaffold split based on '
                             'rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles, '
                             'use scaffold_smiles.')
    parser.add_argument('-sr', '--split-ratio', default='0.8,0.1,0.1', type=str,
                        help='Proportion of the dataset to use for training, validation and test, '
                             '(default: 0.8,0.1,0.1)')
    parser.add_argument('-me', '--metric', choices=['roc_auc_score', 'pr_auc_score', 'f1_score', 'mcc', 'recall', 'acc',
                                                    'precision', 'all_classification_results'],
                        default='roc_auc_score',
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-mo', '--model', choices=['GCN', 'GAT', 'Weave', 'MPNN', 'AttentiveFP',
                                                   'gin_supervised_contextpred',
                                                   'gin_supervised_infomax',
                                                   'gin_supervised_edgepred',
                                                   'gin_supervised_masking',
                                                   ],
                        default='GCN', help='Model to use (default: GCN)')
    parser.add_argument('-a', '--atom-featurizer-type', choices=['canonical', 'attentivefp'],
                        default='canonical',
                        help='Featurization for atoms (default: canonical)')
    parser.add_argument('-b', '--bond-featurizer-type', choices=['canonical', 'attentivefp'],
                        default='canonical',
                        help='Featurization for bonds (default: canonical)')
    parser.add_argument('-n', '--num-epochs', type=int, default=1000,
                        help='Maximum number of epochs allowed for training. '
                             'We set a large number by default as early stopping '
                             'will be performed. (default: 1000)')
    parser.add_argument('-nw', '--num-workers', type=int, default=1,
                        help='Number of processes for data loading (default: 1)')
    parser.add_argument('-pe', '--print-every', type=int, default=20,
                        help='Print the training progress every X mini-batches')
    parser.add_argument('-p', '--result-path', type=str, default='classification_results',
                        help='Path to save training results (default: classification_results)')
    parser.add_argument('-ne', '--num-evals', type=int, default=None,
                        help='Number of trials for hyperparameter search (default: None)')
    parser.add_argument('-es', '--early-stop', type=str, choices=['loss', 'pr_auc_score'],
                        default='loss',
                        help='Basis for selecting early stop (default: loss)')
    args = parser.parse_args().__dict__

    if torch.cuda.is_available():
        args['device'] = torch.device('cuda')
    else:
        args['device'] = torch.device('cpu')

    if args['task_names'] is not None:
        args['task_names'] = args['task_names'].split(',')

    if args['metric'] == 'all_classification_results':
        metric_name = ['roc_auc_score', 'pr_auc_score', 'mcc_score', 'f1_score', 'recall_score', 'precision_score',
                       'acc_score', 'balanced_acc_score', 'confusion_matrix']

    args = init_featurizer(args)  # 添加边的属性和节点的属性
    # Set up directory for saving results
    args, trail_id = init_trial_path(args)
    mkdir_p(args['result_path'])

    args['n_tasks'] = 1   # TODO 由于没做过多分类，所以先把这个只设置成一


    #  设置tensorboard
    global writer
    writer = SummaryWriter(log_dir=args['trial_path']+'/tensorboard{}'.format(trail_id))

    #  有些时候dgllife的数据集分割满足不了我们的需求，这时候就要用自己的分割方式，可以重写data_split文件夹里的程序来达到目的
    #  此处的load_dataset是将数据集转化为边和节点的书记
    if args['split'] == 'custom':
        train_set = pd.read_csv('./dataset/' + 'train.csv')
        val_set = pd.read_csv('./dataset/' + 'validation.csv')
        test_set = pd.read_csv('./dataset/' + 'test.csv')
        train_set = load_dataset(args, train_set)
        val_set = load_dataset(args, val_set)
        test_set = load_dataset(args, test_set)
    else:
        df = pd.read_csv(args['csv_path'])
        dataset = load_dataset(args, df)
        train_set, val_set, test_set = split_dataset(args, dataset)

    exp_config = get_configure(args['model'])

    val_score, test_score, tfpr = main(args, exp_config, train_set, val_set, test_set)

    # 保存训练结果
    df1 = pd.DataFrame.from_dict(val_score, orient='index').T
    df2 = pd.DataFrame.from_dict(test_score, orient='index').T
    df3 = pd.DataFrame.from_dict(tfpr, orient='index').T
    with open('{}/result.txt'.format(args['trial_path']), 'w') as f:
        f.write(' {}\n'.format(val_score))
        f.write(' {}\n'.format(test_score))
        f.write(' {}\n'.format(tfpr))
    df1.to_excel('{}/{}_val_score.xlsx'.format(args['trial_path'], args['model']), index=False)
    df2.to_excel('{}/{}_test_score.xlsx'.format(args['trial_path'], args['model']), index=False)
    df3.to_excel('{}/{}_test_tfpr.xlsx'.format(args['trial_path'], args['model']), index=False)

    # 保存训练参数
    with open(args['trial_path'] + '/configure.json', 'w') as f:
        json.dump(exp_config, f, indent=2)

    # Copy parameters
    copytree(args['trial_path'] + '/tensorboard{}'.format(trail_id), 'configure_compare/{}/tensorboard{}'.format(args['model'], trail_id))
    copyfile(args['trial_path'] + '/configure.json', 'configure_compare/{}/tensorboard{}'.format(args['model'],trail_id) + '/configure.json')
    # Copy finals
    dir_path = '{}/all_results'.format(args['result_path'])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        print(f'{dir_path} already exists')
    copyfile('{}/{}_test_tfpr.xlsx'.format(args['trial_path'], args['model']), '{}/all_results/{}_test_tfpr.xlsx'.format(args['result_path'], args['model']))
    copyfile('{}/{}_test_score.xlsx'.format(args['trial_path'], args['model']), '{}/all_results/{}_test_score.xlsx'.format(args['result_path'], args['model']))



