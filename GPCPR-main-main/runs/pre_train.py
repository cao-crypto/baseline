""" Pre-train phase


"""

import os
import sys
import subprocess
import platform

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloaders.loader import MyPretrainDataset
from models.dgcnn import DGCNN
from models.dgcnn_new import DGCNN_semseg
from utils.logger import init_logger
from utils.checkpoint_util import save_pretrain_checkpoint


def diagnose_environment():
    """Diagnose environment for OpenMP and dependency issues"""
    print('\n=== Environment Diagnostics ===')
    print(f'  Python executable: {sys.executable}')
    print(f'  PyTorch version: {torch.__version__}')
    print(f'  NumPy version: {np.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    
    if platform.system().lower() == "windows":
        print('\n  Checking for duplicate OpenMP DLLs...')
        try:
            result = subprocess.run(
                ['where', 'libiomp5md.dll'],
                capture_output=True,
                text=True,
                timeout=10
            )
            paths = [p.strip() for p in result.stdout.strip().split('\r\n') if p.strip()]
            if len(paths) > 1:
                print(f'  WARNING: Found {len(paths)} libiomp5md.dll files:')
                for i, path in enumerate(paths):
                    print(f'    {i+1}. {path}')
                print('  This may cause OMP Error #15. Clean your environment.')
            else:
                print(f'  Found {len(paths)} libiomp5md.dll')
        except Exception as e:
            print(f'  Failed to check OpenMP DLLs: {e}')
    print('=== End Environment Diagnostics ===\n')


def get_safe_num_workers(args, phase_name):
    """Get safe number of DataLoader workers for the current platform"""
    n_workers = int(args.n_workers)
    if platform.system().lower() == "windows":
        if n_workers > 2:
            print(f"[Warning] Windows DataLoader num_workers={n_workers} is unstable; capping to 2.")
            print(f"          Use --n_workers 0 if any crash occurs.")
            n_workers = 2
    return n_workers


def build_pretrain_loader(dataset, args, shuffle, drop_last, phase_name):
    """Build DataLoader with safe configuration for Windows"""
    safe_n_workers = get_safe_num_workers(args, phase_name)
    
    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        'drop_last': drop_last,
        'num_workers': safe_n_workers,
        'pin_memory': torch.cuda.is_available(),
        'timeout': 0,
    }
    
    # Only pass prefetch_factor and persistent_workers when safe_n_workers > 0
    if safe_n_workers > 0:
        loader_kwargs['prefetch_factor'] = 1
        loader_kwargs['persistent_workers'] = False
    
    return DataLoader(dataset, **loader_kwargs), safe_n_workers


def validate_pretrain_dataset(dataset, dataset_name, logger, max_samples=16):
    """Validate dataset samples in main process before creating DataLoader"""
    # Print OpenMP thread settings
    omp_threads = os.environ.get('OMP_NUM_THREADS', 'not set')
    mkl_threads = os.environ.get('MKL_NUM_THREADS', 'not set')
    numexpr_threads = os.environ.get('NUMEXPR_NUM_THREADS', 'not set')
    logger.cprint(f'[OpenMP] OMP_NUM_THREADS={omp_threads}, MKL_NUM_THREADS={mkl_threads}, NUMEXPR_NUM_THREADS={numexpr_threads}')
    
    if len(dataset) == 0:
        raise RuntimeError(f'{dataset_name} dataset is empty')
    
    logger.cprint(f'\n=== Validating {dataset_name} dataset ===')
    logger.cprint(f'  Total samples: {len(dataset)}')
    
    # Show first few block names
    if hasattr(dataset, 'block_names'):
        num_blocks_to_show = min(5, len(dataset.block_names))
        logger.cprint(f'  First {num_blocks_to_show} blocks: {dataset.block_names[:num_blocks_to_show]}')
    
    num_samples = min(max_samples, len(dataset))
    first_ptcloud_shape = None
    first_label_shape = None
    
    for i in range(num_samples):
        try:
            ptcloud, labels = dataset[i]
            
            # Capture first sample shapes
            if i == 0:
                first_ptcloud_shape = ptcloud.shape
                first_label_shape = labels.shape
            
            # Check ptcloud is tensor
            if not isinstance(ptcloud, torch.Tensor):
                raise RuntimeError(f'ptcloud is {type(ptcloud)}, expected torch.Tensor')
            
            # Check labels is tensor
            if not isinstance(labels, torch.Tensor):
                raise RuntimeError(f'labels is {type(labels)}, expected torch.Tensor')
            
            # Check ptcloud shape [C, num_point]
            if len(ptcloud.shape) != 2:
                raise RuntimeError(f'ptcloud shape is {ptcloud.shape}, expected [C, num_point]')
            
            # Check labels shape [num_point]
            if len(labels.shape) != 1:
                raise RuntimeError(f'labels shape is {labels.shape}, expected [num_point]')
            
            # Check finite values
            if not torch.isfinite(ptcloud).all():
                raise RuntimeError(f'ptcloud contains inf/nan values')
            
            # Check labels are integer type
            if not torch.is_floating_point(labels):
                labels = labels.long()
            
        except Exception as e:
            block_name = getattr(dataset, 'block_names', [None])[i] if i < len(getattr(dataset, 'block_names', [])) else None
            npy_path = os.path.join(dataset.data_path, 'data', f'{block_name}.npy') if block_name else 'unknown'
            raise RuntimeError(
                f'Validation failed for {dataset_name} dataset\n'
                f'  Index: {i}\n'
                f'  Block: {block_name}\n'
                f'  NPY path: {npy_path}\n'
                f'  Error: {str(e)}'
            ) from e
    
    # Print first sample shapes after validation
    logger.cprint(f'  First sample: ptcloud shape={first_ptcloud_shape}, labels shape={first_label_shape}')
    logger.cprint(f'=== Validation passed for {dataset_name} dataset ===\n')


class DGCNNSeg(nn.Module):
    def __init__(self, args, num_classes):
        super(DGCNNSeg, self).__init__()
        if args.use_high_dgcnn:
            self.encoder = DGCNN_semseg(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        else:
            self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        in_dim = args.dgcnn_mlp_widths[-1]
        for edgeconv_width in args.edgeconv_widths:
            in_dim += edgeconv_width[-1]
        self.segmenter = nn.Sequential(
                            nn.Conv1d(in_dim, 256, 1, bias=False),
                            nn.BatchNorm1d(256),
                            nn.LeakyReLU(0.2),
                            nn.Conv1d(256, 128, 1),
                            nn.BatchNorm1d(128),
                            nn.LeakyReLU(0.2),
                            nn.Dropout(0.3),
                            nn.Conv1d(128, num_classes, 1)
                         )

    def forward(self, pc):
        num_points = pc.shape[2]
        edgeconv_feats, point_feat, _ = self.encoder(pc)
        global_feat = point_feat.max(dim=-1, keepdim=True)[0]
        edgeconv_feats.append(global_feat.expand(-1,-1,num_points))
        pc_feat = torch.cat(edgeconv_feats, dim=1)

        logits = self.segmenter(pc_feat)
        return logits


def metric_evaluate(predicted_label, gt_label, NUM_CLASS):
    """
    :param predicted_label: (B,N) tensor
    :param gt_label: (B,N) tensor
    :return: iou: scaler
    """
    gt_classes = [0 for _ in range(NUM_CLASS)]
    positive_classes = [0 for _ in range(NUM_CLASS)]
    true_positive_classes = [0 for _ in range(NUM_CLASS)]

    for i in range(gt_label.size()[0]):
        pred_pc = predicted_label[i]
        gt_pc = gt_label[i]

        for j in range(gt_pc.shape[0]):
            gt_l = int(gt_pc[j])
            pred_l = int(pred_pc[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l == pred_l)

    oa = sum(true_positive_classes)/float(sum(positive_classes))
    print('Overall accuracy: {0}'.format(oa))
    iou_list = []

    for i in range(NUM_CLASS):
        denom = gt_classes[i] + positive_classes[i] - true_positive_classes[i]
        if denom == 0:
            iou_class = float('nan')
        else:
            iou_class = true_positive_classes[i] / float(denom)
        print('Class_%d: iou_class is %f' % (i, iou_class))
        iou_list.append(iou_class)

    # Use nanmean to handle classes with no union
    mean_IoU = np.nanmean(np.array(iou_list[1:]))

    return oa, mean_IoU, iou_list


def pretrain(args):
    logger = init_logger(args.log_dir, args)
    
    # Diagnose environment (optional, helps identify issues)
    diagnose_environment()

    # Init datasets, dataloaders, and writer
    PC_AUGMENT_CONFIG = {'scale': args.pc_augm_scale,
                         'rot': args.pc_augm_rot,
                         'mirror_prob': args.pc_augm_mirror_prob,
                         'jitter': args.pc_augm_jitter,
                         'shift': args.pc_augm_shift,
                         'random_color': args.pc_augm_color,
                         }

    if args.dataset == 's3dis':
        from dataloaders.s3dis import S3DISDataset
        DATASET = S3DISDataset(args.cvfold, args.data_path)
    elif args.dataset == 'scannet':
        from dataloaders.scannet import ScanNetDataset
        DATASET = ScanNetDataset(args.cvfold, args.data_path)
    else:
        raise NotImplementedError('Unknown dataset %s!' % args.dataset)

    CLASSES = DATASET.train_classes
    NUM_CLASSES = len(CLASSES) + 1
    CLASS2SCANS = {c: DATASET.class2scans[c] for c in CLASSES}

    TRAIN_DATASET = MyPretrainDataset(args.data_path, CLASSES, CLASS2SCANS, mode='train',
                                      num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                      pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

    VALID_DATASET = MyPretrainDataset(args.data_path, CLASSES, CLASS2SCANS, mode='test',
                                      num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                      pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

    # Log dataset info
    logger.cprint('=== Pre-train Dataset (classes: {0}) ==='.format(CLASSES))
    logger.cprint('  Train dataset: {0} blocks'.format(len(TRAIN_DATASET)))
    logger.cprint('  Valid dataset: {0} blocks'.format(len(VALID_DATASET)))
    
    # Show first few block names
    if hasattr(TRAIN_DATASET, 'block_names'):
        logger.cprint('  Train blocks (first 5): {0}'.format(TRAIN_DATASET.block_names[:5]))
    if hasattr(VALID_DATASET, 'block_names'):
        logger.cprint('  Valid blocks (first 5): {0}'.format(VALID_DATASET.block_names[:5]))

    # Validate dataset in main process
    validate_pretrain_dataset(TRAIN_DATASET, 'TRAIN', logger)
    validate_pretrain_dataset(VALID_DATASET, 'VALID', logger)

    # Create DataLoaders
    TRAIN_LOADER, train_workers = build_pretrain_loader(TRAIN_DATASET, args, shuffle=True, drop_last=True, phase_name='train')
    VALID_LOADER, valid_workers = build_pretrain_loader(VALID_DATASET, args, shuffle=False, drop_last=True, phase_name='valid')
    
    # Log final worker counts
    logger.cprint('=== DataLoader Configuration ===')
    logger.cprint('  Train loader workers: {0}'.format(train_workers))
    logger.cprint('  Valid loader workers: {0}'.format(valid_workers))

    WRITER = SummaryWriter(log_dir=args.log_dir)

    # Init model and optimizer
    model = DGCNNSeg(args, num_classes=NUM_CLASSES)
    print(model)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr': args.pretrain_lr}, \
                           {'params': model.segmenter.parameters(), 'lr': args.pretrain_lr}], \
                            weight_decay=args.pretrain_weight_decay)
    # Set learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.pretrain_step_size, gamma=args.pretrain_gamma)

    # train
    best_iou = 0
    global_iter = 0
    for epoch in range(args.n_iters):
        model.train()
        for batch_idx, (ptclouds, labels) in enumerate(TRAIN_LOADER):
            if torch.cuda.is_available():
                ptclouds = ptclouds.cuda()
                labels = labels.cuda()

            logits = model(ptclouds)
            loss = F.cross_entropy(logits, labels)

            # Loss backwards and optimizer updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0:
                WRITER.add_scalar('Train/loss', loss, global_iter)
                logger.cprint('=====[Train] Epoch: %d | Iter: %d | Loss: %.4f =====' % (epoch, batch_idx, loss.item()))
            global_iter += 1

        lr_scheduler.step()

        if (epoch+1) % args.eval_interval == 0:
            pred_total = []
            gt_total = []
            model.eval()
            with torch.no_grad():
                for i, (ptclouds, labels) in enumerate(VALID_LOADER):
                    gt_total.append(labels.detach())

                    if torch.cuda.is_available():
                        ptclouds = ptclouds.cuda()
                        labels = labels.cuda()

                    logits = model(ptclouds)
                    loss = F.cross_entropy(logits, labels)

                    # 　Compute predictions
                    _, preds = torch.max(logits.detach(), dim=1, keepdim=False)
                    pred_total.append(preds.cpu().detach())

                    WRITER.add_scalar('Valid/loss', loss, global_iter)
                    # logger.cprint(
                    #     '=====[Valid] Epoch: %d | Iter: %d | Loss: %.4f =====' % (epoch, i, loss.item()))

            pred_total = torch.stack(pred_total, dim=0).view(-1, args.pc_npts)
            gt_total = torch.stack(gt_total, dim=0).view(-1, args.pc_npts)
            accuracy, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, NUM_CLASSES)
            logger.cprint('===== EPOCH [%d]: Accuracy: %f | mIoU: %f =====\n' % (epoch, accuracy, mIoU))
            WRITER.add_scalar('Valid/overall_accuracy', accuracy, global_iter)
            WRITER.add_scalar('Valid/meanIoU', mIoU, global_iter)

            if mIoU > best_iou:
                best_iou = mIoU
                logger.cprint('*******************Model Saved*******************')
                save_pretrain_checkpoint(model, args.log_dir)
            logger.cprint('=====Best IoU Is: %f =====' % (best_iou))
    WRITER.close()