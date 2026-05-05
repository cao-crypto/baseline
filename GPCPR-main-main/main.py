# OpenMP/MKL thread environment bootstrap - must be at absolute top
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""Main function for this repo

"""
import ast
import argparse
import platform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #data
    parser.add_argument('--phase', type=str, default='rchpeval', choices=['pretrain', 'finetune',
                                                                          'gpcprtrain', 'gpcpreval',
                                                                          ])
    parser.add_argument('--dataset', type=str, default='s3dis', help='Dataset name: s3dis|scannet')
    parser.add_argument('--cvfold', type=int, default=0, help='Fold left-out for testing in leave-one-out setting '
                                                              'Options:{0,1}')
    parser.add_argument('--data_path', type=str, default='./datasets/S3DIS/blocks_bs1_s1',help='Directory to the source data')
    parser.add_argument('--pretrain_checkpoint_path', type=str, default=None,
                        help='Path to the checkpoint of pre model for resuming')
    parser.add_argument('--model_checkpoint_path', type=str, default=None,
                        help='Path to the checkpoint of model for resuming')
    parser.add_argument('--save_path', type=str, default='./log_s3dis/',
                        help='Directory to the save log and checkpoints')
    parser.add_argument('--eval_interval', type=int, default=1500,
                        help='iteration/epoch inverval to evaluate model')

    #optimization
    parser.add_argument('--batch_size', type=int, default=1, help='Number of samples/tasks in one batch')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of DataLoader workers. Use 0 on Windows for stable debugging; increase only after dataset validation passes.')
    parser.add_argument('--unsafe_allow_duplicate_openmp', action='store_true', default=False, help='UNSAFE: Allow duplicate OpenMP runtime. Only for diagnosing environment problems. Can cause incorrect results or crashes.')
    parser.add_argument('--pin_memory', action='store_true', help='Enable pin_memory for faster GPU transfer')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of batches to prefetch per worker')
    parser.add_argument('--persistent_workers', action='store_true', help='Keep workers alive between epochs')
    parser.add_argument('--debug_data', action='store_true', help='Enable preflight dataset validation')
    parser.add_argument('--data_check_samples', type=int, default=8, help='Number of samples to check during preflight validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--n_iters', type=int, default=40000, help='number of iterations/epochs to train')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Model (eg. protoNet or MPTI) learning rate [default: 0.001]')
    parser.add_argument('--step_size', type=int, default=5000, help='Iterations of learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')

    parser.add_argument('--pretrain_lr', type=float, default=0.001, help='pretrain learning rate [default: 0.001]')
    parser.add_argument('--pretrain_weight_decay', type=float, default=0., help='weight decay for regularization')
    parser.add_argument('--pretrain_step_size', type=int, default=50, help='Period of learning rate decay')
    parser.add_argument('--pretrain_gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')

    #few-shot episode setting
    parser.add_argument('--n_way', type=int, default=2, help='Number of classes for each episode: 1|3')
    parser.add_argument('--k_shot', type=int, default=1, help='Number of samples/shots for each class: 1|5')
    parser.add_argument('--n_queries', type=int, default=1, help='Number of queries for each class')
    parser.add_argument('--n_episode_test', type=int, default=100,
                        help='Number of episode per configuration during testing')

    # Point cloud processing
    parser.add_argument('--pc_npts', type=int, default=2048, help='Number of input points for PointNet.')
    parser.add_argument('--pc_attribs', default='xyzrgbXYZ',
                        help='Point attributes fed to PointNets, if empty then all possible. '
                             'xyz = coordinates, rgb = color, XYZ = normalized xyz')
    parser.add_argument('--pc_augm', action='store_true', help='Training augmentation for points in each superpoint')
    parser.add_argument('--pc_augm_scale', type=float, default=0,
                        help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', type=int, default=1,
                        help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', type=float, default=0,
                        help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', type=int, default=1,
                        help='Training augmentation: Bool, Gaussian jittering of all attributes')
    parser.add_argument('--pc_augm_shift', type=float, default=0,
                        help='Training augmentation: Probability of shifting points')
    parser.add_argument('--pc_augm_color', type=int, default=0,
                        help='Training augmentation: Bool, random color of all attributes')

    # feature extraction network configuration
    parser.add_argument('--dgcnn_k', type=int, default=20, help='Number of nearest neighbors in Edgeconv')
    parser.add_argument('--edgeconv_widths', default='[[64,64], [64,64], [64,64]]', help='DGCNN Edgeconv widths')
    parser.add_argument('--dgcnn_mlp_widths', default='[512, 256]', help='DGCNN MLP (following stacked Edgeconv) widths')
    parser.add_argument('--base_widths', default='[128, 64]', help='BaseLearner widths')
    parser.add_argument('--output_dim', type=int, default=64,
                        help='The dimension of the final output of attention learner or linear mapper')
    parser.add_argument('--use_attention', action='store_true', help='if incorporate attention learner')

    # protoNet configuration
    parser.add_argument('--dist_method', default='euclidean',
                        help='Method to compute distance between query feature maps and prototypes.[Option: cosine|euclidean]')
    # PAPFZS3D configuration
    parser.add_argument('--use_align', action='store_true', help='if incorporate alignment process')
    parser.add_argument('--use_high_dgcnn', action='store_true', help='if incorporate another dgcnn')
    parser.add_argument('--use_supervise_prototype', action='store_true', help='if incorporate self-reconstruction process')
    parser.add_argument('--use_transformer', action='store_true', help='if incorporate transformer process')
    parser.add_argument('--use_linear_proj', action='store_true', help='if incorporate linear projection process')
    parser.add_argument('--trans_lr', type=float, default=0.0001, help='transformer learning rate')
    parser.add_argument('--generator_lr', type=float, default=0.0002, help='generator learning rate')
    parser.add_argument('--noise_dim', type=int, default=300, help='noise dim for generator')
    parser.add_argument('--gmm_dropout', type=float, default=0.1, help='drop out rate for generator')
    parser.add_argument('--gmm_weight', type=float, default=0.1, help='training weight for generator')
    parser.add_argument('--train_dim', type=int, default=320, help='training dim for transformer')

    # GPCPR configuration
    parser.add_argument('--use_text', action='store_true', help='if use rkd loss function')
    parser.add_argument('--use_text_diff', action='store_true', help='if use rkd loss function')
    parser.add_argument('--embedding_type', type=str, default='gpt35', help='semantic input')
    parser.add_argument('--embedding_num', type=int, default=10, help='semantic input')
    parser.add_argument('--use_pcpr', action='store_true', help='if use rkd loss function')
    parser.add_argument('--use_dd_loss', action='store_true', help='if use dual distillation loss function')
    parser.add_argument('--dd_ratio1', type=float, default=1, help='if use rkd loss function')
    parser.add_argument('--dd_ratio2', type=float, default=2, help='if use rkd loss function')

    args = parser.parse_args()

    # Handle unsafe OpenMP option
    if args.unsafe_allow_duplicate_openmp:
        print('\n*** UNSAFE MODE ENABLED ***')
        print('*** KMP_DUPLICATE_LIB_OK=TRUE can cause incorrect results or crashes ***')
        print('*** Only use this for diagnosing environment problems ***\n')
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    args.edgeconv_widths = ast.literal_eval(args.edgeconv_widths)
    args.dgcnn_mlp_widths = ast.literal_eval(args.dgcnn_mlp_widths)
    args.base_widths = ast.literal_eval(args.base_widths)
    args.pc_in_dim = len(args.pc_attribs)


    if args.phase == 'gpcprtrain':
        args.log_dir = args.save_path + 'S%d_N%d_K%d_Att%d' % (args.cvfold, args.n_way, args.k_shot, args.use_attention)
        from runs.gpcpr_train import train
        train(args)
    elif args.phase in ['gpcpreval']:
        args.log_dir = args.model_checkpoint_path
        from runs.eval import eval
        eval(args)
    elif args.phase == 'pretrain':
        # args.log_dir = args.save_path + 'log_pretrain_%s_S%d' % (args.dataset, args.cvfold)
        args.log_dir = args.save_path + 'pretrain_S%d' % (args.cvfold)
        from runs.pre_train import pretrain
        pretrain(args)
    elif args.phase == 'finetune':
        args.log_dir = args.save_path + 'log_finetune_%s_S%d_N%d_K%d' % (args.dataset, args.cvfold,
                                                                            args.n_way, args.k_shot)
        from runs.fine_tune import finetune
        finetune(args)
    else:
        raise ValueError('Please set correct phase.')