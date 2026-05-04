""" ProtoNet with/without attention learner for Few-shot 3D Point Cloud Semantic Segmentation


"""
import torch
from torch import optim
from torch.nn import functional as F

from models.gpcpr_model import GPCPR
from utils.checkpoint_util import load_pretrain_checkpoint, load_model_checkpoint
import numpy as np
import random
import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table

class GPCPRLearner(object):

    def __init__(self, args, mode='train'):

        self.model = GPCPR(args)
        print(self.model)
        if torch.cuda.is_available():
            self.model.cuda()


        if mode == 'train':
            params_dict = [{'params': self.model.encoder.parameters(), 'lr': 0.0001},
                           {'params': self.model.base_learner.parameters()},
                           {'params': self.model.att_learner.parameters()}]
            if args.use_transformer:
                params_dict.append({'params': self.model.transformer.parameters(), 'lr': args.trans_lr})
            if args.use_text or args.use_text_diff:
                params_dict.append({'params': self.model.text_projector.parameters(), 'lr': args.generator_lr})
            if args.use_text:
                params_dict.append({'params': self.model.text_compressor.parameters(), 'lr': args.trans_lr})
            if args.use_text_diff:
                params_dict.append({'params': self.model.text_compressor_diff.parameters(), 'lr': args.trans_lr})
            if args.use_pcpr:
                params_dict.append({'params': self.model.proto_compressor.parameters(), 'lr': args.trans_lr})
            self.optimizer = torch.optim.Adam(params_dict, lr=args.lr)
            # set learning rate scheduler
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size,gamma=args.gamma)
            # load pretrained model for point cloud encoding
            if args.pretrain_checkpoint_path:
                self.model = load_pretrain_checkpoint(self.model, args.pretrain_checkpoint_path)
            # print("#Model parameters: {}".format(sum([x.nelement() for x in self.model.parameters()])))

        elif mode == 'test':
            # Load model checkpoint
            self.model = load_model_checkpoint(self.model, args.model_checkpoint_path, mode='test')
            # print("#Model parameters: {}".format(sum([x.nelement() for x in self.model.parameters()])))
        else:
            raise ValueError('Wrong GMMLearner mode (%s)! Option:train/test' %mode)

        self.n_way = args.n_way

        # GPCPR add --- Load Text
        self.use_text = args.use_text
        self.use_text_diff = args.use_text_diff
        print("load {} {}".format(args.dataset, args.embedding_type))
        data_embedding_type = {'word2vec': 'glove', 'clip': 'clip_rn50', 'gpt35': 'gpt-3.5-turbo', 'gpt4omini': 'gpt-4o-mini'}
        vec_name = data_embedding_type[args.embedding_type]
        dataName = {'s3dis': 'S3DIS', 'scannet': 'ScanNet'}
        data_bg_ids = {'s3dis': 12, 'scannet': 0}
        self.bg_id = data_bg_ids[args.dataset]
        self.embedding_num = args.embedding_num
        if self.embedding_num==0:
            self.use_text=False
        if self.use_text:
            if args.embedding_type == 'word2vec' or args.embedding_type == 'clip':
                self.embeddings = torch.from_numpy(np.load('dataloaders/{}_{}.npy'.format(dataName[args.dataset], vec_name))).unsqueeze(1)
            elif args.embedding_type in ['gpt35','gpt4omini']:
                print('load text:', 'gpt_prompts/{}_{}_{}.pth'.format(args.dataset, args.embedding_num, vec_name))
                loaded = torch.load('gpt_prompts/{}_{}_{}.pth'.format(args.dataset, args.embedding_num, vec_name),
                                    map_location='cpu')
                if isinstance(loaded, dict):
                    self.embeddings = torch.stack(list(loaded.values()), dim=0).float()
                elif isinstance(loaded, torch.Tensor):
                    self.embeddings = loaded.float()
                else:
                    raise ValueError(f"Unexpected type for loaded object: {type(loaded)}")
            else:
                print('!!! input wrong text embedding_type!!!')
            if args.embedding_type in ['clip','gpt35','gpt4omini']:
                self.embeddings = self.embeddings.float()
            self.embeddings = torch.nn.functional.normalize(self.embeddings, p=2, dim=-1)


        if self.use_text_diff:
            print('load text_diff:','gpt_prompts/{}_visual_geometry_difference2_{}.pth'.format(args.dataset, vec_name))
            self.embeddings_diff = torch.load('gpt_prompts/{}_visual_geometry_difference2_{}.pth'.format(args.dataset, vec_name),map_location='cpu')

    def train(self, data, sampled_classes):
        """
        Args:
            data: a list of torch tensors wit the following entries.
            - support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            - support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            - query_x: query point clouds with shape (n_queries, in_channels, num_points)
            - query_y: query labels with shape (n_queries, num_points)
        """

        # load 3D data
        [support_x, support_y, query_x, query_y] = data
        # load text_data
        support_text_embeddings = None
        if self.use_text:
            support_text_embeddings = torch.cat([self.embeddings[self.bg_id].unsqueeze(0), self.embeddings[sampled_classes]],dim=0).cuda()
        support_text_embeddings_diff = None
        if self.use_text_diff:
            support_text_embeddings_diff = self.extract_diff_text(self.embeddings_diff, self.bg_id, sampled_classes).cuda()

        self.model.train()
        query_logits, loss = self.model(support_x, support_y, query_x, query_y,support_text_embeddings,support_text_embeddings_diff)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        query_pred = F.softmax(query_logits, dim=1).argmax(dim=1)
        correct = torch.eq(query_pred, query_y).sum().item()  # including background class
        accuracy = correct / (query_y.shape[0] * query_y.shape[1])
        return loss, accuracy

    # for GPCPR
    def test(self, data, sampled_classes):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points), each point \in {0,1}.
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        """
        [support_x, support_y, query_x, query_y] = data
        # load text_data
        support_text_embeddings = None
        if self.use_text:
            support_text_embeddings = torch.cat([self.embeddings[self.bg_id].unsqueeze(0), self.embeddings[sampled_classes]], dim=0).cuda()
        support_text_embeddings_diff = None
        if self.use_text_diff:
            support_text_embeddings_diff = self.extract_diff_text(self.embeddings_diff, self.bg_id,sampled_classes).cuda()

        self.model.eval()

        with torch.no_grad():
            query_logits, loss = self.model(support_x, support_y, query_x, query_y, support_text_embeddings, support_text_embeddings_diff)

            query_pred = F.softmax(query_logits, dim=1).argmax(dim=1)
            correct = torch.eq(query_pred, query_y).sum().item()
            accuracy = correct / (query_y.shape[0] * query_y.shape[1])
        return query_pred, loss, accuracy


    def extract_diff_text(self, embeddings, bg_id, sampled_classes):
        idxs = [bg_id] + list(sampled_classes)
        names = list(embeddings.keys())
        out = []
        for cls in idxs:
            out.append(torch.cat([embeddings[names[j]][names[cls]] for j in idxs], dim=0))
        # min version
        min_len = min([iii.shape[0] for iii in out])
        for i in range(len(out)):
            text_len = out[i].shape[0]
            if text_len>min_len:
                index = torch.LongTensor(random.sample(range(text_len), min_len))
                out[i] = torch.index_select(out[i], 0, index)
        out = torch.stack(out, dim=0).float()
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out
