import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import pickle as pkl
import os

# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22


class OurGCL(GraphRecommender):
    def __init__(self, conf, training_set, valid_set, test_set):
        super(OurGCL, self).__init__(conf, training_set, valid_set, test_set)
        self.cl_rate = conf.lmbda
        self.cl_rate2 = conf.lmbda2
        self.eps = conf.eps
        self.gamma = conf.gamma
        self.n_layers = conf.n_layer
        self.norm = conf.norm
        self.neg_agg = conf.neg_agg
        self.model = SimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)
        
        self.config_name = f'{conf.dataset}_{conf.model_name}_lr{conf.learning_rate}_reg{conf.reg_lambda}_dim{self.emb_size}_cl1-{self.cl_rate}_cl2-{self.cl_rate2}_nl{self.n_layers}_gamma{self.gamma}_norm{self.norm}_neg-{self.neg_agg}'
        print()
        print(self.config_name)

        if os.path.exists(f'logs/{self.config_name}.txt'):
            print('Exists.')
            exit(0)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        best_valid, patience, wait_cnt = -1e10, 10, 0
        
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss_1, cl_loss_2 = self.cal_cl_loss(user_idx, pos_idx)
                cl_loss_1 = self.cl_rate * cl_loss_1
                cl_loss_2 = self.cl_rate2 * cl_loss_2
                cl_loss = cl_loss_1 + cl_loss_2
                
                
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
                    
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            #self.fast_evaluation(epoch)
            self.evaluate(self.test('valid'), 'valid')
            result_valid = [r[:-1] for r in self.result]
            self.evaluate(self.test('test'), 'test')
            result_test = [r[:-1] for r in self.result]
            
            for _i in range(3):
                print('Valid\t', result_valid[_i*5], result_valid[_i*5+3], result_valid[_i*5+4])
            for _i in range(3):
                print('Test\t', result_test[_i*5], result_test[_i*5+3], result_test[_i*5+4])
                
            with open(f'logs/{self.config_name}.txt', 'a') as f:
                valid_log, test_log = '', ''
                for _i in range(3):
                    recall = result_valid[_i*5+3].split(':')[1]
                    ndcg = result_valid[_i*5+4].split(':')[1]
                    valid_log += f',{recall},{ndcg}'
                    
                    recall = result_test[_i*5+3].split(':')[1]
                    ndcg = result_test[_i*5+4].split(':')[1]
                    test_log += f',{recall},{ndcg}'
                f.write(f'{epoch+1},valid,{valid_log}\n')
                f.write(f'{epoch+1},test,{test_log}\n')
                
            ndcg_valid = float(result_valid[9].split(':')[1])
            if ndcg_valid > best_valid:
                best_valid = ndcg_valid
                self.best_user_emb = self.model.embedding_dict['user_emb'].detach().cpu()
                self.best_item_emb = self.model.embedding_dict['item_emb'].detach().cpu()
                wait_cnt = 0
            else:
                wait_cnt += 1
                print(f'Patience... {wait_cnt}/{patience}')
                
            if wait_cnt == patience:
                print('Early Stopping!')
                break
            print()
            

    def cal_cl_loss(self, u_idx, i_idx):
        B = len(u_idx)
        
        if self.norm:
            user_emb = F.normalize(self.model.embedding_dict['user_emb'][u_idx], dim=1)
            item_emb = F.normalize(self.model.embedding_dict['item_emb'][i_idx], dim=1)
        else:
            user_emb = self.model.embedding_dict['user_emb'][u_idx]
            item_emb = self.model.embedding_dict['item_emb'][i_idx]
        
        score_ui = user_emb @ item_emb.T
        score_uu = user_emb @ user_emb.T
        score_ii = item_emb @ item_emb.T

        diag_ui = score_ui.diagonal()
        if self.neg_agg == 'max':
            off_diag_ui = torch.max(score_ui[~torch.eye(B, dtype=torch.bool).cuda()].view(B, B - 1), dim=1)[0]
        elif self.neg_agg == 'mean':
            off_diag_ui = torch.mean(score_ui[~torch.eye(B, dtype=torch.bool).cuda()].view(B, B - 1), dim=1)
        elif self.neg_agg == 'all':
            diag_ui = diag_ui.unsqueeze(1)
            off_diag_ui = score_ui[~torch.eye(B, dtype=torch.bool).cuda()].view(B, B - 1)
        
        diag_uu = score_uu.diagonal()
        if self.neg_agg == 'max':
            off_diag_uu = torch.max(score_uu[~torch.eye(B, dtype=torch.bool).cuda()].view(B, B - 1), dim=1)[0]
        elif self.neg_agg == 'mean':
            off_diag_uu = torch.mean(score_uu[~torch.eye(B, dtype=torch.bool).cuda()].view(B, B - 1), dim=1)
        elif self.neg_agg == 'all':
            diag_uu = diag_uu.unsqueeze(1)
            off_diag_uu = score_uu[~torch.eye(B, dtype=torch.bool).cuda()].view(B, B - 1)

        diag_ii = score_ii.diagonal()
        if self.neg_agg == 'max':
            off_diag_ii = torch.max(score_ii[~torch.eye(B, dtype=torch.bool).cuda()].view(B, B - 1), dim=1)[0]
        elif self.neg_agg == 'mean':
            off_diag_ii = torch.mean(score_ii[~torch.eye(B, dtype=torch.bool).cuda()].view(B, B - 1), dim=1)
        elif self.neg_agg == 'all':
            diag_ii = diag_ii.unsqueeze(1)
            off_diag_ii = score_ii[~torch.eye(B, dtype=torch.bool).cuda()].view(B, B - 1)
        
        loss_ui = torch.relu(off_diag_ui - diag_ui + self.gamma).mean()
        loss_uu = torch.relu(off_diag_uu - diag_uu + self.gamma).mean()
        loss_ii = torch.relu(off_diag_ii - diag_ii + self.gamma).mean()
        
        return 0.5 * (loss_uu + loss_ii), loss_ui
    
    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(SimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings
