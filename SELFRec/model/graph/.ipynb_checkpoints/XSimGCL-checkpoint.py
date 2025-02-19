import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import os
# Paper: XSimGCL - Towards Extremely Simple Graph Contrastive Learning for Recommendation


class XSimGCL(GraphRecommender):
    def __init__(self, conf, training_set, valid_set, test_set):
        super(XSimGCL, self).__init__(conf, training_set, valid_set, test_set)
        self.cl_rate = conf.lmbda
        self.eps = conf.eps
        self.n_layers = conf.n_layer
        self.temp = conf.tau
        self.layer_cl = conf.l_star
        self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.layer_cl)

        self.config_name = f'{conf.dataset}_{conf.model_name}_seed{conf.seed}_lr{conf.learning_rate}_reg{conf.reg_lambda}_dim{self.emb_size}_cl{self.cl_rate}_eps{self.eps}_nl{self.n_layers}_tau{self.temp}_lc{self.layer_cl}'
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
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb  = model(True)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb)
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
            

    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class XSimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl):
        super(XSimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
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
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            if k==self.layer_cl-1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        if perturbed:
            return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings
