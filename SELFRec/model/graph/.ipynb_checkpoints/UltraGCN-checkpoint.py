import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import numpy as np

from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise  # or a custom sampler
from base.torch_interface import TorchGraphInterface
from util.loss_torch import l2_reg_loss

##########################################################
# UltraGCN (Skeleton in the style of LightGCN's code)
##########################################################

class UltraGCN(GraphRecommender):
    """
    An example UltraGCN recommender in a style similar to the LightGCN code.
    Follows the same structure with train, predict, etc.
    """
    def __init__(self, conf, training_set, valid_set, test_set):
        super(UltraGCN, self).__init__(conf, training_set, valid_set, test_set)
        
        # Hyperparameters from conf (example; adapt as needed):
        self.gamma = conf.reg_lambda             # L2 regularization weight
        self.lambda_ = 0.1   # item-item co-occ regularization
        self.w1 = 0.1
        self.w2 = 0.1
        self.w3 = 0.1
        self.w4 = 0.1
        self.neg_weight = 1.0   # e.g., negative weight factor
        self.emb_size = self.emb_size
        
        # UltraGCN uses no GCN layers in the standard sense, 
        # but we can store #layers or other config if you want:
        self.n_layers = getattr(conf, 'n_layer', 0)
        
        # Model instantiation
        self.model = UltraGCN_Encoder(self.data, self.emb_size, self.w1, self.w2, 
                                      self.w3, self.w4, self.neg_weight, 
                                      self.gamma, self.lambda_)
        
        # A name for checkpointing / logging:
        self.config_name = f'{conf.dataset}_{conf.model_name}_seed{conf.seed}_lr{conf.learning_rate}_reg{conf.reg_lambda}_dim{self.emb_size}'
        print('\n', self.config_name)
        
        self.save = conf.save  # if we want to store embeddings to disk
        
    def train(self):
        """
        Training loop in the same format as LightGCN.
        """
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        best_valid, patience, wait_cnt = -1e10, 10, 0
        
        for epoch in range(self.maxEpoch):
            # --- Training ---
            model.train()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                # batch should yield (user_idx, pos_idx, neg_idx)
                user_idx, pos_idx, neg_idx = batch
                user_idx = torch.from_numpy(user_idx).long().cuda()
                pos_idx = torch.from_numpy(pos_idx).long().cuda()
                neg_idx = torch.from_numpy(neg_idx).long().cuda()
                
                # Forward pass: get current user/item embeddings
                user_emb, item_emb = model()
                
                # UltraGCN custom loss
                batch_loss = model.ultragcn_loss(
                    user_emb, item_emb, user_idx, pos_idx, neg_idx
                )

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                if n % 100 == 0 and n > 0:
                    print(f'[Epoch {epoch+1}, Batch {n}] batch_loss = {batch_loss.item():.6f}')
            
            # --- After an epoch, store final embeddings for inference ---
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            
            # --- Evaluation: 'valid' and 'test' sets ---
            self.evaluate(self.test('valid'), 'valid')
            result_valid = [r[:-1] for r in self.result]  # example usage
            self.evaluate(self.test('test'), 'test')
            result_test  = [r[:-1] for r in self.result]
            
            # Print logs for top-k metrics
            for i in range(3):
                print('Valid\t', result_valid[i*5], result_valid[i*5+3], result_valid[i*5+4])
            for i in range(3):
                print('Test\t',  result_test[i*5],  result_test[i*5+3],  result_test[i*5+4])
            
            # Save logs
            with open(f'logs/{self.config_name}.txt', 'a') as f:
                valid_log, test_log = '', ''
                for i in range(3):
                    recall = result_valid[i*5+3].split(':')[1]
                    ndcg   = result_valid[i*5+4].split(':')[1]
                    valid_log += f',{recall},{ndcg}'

                    recall = result_test[i*5+3].split(':')[1]
                    ndcg   = result_test[i*5+4].split(':')[1]
                    test_log += f',{recall},{ndcg}'
                f.write(f'{epoch+1},valid,{valid_log}\n')
                f.write(f'{epoch+1},test,{test_log}\n')
            
            # --- Early stopping based on the NDCG in 'valid' set (example) ---
            ndcg_valid = float(result_valid[9].split(':')[1])  # e.g. use top20 ndcg
            if ndcg_valid > best_valid:
                best_valid = ndcg_valid
                wait_cnt   = 0
                # Save best embeddings
                if self.save:
                    # detach and save
                    user_all_emb = self.user_emb.detach().cpu()
                    item_all_emb = self.item_emb.detach().cpu()
                    with open(f'embs/{self.config_name}.pkl', 'wb') as f:
                        pkl.dump([user_all_emb, item_all_emb, self.data.user, self.data.item], f)
            else:
                wait_cnt += 1
                print(f'Patience... {wait_cnt}/{patience}')
                if wait_cnt == patience:
                    print('Early Stopping!')
                    break
            print()

    def predict(self, u):
        """
        Inference: score all items for user u
        """
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class UltraGCN_Encoder(nn.Module):
    """
    UltraGCN core module:
    - Maintains user/item embeddings as trainable parameters.
    - Uses weighting factors w1, w2, w3, w4 for positive/negative examples.
    - Incorporates item–item co-occurrence constraints if available in `data`.
    """
    def __init__(self, data, emb_size, w1, w2, w3, w4, neg_weight, gamma, lambda_):
        super(UltraGCN_Encoder, self).__init__()
        self.data = data
        self.user_num = data.user_num
        self.item_num = data.item_num
        self.emb_size = emb_size
        
        # UltraGCN hyperparams
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.neg_weight = neg_weight
        self.gamma = gamma         # L2 regularization
        self.lambda_ = lambda_     # item-item co-occ factor

        # Precomputed constraints if stored in data:
        # data.constraint_mat => { 'beta_uD': tensor, 'beta_iD': tensor }
        # data.ii_neighbor_mat => item -> top neighbors
        # data.ii_constraint_mat => item -> constraints for neighbors
        self.beta_uD = data.constraint_mat['beta_uD'] if 'beta_uD' in data.constraint_mat else None
        self.beta_iD = data.constraint_mat['beta_iD'] if 'beta_iD' in data.constraint_mat else None
        self.ii_neighbor_mat = getattr(data, 'ii_neighbor_mat', None)
        self.ii_constraint_mat = getattr(data, 'ii_constraint_mat', None)
        
        # Create the embedding matrix
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(torch.randn(self.user_num, self.emb_size) * 0.01),
            'item_emb': nn.Parameter(torch.randn(self.item_num, self.emb_size) * 0.01),
        })

    def forward(self):
        """
        Returns user/item embeddings for the entire set.
        (We keep it simple: no multi-layer propagation like LightGCN.)
        """
        user_emb = self.embedding_dict['user_emb']
        item_emb = self.embedding_dict['item_emb']
        return user_emb, item_emb

    def ultragcn_loss(self, user_emb, item_emb, user_idx, pos_idx, neg_idx):
        """
        Compute the UltraGCN training loss for a mini-batch:
          L = Weighted logistic loss + gamma * L2 + lambda * item–item constraint
        """
        # 1) Weighted logistic loss
        #    We'll build positive and negative logit scores:
        u = user_emb[user_idx]             # (batch, dim)
        pos_i = item_emb[pos_idx]          # (batch, dim)
        neg_i = item_emb[neg_idx]          # (batch, dim)
        
        # scores
        pos_scores = (u * pos_i).sum(dim=-1)   # (batch,)
        neg_scores = (u * neg_i).sum(dim=-1)   # (batch,)
        
        # compute weights for positives & negatives
        omega_pos, omega_neg = self._get_omegas(user_idx, pos_idx, neg_idx)
        # logistic loss
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)
        
        # BCE with weighting
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight=omega_pos, reduction='none')
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight=omega_neg, reduction='none')
        
        # sum (or mean) them, then weight negative part if needed
        logistic_loss = pos_loss.sum() + self.neg_weight * neg_loss.sum()
        
        # 2) L2 reg
        l2_loss = (user_emb**2).sum() + (item_emb**2).sum()
        l2_loss = 0.5 * self.gamma * l2_loss
        
        # 3) item–item co-occ term (optional)
        #    For each pos_idx, we consider its neighbors from ii_neighbor_mat
        #    Weighted by ii_constraint_mat for those neighbors
        item_item_loss = 0.0
        if self.lambda_ > 0 and (self.ii_neighbor_mat is not None) and (self.ii_constraint_mat is not None):
            # shape: (batch, num_neighbors, dim)
            neighbors = self.ii_neighbor_mat[pos_idx]  # top neighbors for each pos
            neighbors = neighbors.cuda() if neighbors.is_cuda == False else neighbors
            neighbor_embeds = item_emb[neighbors]      # (batch, n_neighbors, dim)

            sim_scores = self.ii_constraint_mat[pos_idx]  # (batch, n_neighbors)
            sim_scores = sim_scores.cuda() if sim_scores.is_cuda == False else sim_scores
            
            # For each (user, pos_item), we do - \sum neighbor \log(sigmoid(u dot neighbor_i)) * sim_scores
            # Here we can broadcast multiply:
            u_expand = u.unsqueeze(1)  # (batch, 1, dim)
            # dot product => (batch, n_neighbors)
            dot_ui = (u_expand * neighbor_embeds).sum(dim=-1)
            item_item_loss = -(sim_scores * F.logsigmoid(dot_ui)).sum()
            item_item_loss *= self.lambda_
        
        return logistic_loss + l2_loss + item_item_loss

    def _get_omegas(self, user_idx, pos_idx, neg_idx):
        """
        Compute sample weights for positive & negative pairs, e.g.:
          w_pos = w1 + w2 * (beta_uD[u] * beta_iD[i])
          w_neg = w3 + w4 * (beta_uD[u] * beta_iD[i])
        """
        device = user_idx.device
        batch_size = user_idx.size(0)
        
        if self.beta_uD is None or self.beta_iD is None:
            # fallback: constant weights
            return torch.ones(batch_size, device=device)*self.w1, \
                   torch.ones(batch_size, device=device)*self.w3
        
        beta_u = self.beta_uD[user_idx].to(device)  # (batch,)
        beta_pos_i = self.beta_iD[pos_idx].to(device)  # (batch,)
        beta_neg_i = self.beta_iD[neg_idx].to(device)  # (batch,)

        # compute w_pos
        w_pos = self.w1 + self.w2 * (beta_u * beta_pos_i)
        # compute w_neg
        w_neg = self.w3 + self.w4 * (beta_u * beta_neg_i)
        
        return w_pos, w_neg
