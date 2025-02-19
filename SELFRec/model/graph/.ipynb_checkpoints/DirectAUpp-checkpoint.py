import torch
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss,l2_reg_loss
from model.graph.MF import Matrix_Factorization
from model.graph.LightGCNpp import LGCN_Encoder
import os

class DirectAUpp(GraphRecommender):
    def __init__(self, conf, training_set, valid_set, test_set):
        super(DirectAUpp, self).__init__(conf, training_set, valid_set, test_set)
        #args = self.config['DirectAU']
        self.lmbda = float(conf.lmbda)
        self.n_layers= int(conf.n_layer)
        self.alpha = conf.alpha
        self.beta = conf.beta
        self.gamma = conf.gamma
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers, self.alpha, self.beta, self.gamma)

        self.config_name = f'{conf.dataset}_{conf.model_name}_seed{conf.seed}_lr{conf.learning_rate}_reg{conf.reg_lambda}_dim{self.emb_size}_lmbda{self.lmbda}_alpha{self.alpha}_beta{self.beta}_gamma{self.gamma}'
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
                user_emb, pos_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx]
                batch_loss = self.calculate_loss(user_emb, pos_item_emb)+ l2_reg_loss(self.reg, user_emb,pos_item_emb)/self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
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

    def alignment(self,x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self,x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def calculate_loss(self,user_emb,item_emb):
        align = self.alignment(user_emb, item_emb)
        uniform = self.lmbda * (self.uniformity(user_emb) + self.uniformity(item_emb)) / 2
        return align + uniform

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()