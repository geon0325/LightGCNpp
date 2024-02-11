import os
import world
import utils
from world import cprint
import torch
import numpy as np
import pickle as pkl
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset




if not os.path.exists('logs'):
    os.mkdir('logs')
    
if not os.path.exists('embs'):
    os.mkdir('embs')

config = f'{world.args.dataset}_seed{world.args.seed}_{world.args.model}_dim{world.args.recdim}_lr{world.args.lr}_dec{world.args.decay}_alpha{world.args.alpha}_beta{world.args.beta}_gamma{world.args.gamma}'

if world.args.model == 'lgn':
    config += f'_nl{world.args.layer}'

log_path = f'logs/{config}.txt'
emb_path = f'embs/{config}.pkl'

if os.path.exists(emb_path):
    print('Exists.')
    exit(0)

if world.args.decay == 1e-6 or world.args.decay == 1e-8:
    exit(0)
    
if world.args.layer == 4:
    exit(0)
    
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment))
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    best_valid = -1
    patience = 0
    
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        
        if (epoch + 1) % 5 == 0:
            cprint("[VALIDATION]")
            valid_results = Procedure.Valid(dataset, Recmodel, epoch, w, world.config['multicore'])
            valid_log = [valid_results['ndcg'][0], valid_results['ndcg'][1], valid_results['recall'][0], valid_results['recall'][1], valid_results['precision'][0], valid_results['precision'][1]]
            
            cprint("[TEST]")
            test_results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            test_log = [test_results['ndcg'][0], test_results['ndcg'][1], test_results['recall'][0], test_results['recall'][1], test_results['precision'][0], test_results['precision'][1]]
            
            with open(log_path, 'a') as f:
                f.write(f'valid ' + ' '.join([str(x) for x in valid_log]) + '\n')
                f.write(f'test ' + ' '.join([str(x) for x in test_log]) + '\n')
            
            if valid_results['ndcg'][0] > best_valid:
                best_valid = valid_results['ndcg'][0]
                patience = 0
                
                Recmodel.eval()
                all_users, all_items, _all_users, _all_items = Recmodel.computer()
                all_users, all_items = all_users.detach().cpu(), all_items.detach().cpu()
                _all_users, _all_items = _all_users.detach().cpu(), _all_items.detach().cpu()

                with open(emb_path, 'wb') as f:
                    if world.args.save_layer_emb:
                        pkl.dump([all_users, all_items, _all_users, _all_items], f)
                    else:
                        pkl.dump([all_users, all_items], f)
            else:
                patience += 1

        if patience == 10:
            print('Early Stopping')
            
            with open(log_path, 'a') as f:
                f.write('Early Stopping\n')
                
            exit(0)
finally:
    if world.tensorboard:
        w.close()