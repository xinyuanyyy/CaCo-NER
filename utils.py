import torch.nn.functional as F
import torch
import random
import numpy as np
from fastNLP import Const
from fastNLP import CrossEntropyLoss
from fastNLP import AccuracyMetric
from fastNLP import Tester
import os
from fastNLP import logger
import torch.nn as nn

class MyDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        assert 0<=p<=1
        self.p = p

    def forward(self, x):
        if self.training and self.p>0.001:
            # print('mydropout!')
            mask = torch.rand(x.size())
            # print(mask.device)
            mask = mask.to(x) # to cuda 0 :as x does
            # print(mask.device)
            mask = mask.lt(self.p) # 是否小于定值p self.p==0.5
            x = x.masked_fill(mask, 0)/(1-self.p)
        return x


def should_mask(name, t=''):
    if 'bias' in name:
        return False
    if 'embedding' in name:
        splited = name.split('.')
        if splited[-1]!='weight':
            return False
        if 'embedding' in splited[-2]:
            return False
    if 'c0' in name:
        return False
    if 'h0' in name:
        return False

    if 'output' in name and t not in name:
        return False

    return True
def get_init_mask(model):
    init_masks = {}
    for name, param in model.named_parameters():
        if should_mask(name):
            init_masks[name+'.mask'] = torch.ones_like(param)
            # logger.info(init_masks[name+'.mask'].requires_grad)

    return init_masks

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed+100)
    torch.manual_seed(seed+200)
    torch.cuda.manual_seed_all(seed+300)

def get_parameters_size(model):
    result = {}
    for name,p in model.state_dict().items():
        result[name] = p.size()

    return result

def prune_by_proportion_model(model,proportion,task):
    # print('this time prune to ',proportion*100,'%')
    for name, p in model.named_parameters():
        # print(name)
        if not should_mask(name,task):
            continue

        tensor = p.data.cpu().numpy()
        index = np.nonzero(model.mask[task][name+'.mask'].data.cpu().numpy())
        # print(name,'alive count',len(index[0]))
        alive = tensor[index]
        # print('p and mask size:',p.size(),print(model.mask[task][name+'.mask'].size()))
        percentile_value = np.percentile(abs(alive), (1 - proportion) * 100)
        # tensor = p
        # index = torch.nonzero(model.mask[task][name+'.mask'])
        # # print('nonzero len',index)
        # alive = tensor[index]
        # print('alive size:',alive.shape)
        # prune_by_proportion_model()

        # percentile_value = torch.topk(abs(alive), int((1-proportion)*len(index[0]))).values
        # print('the',(1-proportion)*len(index[0]),'th big')
        # print('threshold:',percentile_value)

        prune_by_threshold_parameter(p, model.mask[task][name+'.mask'],percentile_value)
        # for

def prune_by_proportion_model_global(model,proportion,task):
    # print('this time prune to ',proportion*100,'%')
    alive = None
    for name, p in model.named_parameters():
        # print(name)
        if not should_mask(name,task):
            continue

        tensor = p.data.cpu().numpy()
        index = np.nonzero(model.mask[task][name+'.mask'].data.cpu().numpy())
        # print(name,'alive count',len(index[0]))
        if alive is None:
            alive = tensor[index]
        else:
            alive = np.concatenate([alive,tensor[index]],axis=0)

    percentile_value = np.percentile(abs(alive), (1 - proportion) * 100)

    for name, p in model.named_parameters():
        if should_mask(name,task):
            prune_by_threshold_parameter(p, model.mask[task][name+'.mask'],percentile_value)


def prune_by_threshold_parameter(p, mask, threshold):
    p_abs = torch.abs(p)

    new_mask = (p_abs > threshold).float()
    # print(mask)
    mask[:]*=new_mask


def one_time_train_and_prune_single_task(trainer,PRUNE_PER,
                                         optimizer_init_state_dict=None,
                                         model_init_state_dict=None,
                                         is_global=None,
                                         ):


    from fastNLP import Trainer


    trainer.optimizer.load_state_dict(optimizer_init_state_dict)
    trainer.model.load_state_dict(model_init_state_dict)
    # print('metrics:',metrics.__dict__)
    # print('loss:',loss.__dict__)
    # print('trainer input:',task.train_set.get_input_name())
    # trainer = Trainer(model=model, train_data=task.train_set, dev_data=task.dev_set, loss=loss, metrics=metrics,
    #                   optimizer=optimizer, n_epochs=EPOCH, batch_size=BATCH, device=device,callbacks=callbacks)


    trainer.train(load_best_model=True)
    # tester = Tester(task.train_set, model, metrics, BATCH, device=device, verbose=1,use_tqdm=False)
    # print('FOR DEBUG: test train_set:',tester.test())
    # print('**'*20)
    # if task.test_set:
    #     tester = Tester(task.test_set, model, metrics, BATCH, device=device, verbose=1)
    #     tester.test()
    if is_global:

        prune_by_proportion_model_global(trainer.model, PRUNE_PER, trainer.model.now_task)

    else:
        prune_by_proportion_model(trainer.model, PRUNE_PER, trainer.model.now_task)



# def iterative_train_and_prune_single_task(get_trainer,ITER,PRUNE,is_global=False,save_path=None):
def iterative_train_and_prune_single_task(get_trainer,args,model,train_set,dev_set,test_set,device,save_path=None):

    '''

    :param trainer:
    :param ITER:
    :param PRUNE:
    :param is_global:
    :param save_path: should be a dictionary which will be filled with mask and state dict
    :return:
    '''



    from fastNLP import Trainer
    import torch
    import math
    import copy
    PRUNE = args.prune
    ITER = args.iter
    trainer = get_trainer(args,model,train_set,dev_set,test_set,device)
    optimizer_init_state_dict = copy.deepcopy(trainer.optimizer.state_dict())
    model_init_state_dict = copy.deepcopy(trainer.model.state_dict())
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # if not os.path.exists(os.path.join(save_path, 'model_init.pkl')):
        #     f = open(os.path.join(save_path, 'model_init.pkl'), 'wb')
        #     torch.save(trainer.model.state_dict(),f)


    mask_count = 0
    model = trainer.model
    task = trainer.model.now_task
    for name, p in model.mask[task].items():
        mask_count += torch.sum(p).item()
    init_mask_count = mask_count
    logger.info('init mask count:{}'.format(mask_count))
    # logger.info('{}th traning mask count: {} / {} = {}%'.format(i, mask_count, init_mask_count,
    #                                                             mask_count / init_mask_count * 100))

    prune_per_iter = math.pow(PRUNE, 1 / ITER)


    for i in range(ITER):
        trainer = get_trainer(args,model,train_set,dev_set,test_set,device)
        one_time_train_and_prune_single_task(trainer,prune_per_iter,optimizer_init_state_dict,model_init_state_dict)
        if save_path is not None:
            f = open(os.path.join(save_path,task+'_mask_'+str(i)+'.pkl'),'wb')
            torch.save(model.mask[task],f)

        mask_count = 0
        for name, p in model.mask[task].items():
            mask_count += torch.sum(p).item()
        logger.info('{}th traning mask count: {} / {} = {}%'.format(i,mask_count,init_mask_count,mask_count/init_mask_count*100))


def get_appropriate_cuda(task_scale='s'):
    if task_scale not in {'s','m','l'}:
        logger.info('task scale wrong!')
        exit(2)
    import pynvml
    pynvml.nvmlInit()
    total_cuda_num = pynvml.nvmlDeviceGetCount()
    for i in range(total_cuda_num):
        logger.info(i)
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 这里的0是GPU id
        memInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilizationInfo = pynvml.nvmlDeviceGetUtilizationRates(handle)
        logger.info(i, 'mem:', memInfo.used / memInfo.total, 'util:',utilizationInfo.gpu)
        if memInfo.used / memInfo.total < 0.15 and utilizationInfo.gpu <0.2:
            logger.info(i,memInfo.used / memInfo.total)
            return 'cuda:'+str(i)

    if task_scale=='s':
        max_memory=2000
    elif task_scale=='m':
        max_memory=6000
    else:
        max_memory = 9000

    max_id = -1
    for i in range(total_cuda_num):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
        memInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilizationInfo = pynvml.nvmlDeviceGetUtilizationRates(handle)
        if max_memory < memInfo.free:
            max_memory = memInfo.free
            max_id = i

    if id == -1:
        logger.info('no appropriate gpu, wait!')
        exit(2)

    return 'cuda:'+str(max_id)

        # if memInfo.used / memInfo.total < 0.5:
        #     return

def print_mask(mask_dict):
    def seq_mul(*X):
        res = 1
        for x in X:
            res*=x
        return res

    for name,p in mask_dict.items():
        total_size = seq_mul(*p.size())
        unmasked_size = len(np.nonzero(p))

        print(name,':',unmasked_size,'/',total_size,'=',unmasked_size/total_size*100,'%')


    print()


def check_words_same(dataset_1,dataset_2,field_1,field_2):
    if len(dataset_1[field_1]) != len(dataset_2[field_2]):
        logger.info('CHECK: example num not same!')
        return False

    for i, words in enumerate(dataset_1[field_1]):
        if len(dataset_1[field_1][i]) != len(dataset_2[field_2][i]):
            logger.info('CHECK {} th example length not same'.format(i))
            logger.info('1:{}'.format(dataset_1[field_1][i]))
            logger.info('2:'.format(dataset_2[field_2][i]))
            return False

        # for j,w in enumerate(words):
        #     if dataset_1[field_1][i][j] != dataset_2[field_2][i][j]:
        #         print('CHECK', i, 'th example has words different!')
        #         print('1:',dataset_1[field_1][i])
        #         print('2:',dataset_2[field_2][i])
        #         return False

    logger.info('CHECK: totally same!')

    return True

def get_now_time():
    import time
    from datetime import datetime, timezone, timedelta
    dt = datetime.utcnow()
    # print(dt)
    tzutc_8 = timezone(timedelta(hours=8))
    local_dt = dt.astimezone(tzutc_8)
    result = ("_{}_{}_{}__{}_{}_{}".format(local_dt.year, local_dt.month, local_dt.day, local_dt.hour, local_dt.minute,
                                      local_dt.second))

    return result


def get_bigrams(words):
    result = []
    for i,w in enumerate(words):
        if i!=len(words)-1:
            result.append(words[i]+words[i+1])
        else:
            result.append(words[i]+'<end>')

    return result

def print_info(*inp,islog=True,sep=' '):
    from fastNLP import logger
    if islog:
        print(*inp,sep=sep)
    else:
        inp = sep.join(map(str,inp))
        logger.info(inp)

def better_init_rnn(rnn,coupled=False):
    import torch.nn as nn
    if coupled:
        repeat_size = 3
    else:
        repeat_size = 4
    # print(list(rnn.named_parameters()))
    if hasattr(rnn,'num_layers'):
        for i in range(rnn.num_layers):
            nn.init.orthogonal_(getattr(rnn,'weight_ih_l'+str(i)).data)
            weight_hh_data = torch.eye(rnn.hidden_size)
            weight_hh_data = weight_hh_data.repeat(1, repeat_size)
            with torch.no_grad():
                getattr(rnn,'weight_hh_l'+str(i)).set_(weight_hh_data)
            nn.init.constant_(getattr(rnn,'bias_ih_l'+str(i)).data, val=0)
            nn.init.constant_(getattr(rnn,'bias_hh_l'+str(i)).data, val=0)

        if rnn.bidirectional:
            for i in range(rnn.num_layers):
                nn.init.orthogonal_(getattr(rnn, 'weight_ih_l' + str(i)+'_reverse').data)
                weight_hh_data = torch.eye(rnn.hidden_size)
                weight_hh_data = weight_hh_data.repeat(1, repeat_size)
                with torch.no_grad():
                    getattr(rnn, 'weight_hh_l' + str(i)+'_reverse').set_(weight_hh_data)
                nn.init.constant_(getattr(rnn, 'bias_ih_l' + str(i)+'_reverse').data, val=0)
                nn.init.constant_(getattr(rnn, 'bias_hh_l' + str(i)+'_reverse').data, val=0)


    else:
        nn.init.orthogonal_(rnn.weight_ih.data)
        weight_hh_data = torch.eye(rnn.hidden_size)
        weight_hh_data = weight_hh_data.repeat(repeat_size,1)
        with torch.no_grad():
            rnn.weight_hh.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        print('rnn param size:{},{}'.format(rnn.weight_hh.size(),type(rnn)))
        if rnn.bias:
            nn.init.constant_(rnn.bias_ih.data, val=0)
            nn.init.constant_(rnn.bias_hh.data, val=0)

    # print(list(rnn.named_parameters()))


def get_crf_zero_init(label_size, include_start_end_trans=False, allowed_transitions=None,
                 initial_method=None):
    import torch.nn as nn
    from fastNLP.modules import ConditionalRandomField
    crf = ConditionalRandomField(label_size, include_start_end_trans)

    crf.trans_m = nn.Parameter(torch.zeros(size=[label_size, label_size], requires_grad=True))
    if crf.include_start_end_trans:
        crf.start_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
        crf.end_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
    return crf

def get_peking_time():
    import time
    import datetime
    import pytz

    tz = pytz.timezone('Asia/Shanghai')  # 东八区

    t = datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime('%Y_%m_%d_%H_%M_%S')
    return t

def norm_static_embedding(x,norm=1):
    with torch.no_grad():
        x.embedding.weight /= (torch.norm(x.embedding.weight, dim=1, keepdim=True) + 1e-12)
        x.embedding.weight *= norm

def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))

def size2MB(size_,type_size=4):
    num = 1
    for s in size_:
        num*=s

    return num * type_size /1000 /1000

def load_label_mapping(file_path, label_to_id):
    """
    Reads the hierarchy file and maps label IDs to their General Categories.
    
    Args:
        file_path (str): Path to the hierarchy file.
        label_to_id (dict): Dictionary mapping label strings to IDs.
        
    Returns:
        tuple: (disease_ids, other_ids, ignore_ids)
    """
    # Read hierarchy file
    label_category_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                label = parts[0]
                category = parts[1]
                label_category_map[label] = category

    disease_ids = set()
    other_ids = set()
    ignore_ids = set()

    for label, label_id in label_to_id.items():
        # Handle 'O' and padding
        if label == 'O' or label == '<pad>' or label == '<unk>':
            ignore_ids.add(label_id)
            continue
        
        # Strip BIO/BMES tags
        # Assuming format like B-Type or I-Type
        if '-' in label:
            clean_label = label.split('-', 1)[1]
        else:
            clean_label = label
            
        category = label_category_map.get(clean_label)
        
        if category == 'disease' or category == 'dis':
            disease_ids.add(label_id)
        elif category in ['symptom', 'body', 'sym']:
            other_ids.add(label_id)
        else:
            # If not found in hierarchy, maybe treat as ignore or just skip?
            # User said: "ignore_ids: IDs for 'O' tag or padding."
            # If it's a label that is not 'O' but not in hierarchy, it's ambiguous.
            # But based on the task, it seems all relevant labels should be in hierarchy.
            pass
            
    return disease_ids, other_ids, ignore_ids

def calculate_hsr_loss(hidden_states, labels, disease_ids, other_ids, alpha=60.0):
    """
    CPU-Offloaded HSR Loss.
    
    Moves the linear algebra operations to CPU to avoid:
    1. MAGMA errors (Singular matrix on GPU)
    2. cusparse errors (Backward pass implementation missing/broken on GPU)
    """
    batch_size, seq_len, dim = hidden_states.shape
    
    # Flatten
    hidden_states_flat = hidden_states.reshape(-1, dim) # [N_tokens, Dim]
    labels_flat = labels.view(-1)
    
    original_device = hidden_states.device
    
    # 1. 提取向量 (仍在 GPU 上进行，利用并行优势)
    disease_mask = torch.zeros_like(labels_flat, dtype=torch.bool)
    other_mask = torch.zeros_like(labels_flat, dtype=torch.bool)
    
    for did in disease_ids:
        disease_mask |= (labels_flat == did)
    for oid in other_ids:
        other_mask |= (labels_flat == oid)
        
    V_D = hidden_states_flat[disease_mask] # [N_D, Dim]
    V_N = hidden_states_flat[other_mask]   # [N_N, Dim]
    
    if V_D.shape[0] == 0 or V_N.shape[0] == 0:
        return torch.tensor(0.0, device=original_device, requires_grad=True)

    # ================= 核心修改：搬运到 CPU =================
    # 将用于矩阵运算的张量转移到 CPU，并转为 double 精度
    # PyTorch 会自动记录这次移动，反向传播时梯度会正确传回 GPU
    Z_D_cpu = V_D.t().cpu().double() # [Dim, N_D]
    Z_N_cpu = V_N.t().cpu().double() # [Dim, N_N]
    
    # 2. 计算特征协方差矩阵 (Dim x Dim)
    # 此时在 CPU 上运行，使用 MKL/OpenBLAS，极其稳定
    Cov_feat = Z_D_cpu @ Z_D_cpu.t() 
    
    # 3. 添加正则化
    # 确保对角线数值足够大，防止奇异
    identity = torch.eye(dim, dtype=torch.double) # CPU tensor
    Cov_feat = Cov_feat + alpha * identity
    
    # 双重保险：加一个微小的 jitter 防止纯 0 输入导致的数值问题
    Cov_feat = Cov_feat + 1e-5 * identity
    
    # 4. 求解线性方程
    # M = (Cov_feat)^-1 @ Z_N
    try:
        # 在 CPU 上，solve 非常可靠
        # 使用 torch.linalg.solve (PyTorch 1.9+) 或 torch.solve (旧版)
        try:
            M = torch.linalg.solve(Cov_feat, Z_N_cpu)
        except AttributeError:
            # 兼容旧版本 PyTorch (如 1.7/1.8)
            M, _ = torch.solve(Z_N_cpu, Cov_feat)
    except RuntimeError:
        # 如果连 CPU 都解不出来，说明数据本身有问题（比如全是NaN），返回0梯度避免训练崩溃
        return torch.tensor(0.0, device=original_device, requires_grad=True)
    
    # 5. 计算投影并计算 Loss
    # G_hat = Z_D @ Z_D^T @ M
    # 为了节省显存和避免复杂计算，我们在 CPU 上算完再送回 GPU
    Kernel = Z_D_cpu @ Z_D_cpu.t()
    G_hat_cpu = Kernel @ M
    
    # 6. 将结果送回 GPU 并在 GPU 上计算最终的 Norm
    # 这一步将梯度链重新接回 GPU
    G_hat_gpu = G_hat_cpu.to(original_device).type_as(hidden_states).t()
    V_N_gpu = V_N.to(original_device).type_as(hidden_states)
    
    loss = torch.norm(V_N_gpu - G_hat_gpu, p='fro')
    
    return loss

if __name__ == '__main__':
    a = get_peking_time()
    print(a)
    print(type(a))






