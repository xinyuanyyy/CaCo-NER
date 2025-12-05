import sys
sys.path.append("..")
from modeling.flat_main import *
from datetime import datetime

print("start training!")
grid_sep='='*119+'\n'
line_sep='-'*30+'\n'

stage1_param = {
    'batch': 16,
    'lr':   0.0008,
    'dim': 20,
    'head':10,
    'wd':0.2,
    'warmup':0.1,
    'temp':0.07,
    'hard_k': 10,
    'hard_weight': 1
}
stage2_param = {
    'batch_2':16,
    'lr_2':0.0004,
    'warmup_2':0.1,
    'wd_2':  0,
    'lambda_hsr': 0.1
}


##################### Manual set params #########################
data='cmeee_v2'
device='0'
fixed_seed=1080956
#################################################################

is_ctr=False


output_dir = '../runs/' + data + '-' + str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
writer_file = output_dir  + '/'+str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))


writer=open(writer_file,'a+')


args={'data':data,'device':device,}
writer.write(str(args)+'\n\n')

writer.flush()


# stage1
writer.write('\n'+'-'*27+' Stage1 Parameter '+'-'*27+'\n')
writer.write(str(stage1_param)+'\n')
writer.flush()

metric_ctr,state_path_ctr=flat_main(stage1_param['batch'], stage1_param['lr'], stage1_param['dim'], 
                                    stage1_param['head'], stage1_param['warmup'],
                                    dataset=data,device=device,ctr=True,
                                    output_dir=output_dir,weight_decay=stage1_param['wd'],
                                    temp=stage1_param['temp'],seed=fixed_seed,
                                    hard_k=stage1_param['hard_k'], hard_weight=stage1_param['hard_weight']) # ck=checkpoint

writer.write(str(metric_ctr)+'\n')
writer.write(state_path_ctr+'\n')
writer.write('-' * 72 + '\n\n')
writer.flush()

# state_path_ctr = '../runs/cmeee_v22025-12-02-12-44-17/history/result_2025_12_02_12_44_18_CTR/Flat_eval_bert-base-chinese_e_56_f_185.3918218305034.bin'


# stage2
writer.write('\n'+'-'*27+' Stage2 Parameter '+'-'*27+'\n')
writer.write(str(stage2_param)+'\n')
writer.flush()

second_metric, state_path = flat_main(stage2_param['batch_2'], stage2_param['lr_2'],stage1_param['dim'], 
                                      stage1_param['head'], stage2_param['warmup_2'], dataset=data, device=device,
                               ck=state_path_ctr, output_dir=output_dir,
                               weight_decay=stage2_param['wd_2'],only_head=False,
                               seed=fixed_seed, lambda_hsr=stage2_param['lambda_hsr'])



writer.write(str(second_metric) + '\n')
writer.write(state_path+ '\n\n')
writer.flush()