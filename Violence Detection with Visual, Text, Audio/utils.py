#utils.py
import visdom
import numpy as np
import torch
import random
import os
import torch.nn as nn

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def disp_image(self, name, img):
        self.vis.image(img=img, win=name, opts=dict(title=name))

    def lines(self, name, line, X=None):
        if X is None:
            self.vis.line(Y=line, win=name)
        else:
            self.vis.line(X=X, Y=line, win=name)

    def scatter(self, name, data):
        self.vis.scatter(X=data, win=name)

# 특징을 지정된 길이로 조정하는 함수
#def process_feat(feat, length):
#    if len(feat.shape) == 1:
#        feat = np.expand_dims(feat, axis=1)
#    elif len(feat.shape) != 2:
#        raise ValueError(f"Unexpected feature shape: {feat.shape}")
    
#    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
#    r = np.linspace(0, len(feat), length + 1, dtype=int)  # 샘플링할 프레임 인덱스 계산
#    for i in range(length):
#        if r[i] != r[i + 1]:
#            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)  # 범위 내 특징 평균
#        else:
#            new_feat[i, :] = feat[r[i], :]  # 사용할 수 있는 마지막 특징으로 패딩
#    return new_feat

def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=int)  # len=33,存入要取的frame index
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)  # r[i]:r[i+1]这些feat求平均
        else:
            new_feat[i, :] = feat[r[i], :]  # 不足32帧补全
    return new_feat



def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=0)[0])
        min_val = relu(torch.min(act_map, dim=0)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret

def modelsize(model, input, type_size=4):
    # check GPU utilisation
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
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))

def save_best_record(test_info, file_path, metrics):
    with open(file_path, "w") as fo:
        fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
        fo.write(metrics + ": " + str(test_info[metrics][-1]))

def vid_name_to_path(vid_name, mode):  # TODO: change absolute paths! (only used by visual codes)
    root_dir = '/home/acsguser/Codes/SwinBERT/datasets/Crime/data/'
    types = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery",
             "Shooting", "Shoplifting", "Stealing", "Vandalism"]
    for t in types:
        if vid_name.startswith(t):
            path = root_dir + t + '/' + vid_name
            return path
    if vid_name.startswith('Normal'):
        if mode == 'train':
            path = root_dir + 'Training_Normal_Videos_Anomaly/' + vid_name
        else:
            path = root_dir + 'Testing_Normal_Videos_Anomaly/' + vid_name
        return path
    raise Exception("Unknown video type!!!")

def seed_everything(seed=4869):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gt(ds, gt_file):
    if gt_file is not None:
        gt = np.load(gt_file)
    else:
        if 'violence' in ds:
            gt = np.load('list/gt-violence.npy')
        else:
            raise Exception("Dataset undefined!!!")
    return gt
