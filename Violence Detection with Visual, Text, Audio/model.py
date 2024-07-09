# python main.py --dataset violence --feature-group both --fusion concat --extra_loss
import torch
import torch.nn as nn
import torch.nn.init as torch_init

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# 자비에 초기화
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

# NLB
class _NonLocalBlockND(nn.Module):
    # in_channels: 입력 채널 수
    # inter_channels: 중간 채널 수 (None일 경우 입력 채널 수의 절반)
    # dimension: 차원 (1, 2, 3 중 하나)
    # sub_sample: 하위 샘플링 여부
    # bn_layer: 배치 정규화 레이어 포함 여부
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        # 초기화 
        assert dimension in [1, 2, 3]
        self.dimension = dimension    
        self.sub_sample = sub_sample   
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # 중간 채널 수가 None인 경우 기본값(inchannels//2)
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # 차원에 따라 Conv 및 MaxPool 레이어 설정
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # g 레이어 정의 (1x1 컨볼루션, 채널 축소)
        # 1x1 컨볼루션을 통해 입력 채널 수를 중간 채널 수로 축소
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        
        # W 레이어 정의 (BatchNorm 포함 여부에 따라 다름)
        # 중간 체널 -> 입력 체널 수로 확장?
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

         # theta 레이어 정의 (1x1 컨볼루션, 채널 축소)
         # 1x1 컨볼루션을 통해 입력 채널 수를 중간 채널 수로 축소
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        # phi 레이어 정의 (1x1 컨볼루션, 채널 축소)
        # 1x1 컨볼루션을 통해 입력 채널 수를 중간 채널 수로 축소
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        
        # 하위 샘플링이 필요한 경우 g와 phi에 MaxPool 레이어 추가
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        # x : (640, 480, 32)
        batch_size = x.size(0)     # 640

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)    # g_x : (640, 240, 32)
        g_x = g_x.permute(0, 2, 1)      # g_x : (640, 32, 240)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # theta_x : (640, 240, 32)
        theta_x = theta_x.permute(0, 2, 1)          # theta_x : (640, 32, 240)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)   # phi_x : (640, 240, 32)

        f = torch.matmul(theta_x, phi_x)    # f : (640,32,32)
        N = f.size(-1)              # 32
        f_div_C = f / N             # f_div_C : (640,32,32)

        y = torch.matmul(f_div_C, g_x)       # y : (640,32,240)
        y = y.permute(0, 2, 1).contiguous()  # y : (640,240,32)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # y : (640,240,32)
        W_y = self.W(y)          # W_y : (640, 480, 32)
        z = W_y + x              # z : (640, 480, 32)

        if return_nl_map:
            return z, f_div_C
        return z

class NONLocalBlock1D(_NonLocalBlockND):
    # input : (640, 480, 32)
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

# Multiscale Temporal Network
class Aggregate(nn.Module):
    def __init__(self, len_feature):
        # concat 기준 len_feature = 1024+768+128 (1920)
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature    # 1920
        
        # 확장 컨볼루션 dilation = 1
        # (batch, len_feature, seq_len) -> (batch, len_feature/4, seq_len)
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=3,
                      stride=1, dilation=1, padding=1),
            nn.ReLU(),
            bn(int(len_feature/4))
        )
        # 확장 컨볼루션 dilation = 2
        # (batch, len_feature, seq_len) -> (batch, len_feature/4, seq_len)
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(int(len_feature/4))
        )
        # 확장 컨볼루션 dilation = 4
        # (batch, len_feature, seq_len) -> (batch, len_feature/4, seq_len)
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(int(len_feature/4))
        )
        # Conv1d 블록 (1x1 컨볼루션) : NLB입력 전 차원 조정
        # (batch, len_feature, seq_len) -> (batch, len_feature/4, seq_len)
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        # Conv1d 블록 (3x3 컨볼루션) : PDC, NLB concat후 차원 조정
        # (batch, len_feature, seq_len) -> (batch, len_feature, seq_len)
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(len_feature)
        )
        # Non-local block 정의
        self.non_local = NONLocalBlock1D(int(len_feature/4), sub_sample=False, bn_layer=True)

    def forward(self, x):
        # # f(vta) : (5*batch, 32, 1024+768+128)
        out = x.permute(0, 2, 1)   # out : (5*batch, 1024+768+128, 32)
        residual = out

        # 3layers pyramid dilated convolution block
        out1 = self.conv_1(out)    # out1 : (640, 480, 32)      1920 / 4 == 480
        out2 = self.conv_2(out)    # out2 : (640, 480, 32)
        out3 = self.conv_3(out)    # out3 : (640, 480, 32)
        out_d = torch.cat((out1, out2, out3), dim=1)   # out_d : (640, 1440, 32)

        out = self.conv_4(out)         # out : (640, 480, 32)   --> nlb 들어가기 전 조정
        out = self.non_local(out)      # out : (640, 480, 32)   --> nlb 후
        out = torch.cat((out_d, out), dim=1)   # (640, 1920, 32)  --> nlb, pdc concat 후
        out = self.conv_5(out)          # (640, 1920, 32)
        out = out + residual            # (640, 1920, 32)   --> 잔차 연결
        out = out.permute(0, 2, 1)      # (640, 32, 1920)
        return out
        

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.fusion = args.fusion
        self.batch_size = args.batch_size
        self.feature_group = args.feature_group
        self.aggregate_text = args.aggregate_text
        self.aggregate_audio = args.aggregate_audio
        self.num_segments = 32
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        args.hidden_dim = 256  # Set hidden_dim to 256

        #######################################
        if self.fusion == 'concat':
            self.Aggregate = Aggregate(len_feature=args.feature_size + args.audio_dim + args.emb_dim)
            self.fc1 = nn.Linear(args.feature_size + args.emb_dim + args.audio_dim, 512)
        elif self.fusion == 'add':
            self.Aggregate = Aggregate(args.audio_dim)
            self.fc_v = nn.Linear(args.feature_size, args.audio_dim)
            self.fc_t = nn.Linear(args.emb_dim, args.audio_dim)
            self.fc1 = nn.Linear(args.audio_dim, 512)
        elif self.fusion == 'product':
            self.Aggregate = Aggregate(args.audio_dim)
            self.fc_v = nn.Linear(args.feature_size, args.audio_dim)
            self.fc_t = nn.Linear(args.emb_dim, args.audio_dim)
            self.fc1 = nn.Linear(args.audio_dim, 512)
        elif self.fusion == 'projected':
            self.fc_v = nn.Linear(args.feature_size, args.hidden_dim)
            self.fc_t = nn.Linear(args.emb_dim, args.hidden_dim)
            self.fc_a = nn.Linear(args.audio_dim, args.hidden_dim)
            self.Aggregate = Aggregate(len_feature=args.hidden_dim * 3)
            self.fc1 = nn.Linear(args.hidden_dim * 3, 512)
        elif self.fusion == 'detour':
            self.conv1d1 = nn.Conv1d(in_channels=args.feature_size, out_channels=512, kernel_size=1, padding=0)
            self.conv1d2 = nn.Conv1d(in_channels=512, out_channels=args.audio_dim, kernel_size=1, padding=0)
            self.conv1d3 = nn.Conv1d(in_channels=args.emb_dim, out_channels=512, kernel_size=1, padding=0)
            
            self.fc1 = nn.Linear(args.audio_dim * 3, 512)
            self.Aggregate = Aggregate(len_feature=args.audio_dim * 3)
        #######################################
        
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs, text, audio):
        # top k : 이상탐지에 사용할 스니펫 수 32//3 == 3
        k_abn = self.k_abn
        k_nor = self.k_nor
    
        #######################################################################################
        # initial feautres
        # batch = arg.batch_size * 2 : (normal + abnormal)
        # (batch, crop, segments, feature)
        out = inputs
        bs, ncrops, t, f = out.size()        # visual: (batch, 5, 32, 1024)
        bs2, ncrops2, t2, f2 = text.size()   # text: (batch, 5, 32, 768)
        bs3, ncrops3, t3, f3 = audio.size()  # audio: (batch, 5, 32, 128)
    
        out = out.view(-1, t, f)        # out: (5*batch, 32, 1024)
        out2 = text.view(-1, t2, f2)    # out2: (5*batch, 32, 768)
        out3 = audio.view(-1, t3, f3)   # out3: (5*batch, 32, 128)
    
        # 초기융합 전 특징 크기 맞추기(test시 각 특징의 스니펫수가 다를 수 있음)
        if out.shape[1] < out2.shape[1]:
            out2 = out2[:, :out.shape[1], :]
        elif out.shape[1] > out2.shape[1]:
            repeat_factor = (out.shape[1] + out2.shape[1] - 1) // out2.shape[1]
            out2 = out2.repeat(1, repeat_factor, 1)[:, :out.shape[1], :]
        
        #######################################################################################
        # early fusion
        if self.fusion == 'concat':
            out = torch.cat((out, out2, out3), -1)    # f(vta) : (5*batch, 32, 1024+768+128)
        elif self.fusion == 'add':
            out = self.fc_v(out)                      # f'(v) : (5*batch, 32, 128)
            out2 = self.fc_t(out2)                    # f'(t) : (5*batch, 32, 128)
            out = out + out2 + out3                   # f(vta) : (5*batch, 32, 128)
        elif self.fusion == 'product':
            out = self.fc_v(out)                      # f'(v) : (5*batch, 32, 128)
            out2 = self.fc_t(out2)                    # f'(t) : (5*batch, 32, 128)
            out = out * out2 * out3                   # f(vta) : (5*batch, 32, 128)
        elif self.fusion == 'projected':
            out_v = self.fc_v(out)                     # f'(v) : (5*batch, 32, 256)
            out_t = self.fc_t(out2)                    # f'(t) : (5*batch, 32, 256)
            out_a = self.fc_a(out3)                    # f'(a) : (5*batch, 32, 256)
            out = torch.cat((out_v, out_t, out_a), -1) # f(vta) : (5*batch, 32, 3*256)
        elif self.fusion == 'detour':
            out = out.permute(0, 2, 1)                 # for conv1d -> f(v) : (5*batch, 1024, 32)
            out = self.relu(self.conv1d1(out))         # f'(v) : (5*batch, 512, 32)
            out = self.drop_out(out)
            out = self.relu(self.conv1d2(out))         # f'(v) : (5*batch, 128, 32)
            out = self.drop_out(out)
            out = out.permute(0, 2, 1)                 # back -> f'(v) : (5*batch, 32, 128)
            out2 = out2.permute(0, 2, 1)               # for conv1d -> f(t) : (5*batch, 768, 32)
            out2 = self.relu(self.conv1d3(out2))       # f'(t) : (5*batch, 512, 32)
            out2 = self.drop_out(out2)
            out2 = self.relu(self.conv1d2(out2))       # f'(t) : (5*batch, 128, 32)
            out2 = self.drop_out(out2)
            out2 = out2.permute(0, 2, 1)               # back -> f'(t) : (5*batch, 32, 128)
            out = torch.cat((out, out2, out3), -1)     # f(vta) : (5*batch, 32, 3*256)
        #######################################################################################
        # 이제부터 concat 기준
        # Multiscale temporal network 
        out = self.Aggregate(out)    # f(MTN) : (5*batch, 32, 1024+768+128)
        out = self.drop_out(out)
        #######################################################################################
        # 특징 크기 및 점수 계산
        t = out.shape[1]     # t : 스니펫(세그먼트)수
    
        features = out
        scores = self.relu(self.fc1(features))        # (5*batch, 32, 512)
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))          # (5*batch, 32, 128)
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))       #  (5*batch, 32, 1)
        scores = scores.view(bs, ncrops, -1).mean(1)  # (batch, 5, 32) -> 두번째 차원(ncrop)에 따라 평균 계산 (batch, 32)
        scores = scores.unsqueeze(dim=2)              # (batch, 32, 1)
        
        # 정상 비디오와 비정상 비디오의 특징 및 점수 분리(앞쪽 normal / 뒷쪽 abnormal)
        # 배치사이즈 : arg.batch_size , batch : 배치사이즈*2
        normal_features = features[0:self.batch_size * ncrops]      # (5*배치사이즈, 32, 1024+768+128)
        normal_scores = scores[0:self.batch_size]                   # (배치사이즈, 32, 1)
        abnormal_features = features[self.batch_size * ncrops:]     # (5*배치사이즈, 32, 1024+768+128)
        abnormal_scores = scores[self.batch_size:]                  # (배치사이즈, 32, 1)

        # 특징 크기 계산(L2-Norm 사용)
        # 텐서의 2번 축(즉, 특징 축)을 따라 각 벡터의 L2 노름(즉, 유클리드 거리)을 계산
        feat_magnitudes = torch.norm(features, p=2, dim=2)               # (5*batch, 32)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)   # (batch, 5, 32) -> (batch, 32)
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]             # (배치사이즈, 32)
        afea_magnitudes = feat_magnitudes[self.batch_size:]              # (배치사이즈, 32) 
        n_size = nfea_magnitudes.shape[0]

        # test일때
        if nfea_magnitudes.shape[0] == 1:
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        # 비정상 비디오 처리 : 상위 k개 특징 크기 선택
        select_idx = torch.ones_like(nfea_magnitudes).cuda()
        select_idx = self.drop_out(select_idx)
        afea_magnitudes_drop = afea_magnitudes * select_idx              # (배치사이즈,32)
        # 드롭아웃 적용된 피처 크기 중 상위 3개의 인덱스를 선택
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]      # (배치사이즈, 3)
        # 선택된 상위 3개 피처의 인덱스를 확장하여 3차원 인덱스를 생성
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])   # (배치, 3, 특징)
        abnormal_features = abnormal_features.view(n_size, ncrops, t, -1)        # (5*배치,32,특징) -> (배치, 5, 32, 특징)
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)                # (5, 배치, 32, 특징)
        total_select_abn_feature = torch.zeros(0)
        # 상위 3개의 특징 크기 선택
        for abnormal_feature in abnormal_features:
            # abnormal_feat:(배치,스니펫수,특징)에 대해 스니펫 기준으로 상위 인덱스 선택
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)
            # total_select_abn_feature : (배치*5, 3, 특징)
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))
        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])        # (배치, 3, 1)
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)    # (배치,3,1) -> 평균 후 (배치,1)

        # 정상 비디오 처리 : 상위 k개 특징 크기 선택
        select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])
    
        normal_features = normal_features.view(n_size, ncrops, t, -1)
        normal_features = normal_features.permute(1, 0, 2, 3)
    
        total_select_nor_feature = torch.zeros(0)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))
    
        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)
    
        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

    	# score_abnormal : 비정상 비디오의 상위 3개 피처의 점수 평균 (배치,1)
        # score_normal : 정상 비디오의 상위 3개 피처의 점수 평균 (배치,1)
        # feat_select_abn : 비정상 비디오에서 선택된 상위 3개의 피처 (배치*5, 3, 특징)
        # feat_select_normal : 정상 비디오에서 선택된 상위 3개의 피처 (배치*5, 3, 특징)
        # scores : 비디오의 각 스니펫에 대한 점수 (2*배치, 32, 1)
        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes