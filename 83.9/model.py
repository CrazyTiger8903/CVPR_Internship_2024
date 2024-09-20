# model.py
import torch
import torch.nn as nn
import torch.nn.init as torch_init

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

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

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

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

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z

class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        #residual=x
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out
        #return out+residual

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, n_heads=8, ff_dim=1024, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(in_channels, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, in_channels),
        )
        self.layer_norm1 = nn.LayerNorm(in_channels)
        self.layer_norm2 = nn.LayerNorm(in_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, in_channels)
        x = x.permute(1, 0, 2)  # (seq_len, batch, in_channels) for MultiheadAttention
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(x + attn_output)

        out2 = self.ffn(out1)
        out2 = self.dropout2(out2)
        out = self.layer_norm2(out1 + out2)

        return out.permute(1, 0, 2)  # (batch, seq_len, in_channels)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=3,
                      stride=1, dilation=1, padding=1),
            nn.ReLU(),
            bn(int(len_feature/4))
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(int(len_feature/4))
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(int(len_feature/4))
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(len_feature)
        )

        #self.non_local = NONLocalBlock1D(int(len_feature/4), sub_sample=False, bn_layer=True)
        self.transformer = TransformerBlock(int(self.len_feature/4))
        #self.cbam_block = CBAM(1440)
        self.se_block = SEBlock(1440)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        residual = out

        out1 = self.conv_1(out)
        out2 = self.conv_2(out)
        out3 = self.conv_3(out)
        out_d = torch.cat((out1, out2, out3), dim=1)

        #out_d = self.cbam_block(out_d)
        out_d = self.se_block(out_d)

        out = self.conv_4(out)
        out = out.permute(2, 0, 1)  # out4: (seq_len, batch, 480) for Transformer
        out = self.transformer(out)
        out = out.permute(1, 2, 0)  # out4: (batch, 480, seq_len)

        out = torch.cat((out_d, out), dim=1)
        #out = self.conv_5(out)
        out = out + residual
        out = out.permute(0, 2, 1)
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

        self.Aggregate = Aggregate(len_feature=args.feature_size+args.audio_dim+args.emb_dim)  # Detour Fusion 후의 길이
        #self.Aggregate = Aggregate(len_feature=2*args.audio_dim)  # Detour Fusion 후의 길이
        #self.Aggregate = Aggregate(len_feature=args.audio_dim)  # Detour Fusion 후의 길이
        #self.Aggregate_text = Aggregate(len_feature=args.emb_dim+args.audio_dim)

        # Detour Fusion을 위한 Conv1D 레이어 추가
        self.conv1d1 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)
        self.conv1d2 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, padding=0)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(args.dropout)

        if self.feature_group == 'both':
            if args.fusion == 'concat':
                self.fc1 = nn.Linear(args.emb_dim + args.feature_size + args.audio_dim, 512)
                #self.fc1 = nn.Linear(args.emb_dim + 2 * args.audio_dim, 512)
                #self.fc1 = nn.Linear(args.emb_dim + args.audio_dim, 512)
            elif args.fusion == 'add' or args.fusion == 'product':
                #self.fc0 = nn.Linear(args.audio_dim, args.emb_dim)
                #self.fc0 = nn.Linear(2 * args.audio_dim, args.emb_dim)
                #self.fc0 = nn.Linear(args.feature_size+args.audio_dim, args.emb_dim+args.audio_dim)
                self.fc1 = nn.Linear(args.feature_size+args.emb_dim+args.audio_dim, 512)
            elif 'up' in args.fusion:
                #self.fc_vis = nn.Linear(args.audio_dim, args.emb_dim + args.audio_dim)
                #self.fc_text = nn.Linear(args.emb_dim, args.emb_dim + args.audio_dim)
                #self.fc1 = nn.Linear(args.emb_dim + args.audio_dim, 512)
                self.fc_vis = nn.Linear(2 * args.audio_dim, args.emb_dim + 2*args.audio_dim)
                self.fc_text = nn.Linear(args.emb_dim, args.emb_dim + 2*args.audio_dim)
                self.fc1 = nn.Linear(args.emb_dim + 2*args.audio_dim, 512)
            else:
                raise ValueError('Unknown fusion method: {}'.format(args.fusion))
        elif self.feature_group == 'text':
            self.fc1 = nn.Linear(args.emb_dim, 512)
        else:
            self.fc1 = nn.Linear(args.feature_size + args.audio_dim, 512)  # Detour Fusion 후의 길이
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs, text, audio):
        k_abn = self.k_abn
        k_nor = self.k_nor
    
        out = inputs
        bs, ncrops, t, f = out.size()   # out: (5, 32, 1024)
        #print(f'Initial out shape: {out.size()}')
        bs2, ncrops2, t2, f2 = text.size()   # text: (5, 32, 768)
        #print(f'Initial text shape: {text.size()}')
        bs3, ncrops3, t3, f3 = audio.size()   # audio: (5, 32, 128)
        #print(f'Initial audio shape: {audio.size()}')
    
        out = out.view(-1, t, f)   # out: (5*32, 32, 1024)
        #print(f'Reshaped out: {out.size()}')
        out2 = text.view(-1, t2, f2)   # out2: (5*32, 32, 768)
        #print(f'Reshaped text: {out2.size()}')
        out3 = audio.view(-1, t3, f3)   # out3: (5*32, 32, 128)
        #print(f'Reshaped audio: {out3.size()}')
    
        # Detour Fusion 수정 시작
        # 오디오와 비디오 특징을 결합하여 새로운 특징을 생성
        #out = out.permute(0, 2, 1)  # for conv1d, out: (5*32, 1024, 32)
        #print(f'Permuted out for conv1d: {out.size()}')
        #out = self.relu(self.conv1d1(out))  # out: (5*32, 512, 32)
        #print(f'After conv1d1: {out.size()}')
        #out = self.dropout(out)
        #out = self.relu(self.conv1d2(out))  # out: (5*32, 128, 32)
        #print(f'After conv1d2: {out.size()}')
        #out = self.dropout(out)
        #out = out.permute(0, 2, 1)  # b*t*c, out: (5*32, 32, 128)
        #print(f'Permuted out after conv1d: {out.size()}')

        #out size: torch.Size([640, 32, 1024]), out2 size: torch.Size([640, 32, 768]), out3 size: torch.Size([640, 32, 128])
        #out size: torch.Size([5, 382, 1024]), out2 size: torch.Size([5, 16, 768]), out3 size: torch.Size([5, 382, 128])
       
        # Debugging shapes
        #print(f"out size: {out.size()}, out2 size: {out2.size()}, out3 size: {out3.size()}")
    
            


        if out.shape[1] < out2.shape[1]:
            out2 = out2[:, :out.shape[1], :]
        elif out.shape[1] > out2.shape[1]:
            repeat_factor = (out.shape[1] + out2.shape[1] - 1) // out2.shape[1]
            out2 = out2.repeat(1, repeat_factor, 1)[:, :out.shape[1], :]
             
         # 비디오와 오디오 특징 결합
        out = torch.cat((out, out2, out3), -1)
        #out = out + out3
        #print(f'Concatenated video and audio features: {out.size()}')
        # Detour Fusion 수정 끝
    
        out = self.Aggregate(out)  # out: (5*32, 32, 256)
        #print(f'After Aggregate: {out.size()}')
        out = self.drop_out(out)

        # if self.aggregate_text:
        #     out2 = self.Aggregate_text(out2)
        #     #print(f'After Aggregate_text: {out2.size()}')
        #     out2 = self.drop_out(out2)

        # # Aggregation 후 텍스트와 결합할 데이터를 맞추는 부분 수정
        # if out.shape[1] < out2.shape[1]:
        #     out2 = out2[:, :out.shape[1], :]
        # elif out.shape[1] > out2.shape[1]:
        #     repeat_factor = (out.shape[1] + out2.shape[1] - 1) // out2.shape[1]
        #     out2 = out2.repeat(1, repeat_factor, 1)[:, :out.shape[1], :]
        # #print(f'Aligned text shape: {out2.size()}')
    
        t = out.shape[1]
        
        if self.fusion == 'concat':
            if self.feature_group == 'both':
                out = torch.cat([out, out2], dim=2)  # out: (5*32, 32, 256 + 768)
                #print(f'After concat fusion: {out.size()}')
            elif self.feature_group == 'text':
                out, ncrops, f = out2, ncrops2, f2
        elif self.fusion == 'product':
            out = self.relu(self.fc0(out))
            #print(f'After fc0: {out.size()}')
            out = self.drop_out(out)
            out2 = self.relu(self.fc_text(out2))
            out2 = self.drop_out(out2)
            out = out * out2
            #print(f'After product fusion: {out.size()}')
        elif self.fusion == 'add':
            # out = self.relu(self.fc0(out))  # vis feature reduces to dim=args.emb_dim
            # out = self.drop_out(out)
            # out = out + out2
            #print(f'After add fusion: {out.size()}')
            pass
        elif self.fusion == 'up':
            out = self.relu(self.fc_vis(out))
            out = self.drop_out(out)
            out2 = self.relu(self.fc_text(out2))
            out2 = self.drop_out(out2)
            out = out + out2
            #print(f'After up fusion: {out.size()}')
        else:
            raise ValueError('Unknown fusion method: {}'.format(self.fusion))
    
        features = out
        scores = self.relu(self.fc1(features))
        #print(f'After fc1: {scores.size()}')
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        #print(f'After fc2: {scores.size()}')
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        #print(f'After fc3: {scores.size()}')
        scores = scores.view(bs, ncrops, -1).mean(1)
        scores = scores.unsqueeze(dim=2)
        #print(f'Final scores shape: {scores.size()}')
    
        normal_features = features[0:self.batch_size * ncrops]
        normal_scores = scores[0:self.batch_size]
    
        abnormal_features = features[self.batch_size * ncrops:]
        abnormal_scores = scores[self.batch_size:]
    
        feat_magnitudes = torch.norm(features, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]
        afea_magnitudes = feat_magnitudes[self.batch_size:]
        n_size = nfea_magnitudes.shape[0]
    
        if nfea_magnitudes.shape[0] == 1:
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features
    
        select_idx = torch.ones_like(nfea_magnitudes).cuda()
        select_idx = self.drop_out(select_idx)
        afea_magnitudes_drop = afea_magnitudes * select_idx
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])
    
        abnormal_features = abnormal_features.view(n_size, ncrops, t, -1)
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)
    
        total_select_abn_feature = torch.zeros(0)
        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))
    
        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)
    
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
    
        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes
