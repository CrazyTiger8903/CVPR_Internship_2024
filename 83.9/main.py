# main.py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from utils import *
from config import *

if __name__ == '__main__':
    # 명령줄 인수 파싱 및 설정 초기화
    args = option.parser.parse_args()
    config = Config(args)
    seed_everything(args.seed)

    # 설정에 따른 옵션 및 시각화 도구 설정
    text_opt = "text_agg" if args.aggregate_text else "no_text_agg"
    extra_loss_opt = "extra_loss" if args.extra_loss else "no_loss"
    if args.emb_folder == "":
        sb_pt_name = "vatex"
    else:
        sb_pt_name = args.emb_folder[11:]  # 'sent_emb_n_XXX'에서 'XXX' 부분을 사용
    print("Using SwinBERT pre-trained model: ", sb_pt_name)

    viz_name = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
        args.dataset, args.feature_group, text_opt, args.fusion, 
        args.normal_weight, args.abnormal_weight, extra_loss_opt, 
        args.alpha, sb_pt_name)
    viz = Visualizer(env=viz_name, use_incoming_socket=False)
    
    # 정상 비디오에 대한 데이터 로더 설정
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True, 
                               generator=torch.Generator(device='cuda'))
    # 비정상 비디오에 대한 데이터 로더 설정
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True, 
                               generator=torch.Generator(device='cuda'))
    # 테스트 데이터 로더 설정
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False, 
                             generator=torch.Generator(device='cuda'))

    # 모델 초기화 및 사전 학습된 가중치 로드
    model = Model(args)
    if args.pretrained_ckpt is not None:
        print("Loading pretrained model " + args.pretrained_ckpt)
        model.load_state_dict(torch.load(args.pretrained_ckpt))
    
    # 장치 설정 (GPU 사용 가능 여부에 따라 설정)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 체크포인트 디렉터리 생성
    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(),
                           lr=config.lr[0], weight_decay=0.0005)  # 기본 학습률=0.001

    # 테스트 정보 및 최적 결과 초기화
    test_info = {"epoch": [], "test_AUC": [], "test_AP": []}
    best_AUC, best_ap = -1, -1
    best_epoch = -1
    output_path = 'output'   # 결과를 저장할 경로 설정

    # 학습 루프
    for step in tqdm(range(1, args.max_epoch + 1), total=args.max_epoch, dynamic_ncols=True):
        # 학습률 조정
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        # 데이터 로더 이터레이터 생성
        if (step - 1) % len(train_nloader) == 0:  # step=1일 때 이터레이터 생성
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:  # step=1일 때 이터레이터 생성
            loadera_iter = iter(train_aloader)

        # 모델 학습
        train(loadern_iter, loadera_iter, model, args, optimizer, viz, device)

        # 테스트 및 체크포인트 저장
        if step % 5 == 0 and step > 50:
            auc, ap = test(test_loader, model, args, viz, device)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)
            test_info["test_AP"].append(ap)

            if "violence" in args.dataset:  # 폭력 데이터셋의 경우 AUPR 사용
                if test_info["test_AP"][-1] > best_ap:
                    best_ap = test_info["test_AP"][-1]
                    best_epoch = step
                    ckpt_path = './ckpt/' + '{}-{}-{}-{}-{}-{}-{}-{}-{}.pkl'.format(
                        args.dataset, args.feature_group, text_opt, args.fusion, 
                        args.alpha, extra_loss_opt, step, args.seed, sb_pt_name)
                    torch.save(model.state_dict(), ckpt_path)
                    save_best_record(test_info, os.path.join(output_path, 
                                                             '{}-{}-{}-{}-{}-{}-{}-{}-AP.txt'.format(
                                                                 args.dataset, args.feature_group, text_opt, 
                                                                 args.fusion, args.alpha, extra_loss_opt, 
                                                                 step, sb_pt_name)), "test_AP")
                # 테스트 결과 출력
                APs = test_info["test_AP"]
                APs_mean, APs_median, APs_std, APs_max, APs_min = np.mean(APs), np.median(APs), np.std(APs), np.max(APs), np.min(APs)
                print("std\tmean\tmedian\tmin\tmax\tAP")
                print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
                    APs_std * 100, APs_mean * 100, APs_median * 100, APs_min * 100, APs_max * 100))
            else:  # 다른 데이터셋의 경우 AUROC 사용
                if test_info["test_AUC"][-1] > best_AUC:
                    best_AUC = test_info["test_AUC"][-1]
                    best_epoch = step
                    ckpt_path = './ckpt/' + '{}-{}-{}-{}-{}-{}-{}-{}-{}.pkl'.format(
                        args.dataset, args.feature_group, text_opt, args.fusion, 
                        args.alpha, extra_loss_opt, step, args.seed, sb_pt_name)
                    torch.save(model.state_dict(), ckpt_path)
                    save_best_record(test_info, os.path.join(output_path, 
                                                             '{}-{}-{}-{}-{}-{}-{}-{}-AUC.txt'.format(
                                                                 args.dataset, args.feature_group, text_opt, 
                                                                 args.fusion, args.alpha, extra_loss_opt, 
                                                                 step, sb_pt_name)), "test_AUC")
                # 테스트 결과 출력
                AUCs = test_info["test_AUC"]
                AUCs_mean, AUCs_median, AUCs_std, AUCs_max, AUCs_min = np.mean(AUCs), np.median(AUCs), np.std(AUCs), np.max(AUCs), np.min(AUCs)
                print("std\tmean\tmedian\tmin\tmax\tAUC")
                print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
                    AUCs_std * 100, AUCs_mean * 100, AUCs_median * 100, AUCs_min * 100, AUCs_max * 100))

    print("Best result:" + viz_name + "-" + str(best_epoch))
    torch.save(model.state_dict(), './ckpt/' + args.dataset + 'final.pkl')
