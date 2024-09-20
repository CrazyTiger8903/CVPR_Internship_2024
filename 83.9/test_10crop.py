# test_10crop.py
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import numpy as np
from utils import get_gt

def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, (input, text, audio) in enumerate(dataloader):
            input = input.to(device)
            text = text.to(device)
            audio = audio.to(device)
            
            input = input.permute(0, 2, 1, 3)
            text = text.permute(0, 2, 1, 3)
            audio = audio.permute(0, 2, 1, 3)


            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(input, text, audio)  # 여기서 score_abnormal과 score_normal은 1차원이며, 각 비디오에 대한 점수이고, logits은 T차원의 벡터로 각 스니펫에 대해 점수를 매깁니다.
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))

        gt = get_gt(args.dataset, args.gt)

        pred = pred.cpu().detach().numpy()
        pred = np.repeat(pred, 16)  # 배열의 각 요소를 16번 반복합니다. 즉, 동일한 클립 내의 16프레임은 동일한 예측 결과를 공유합니다.

        # Truncate or pad the predictions to match the length of the ground truth
        if len(pred) > len(gt):
            pred = pred[:len(gt)]
        elif len(pred) < len(gt):
            pred = np.pad(pred, (0, len(gt) - len(pred)), 'constant', constant_values=0)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        rec_auc = auc(fpr, tpr)
        ap = average_precision_score(list(gt), pred)
        print('ap : ' + str(ap))
        print('auc : ' + str(rec_auc))

        # 시각화 도구를 사용하여 PR AUC, ROC AUC 및 기타 정보를 플롯합니다.
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)

        if args.save_test_results:
            np.save('results/' + args.dataset + '_pred.npy', pred)
            np.save('results/' + args.dataset + '_fpr.npy', fpr)
            np.save('results/' + args.dataset + '_tpr.npy', tpr)
            np.save('results/' + args.dataset + '_precision.npy', precision)
            np.save('results/' + args.dataset + '_recall.npy', recall)
            np.save('results/' + args.dataset + '_auc.npy', rec_auc)
            np.save('results/' + args.dataset + '_ap.npy', ap)

        # ROC 커브와 PR 커브를 플롯합니다.
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % rec_auc)
        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='green', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        plt.show()

        return rec_auc, ap
