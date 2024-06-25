# option.py
import argparse

parser = argparse.ArgumentParser(description='RTFM')

# Feature extractor and feature size settings
parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'], help='Feature extractor to use (default: i3d)')
parser.add_argument('--feature_size', type=int, default=1024, help='Size of feature (default: 2048)')
parser.add_argument('--audio-size', type=int, default=128, help='Size of feature (default: 2048)')
parser.add_argument('--hidden_dim', type=int, default=256, help='Size of feature (default: 2048)')

# Modality settings
parser.add_argument('--modality', default='RGB', help='Type of the input: RGB, AUDIO, or MIX')

# Ground truth and GPU settings
parser.add_argument('--gt', default=None, help='File of ground truth')
parser.add_argument('--gpus', default=1, type=int, choices=[0], help='Number of GPUs to use (default: 1)')

# Learning rate and batch size settings
parser.add_argument('--lr', type=str, default='[1e-3]*15000', help='Learning rates for steps (list form)')
parser.add_argument('--alpha', type=float, default=1e-4, help='Weight for RTFM loss')
parser.add_argument('--batch-size', type=int, default=64, help='Number of instances in a batch of data (default: 32)')

# DataLoader settings
parser.add_argument('--workers', type=int, default=4, help='Number of workers in dataloader (default: 4)')

# Model and checkpoint settings
parser.add_argument('--model-name', default='rtfm', help='Name to save the model')
parser.add_argument('--pretrained-ckpt', default=None, help='Checkpoint for pretrained model')

# Class and dataset settings
parser.add_argument('--num-classes', type=int, default=1, help='Number of classes (default: 1)')
parser.add_argument('--dataset', default='ucf', help='Dataset to train on (shanghai, ucf, ped2, violence, TE2)')

# Plotting and random seed settings
parser.add_argument('--plot-freq', type=int, default=10, help='Frequency of plotting (default: 10)')
parser.add_argument('--seed', type=int, default=4869, help='Random seed (default: 4869)')

# Training settings
parser.add_argument('--max-epoch', type=int, default=3000, help='Maximum number of epochs to train (default: 1000)')
parser.add_argument('--feature-group', default='both', choices=['both', 'vis', 'text'], help='Feature groups used for the model')
parser.add_argument('--fusion', type=str, default='concat', help='How to fuse visual and text features')
parser.add_argument('--normal_weight', type=float, default=1, help='Weight for normal loss weights')
parser.add_argument('--abnormal_weight', type=float, default=1, help='Weight for abnormal loss weights')
parser.add_argument('--aggregate_text', action='store_true', default=True, help='Whether to aggregate text features')
parser.add_argument('--extra_loss', action='store_true', default=False, help='Whether to use extra loss')
parser.add_argument('--save_test_results', action='store_true', default=False, help='Whether to save test results')

parser.add_argument('--emb_folder', type=str, default='sent_emb_n', help='Folder for text embeddings, used to differentiate different swinbert pretrained models')
parser.add_argument('--audio_folder', type=str, default='Violence_vggish', help='Folder for text embeddings, used to differentiate different swinbert pretrained models')
parser.add_argument('--emb_dim', type=int, default=768, help='Dimension of text embeddings')
parser.add_argument('--audio_dim', type=int, default=128, help='Dimension of text embeddings')
# Feature list settings
parser.add_argument('--rgb-list', default='list/violence-i3d.list', help='List of RGB features')
parser.add_argument('--audio-list', default='list/audio.list', help='List of audio features')
parser.add_argument('--test-rgb-list', default='list/violence-i3d-test.list', help='List of test RGB features')
parser.add_argument('--test-audio-list', default='list/audio_test.list', help='List of test audio features')

# Audio aggregation setting
parser.add_argument('--aggregate_audio', action='store_true', default=True, help='Whether to aggregate audio features')


parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate (default: 0.5)')

# 파서 객체를 모듈 외부에서 사용할 수 있도록 함수화
def get_args():
    return parser.parse_args()
