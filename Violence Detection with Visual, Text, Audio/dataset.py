import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality  # 데이터 모달리티 설정
        self.emb_folder = args.emb_folder  # 임베딩 폴더 경로 설정
        self.audio_folder = args.audio_folder  # 임베딩 폴더 경로 설정
        self.is_normal = is_normal  # 정상 데이터 여부 설정
        self.dataset = args.dataset  # 데이터셋 이름 설정
        self.feature_size = args.feature_size  # 피처 크기 설정

        if test_mode:
            self.rgb_list_file = args.test_rgb_list
            self.audio_list_file = args.test_audio_list
        else:
            self.rgb_list_file = args.rgb_list
            self.audio_list_file = args.audio_list

        if 'v2' in self.dataset:
            self.feat_ver = 'v2'
        elif 'v3' in self.dataset:
            self.feat_ver = 'v3'
        else:
            self.feat_ver = 'v1'

        self.transform = transform  # 데이터 변환 설정
        self.test_mode = test_mode  # 테스트 모드 설정
        self._parse_list()  # 리스트 파싱
        self.num_frame = 0  # 프레임 수 초기화
        self.labels = None  # 레이블 초기화

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        self.audio_list = list(open(self.audio_list_file))
        if not self.test_mode:
            if 'violence' in self.dataset:
                if self.is_normal:
                    self.list = self.list[1904:]
                    print('폭력 데이터셋의 정상 리스트')
                else:
                    self.list = self.list[:1904]
                    print('폭력 데이터셋의 비정상 리스트')
            else:
                raise Exception("데이터셋이 정의되지 않았습니다!!!")

    def __getitem__(self, index):
        label = self.get_label()
        i3d_path = self.list[index].strip('\n')

        if self.feat_ver == 'v2':
            i3d_path = i3d_path.replace('i3d_v1', 'i3d_v2')
        elif self.feat_ver == 'v3':
            i3d_path = i3d_path.replace('i3d_v1', 'i3d_v3')

        features = np.load(i3d_path, allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        text_path = f"save/Violence/{self.emb_folder}/{i3d_path.split('/')[-1][:-7]}emb.npy"
        text_features = np.load(text_path, allow_pickle=True)
        text_features = np.array(text_features, dtype=np.float32)

        audio_path = f"save/Violence/{self.audio_folder}/{i3d_path.split('/')[-1][:-7]}_vggish.npy"
        audio_features = np.load(audio_path, allow_pickle=True)
        audio_features = np.array(audio_features, dtype=np.float32)

        #print(f"Original feature shape: {features.shape}")
        #print(f"Original text feature shape: {text_features.shape}")
        #print(f"Original audio feature shape: {audio_features.shape}")
        
        if self.feature_size == 1024:
            text_features = np.tile(text_features, (5, 1, 1))
            audio_features = np.tile(audio_features, (5, 1, 1))
        else:
            raise Exception("피처 크기가 정의되지 않았습니다!!!")

        #print(f"Tiled text feature shape: {text_features.shape}")
        #print(f"Tiled audio feature shape: {audio_features.shape}")
        
        if self.transform is not None:
            features = self.transform(features)
            audio_features = self.transform(audio_features)

        if self.test_mode:
            text_features = text_features.transpose(1, 0, 2)
            audio_features = audio_features.transpose(1, 0, 2)

            return features, text_features, audio_features
        else:
            features = features.transpose(1, 0, 2)
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            div_feat_text = []
            for text_feat in text_features:
                text_feat = process_feat(text_feat, 32)
                div_feat_text.append(text_feat)
            div_feat_text = np.array(div_feat_text, dtype=np.float32)

            div_feat_audio = []
            for audio_feat in audio_features:
                audio_feat = process_feat(audio_feat, 32)
                div_feat_audio.append(audio_feat)
            div_feat_audio = np.array(div_feat_audio, dtype=np.float32)

            assert divided_features.shape[1] == div_feat_text.shape[1] == div_feat_audio.shape[1], \
                f"{self.test_mode}\t{divided_features.shape[1]}\t{div_feat_text.shape[1]}\t{div_feat_audio.shape[1]}"

            #print(f"최종 feature shape: {divided_features.shape}")
            #print(f"최종 text feature shape: {div_feat_text.shape}")
            #print(f"최종 audio feature shape: {div_feat_audio.shape}")

            return divided_features, div_feat_text, div_feat_audio, label

    def get_label(self):
        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
