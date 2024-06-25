import os
import json
import torch
import torch.nn.functional as F
from torchvision.transforms.transforms import *
from torchvision import transforms
from easydict import EasyDict as edict
from PIL import Image
import torchaudio
from test import get_model_attr
import argparse
from model.bert_tokenizer import BertTokenizer
from model.pretrain import VALOR

parser = argparse.ArgumentParser()
parser.add_argument("--video_dir", default=None, type=str)
parser.add_argument("--task", default=None, type=str)
parser.add_argument("--question", default=None, type=str)
parser.add_argument("--model_dir", default=None, type=str)

args = parser.parse_args()

def clean(self, text):
    """remove duplicate spaces, lower and remove punctuations """
    text = ' '.join([i for i in text.split(' ') if i != ''])
    text = text.lower()
    for i in self.punctuations:
        text = text.replace(i,'')
    return text

def get_padded_tokens(self,txt_tokens, type, max_len=None):
    max_len = self.max_len if max_len is None else max_len
    txt_tokens = txt_tokens[:max_len]
    if type=='bert':
        txt_tokens = [self.cls_token] + txt_tokens + [self.sep_token]  
    elif type=='clip':
        txt_tokens = [self.sot_token] + txt_tokens + [self.eot_token] 
    txt_tokens = torch.tensor(txt_tokens, dtype=torch.long)
    output = torch.zeros(max_len + 2, dtype=torch.long)
    output[:len(txt_tokens)] = txt_tokens
    return output

def split(frame_name_lists, sample_num):
    if len(frame_name_lists) < sample_num:   ###padding with the last frame
        frame_name_lists += [frame_name_lists[-1]]*(sample_num - len(frame_name_lists))
    k, m = divmod(len(frame_name_lists), sample_num)
    return [frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(sample_num))]

def load_from_pretrained_dir(pretrain_dir):
    checkpoint_dir = os.path.join(pretrain_dir,'ckpt')
    checkpoint_ls = [i for i in os.listdir(checkpoint_dir) if i.startswith('model_step')]
    checkpoint_ls = [int(i.split('_')[2].split('.')[0]) for i in checkpoint_ls]
    checkpoint_ls.sort()    
    step = checkpoint_ls[-1]
    checkpoint_name = 'model_step_'+str(step)+'.pt'
    ckpt_file = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_file, map_location='cpu')
    checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
    pretrain_cfg = edict(json.load(open(os.path.join(pretrain_dir,'log','hps.json'))))
    if 'video_frame_embedding' in checkpoint:
        checkpoint['video_frame_embedding'][:,pretrain_cfg.video_sample_num:] = checkpoint['video_frame_embedding'][:,pretrain_cfg.video_sample_num-1].clone()
    if 'audio_frame_embedding' in checkpoint: 
        checkpoint['audio_frame_embedding'][:,pretrain_cfg.audio_sample_num:] = checkpoint['audio_frame_embedding'][:,pretrain_cfg.audio_sample_num-1].clone()
    return checkpoint, pretrain_cfg

checkpoint, pretrain_cfg = load_from_pretrained_dir(args.model_dir)

model = VALOR.from_pretrained(pretrain_cfg, checkpoint)
model.eval().cuda()

video_dir = args.video_dir
video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # 파일명에서 확장자 제거
    output_dir = f'./inference2/{video_name}'
    fps_frame_dir = os.path.join(output_dir, f"frames_fps24")
    os.makedirs(fps_frame_dir, exist_ok=True)
    cmd = "ffmpeg -loglevel error -i {} -vsync 0 -f image2 -vf fps=fps=24 -qscale:v 2 {}/frame_%04d.jpg".format(
            video_path, fps_frame_dir)
    os.system(cmd)

    # Extract Audio
    audio_file_path = os.path.join(output_dir, video_name + '.wav')
    cmd = "ffmpeg -i {} -loglevel error -f wav -vn -ac 1 -ab 16k -ar {} -y {}".format(
            video_path, 22050, audio_file_path)
    os.system(cmd)

    if pretrain_cfg.video_encoder_type.startswith('clip'):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std  = [0.26862954, 0.26130258, 0.27577711]
    else:       
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    test_transforms = transforms.Compose([Resize((224,224)), Normalize(mean, std)])
    frames = os.listdir(fps_frame_dir)
    frames.sort()

    def get_snippet_frames(start_idx, end_idx, frame_path, sample_num=32):
        snippet_frames = []
        snippet_frame_names = frames[start_idx:end_idx]
        snippet_frame_names = snippet_frame_names + [snippet_frame_names[-1]] * (sample_num - len(snippet_frame_names))
        for frame_name in snippet_frame_names:
            frame = Image.open(os.path.join(frame_path, frame_name))
            frame = transforms.ToTensor()(frame)
            snippet_frames.append(frame.unsqueeze(0))
        snippet_frames = torch.cat(snippet_frames, dim=0)
        return test_transforms(snippet_frames)

    def get_snippet_audio(start_time, end_time, audio_file_path, sample_rate=22050):
        try:
            snippet_audio, sr = torchaudio.load(audio_file_path, frame_offset=int(start_time * sample_rate), num_frames=int((end_time - start_time) * sample_rate))
            snippet_audio = snippet_audio - snippet_audio.mean()

            # Define the window size and ensure it's within the bounds
            window_size = int(0.025 * sr)  # 25ms window size
            window_shift = int(0.01 * sr)  # 10ms window shift

            if snippet_audio.shape[1] < window_size:
                # Pad the audio snippet to ensure the minimum length for fbank
                pad_len = window_size - snippet_audio.shape[1]
                snippet_audio = F.pad(snippet_audio, (0, pad_len), "constant")

            fbank = torchaudio.compliance.kaldi.fbank(snippet_audio, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                      window_type='hanning', num_mel_bins=pretrain_cfg.audio_melbins, dither=0.0, frame_shift=pretrain_cfg.audio_frame_shift)

            src_length = fbank.shape[0]
            target_length = 512
            pad_len = target_length - src_length % target_length
            fbank = torch.nn.ZeroPad2d((0, 0, 0, pad_len))(fbank)
            total_slice_num = fbank.shape[0] // target_length
            total_slice_num = list(range(total_slice_num))
            total_slice_num = split(total_slice_num, 1)
            sample_idx = [i[(len(i) + 1) // 2 - 1] for i in total_slice_num]
            fbank = torch.stack([fbank[i * target_length: (i + 1) * target_length] for i in sample_idx], dim=0).permute(0, 2, 1)
            fbank = (fbank - pretrain_cfg.audio_mean) / (pretrain_cfg.audio_std * 2)
            return fbank
        except Exception as e:
            print(f"Failed to process audio snippet from {start_time} to {end_time} seconds: {e}")
            return None

    captions = {}
    for start_idx in range(0, len(frames), 16):  # 16프레임 간격으로 슬라이딩 윈도우 적용
        end_idx = start_idx + 32
        if end_idx > len(frames):
            end_idx = len(frames)
        video_pixels = get_snippet_frames(start_idx, end_idx, fps_frame_dir, sample_num=32)
        start_time = start_idx / 24.0
        end_time = end_idx / 24.0
        fbank = get_snippet_audio(start_time, end_time, audio_file_path)
        
        if fbank is None:
            continue
        
        batch = {
            'ids': None,
            'txt_tokens': None,
            'video_pixels': video_pixels.unsqueeze(0).cuda(),
            'audio_spectrograms': fbank.unsqueeze(0).cuda(),
            'ids_txt': None,
            'sample_num': None
        }
        evaluation_dict = model(batch, 'cap%tva', compute_loss=False)
        sents = evaluation_dict['generated_sequences_t_va']
        sents = get_model_attr(model, 'decode_sequence')(sents.data)
        
        if video_path not in captions:
            captions[video_path] = []
        captions[video_path].extend(sents)
    
    return captions

all_captions = {}
for video_path in video_files:
    video_captions = process_video(video_path)
    all_captions.update(video_captions)

output_json_path = './inference2.json'
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(all_captions, f, ensure_ascii=False, indent=4)

# Print confirmation
print(f"Captions saved to {output_json_path}")
