import json
import math
import pandas as pd
from tqdm import tqdm
import clip
import ffmpeg
import numpy as np
import torch as th 
from narration_ASR_whisper import ASR_whisper
import whisper
import torch.nn.functional as F



def _get_video_dim(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
        None,
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    num, denum = video_stream["avg_frame_rate"].split("/")
    frame_rate = int(num) / int(denum)
    return height, width, frame_rate


def _get_output_dim(size,h, w):
    if isinstance(size, tuple) and len(size) == 2:
        return size
    elif h >= w:
        return int(h * size / w), size
    else:
        return size, int(w * size / h)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = th.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = th.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor

class Preprocessing(object):
    def __init__(self):
        self.norm = Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    def __call__(self, tensor):
        tensor = tensor / 255.0
        tensor = self.norm(tensor)
        return tensor
    

# create csv file with video_id, duration,caption, start, end, asr string, asr_start, asr end, features as columns
csv_df = pd.DataFrame(columns=['video_id', 'duration', 'caption', 'start', 'end', 'asr_string', 'asr_start', 'asr_end', 'features'])



video_path = '../../mnt/welles/scratch/datasets/YouCook2/YouCookII/raw_videos/training/101/0O4bxhpFX9o.webm'
seg_path = '/mnt/welles/scratch/datasets/YouCook2/YouCookII/annotations/youcookii_annotations_trainval.json'

print('Reading segments json file')
with open(seg_path) as f:
    segs = json.load(f)

batch_size = 128
feature_dim = 768
l2_normalize = 0
half_precision = 1

print('Reading segments info')
start_time_list = []
end_time_list = []
duration_list = []
id_list = []
sentence_list = []
segments_info = segs['database']['0O4bxhpFX9o']['annotations']
for id in range(0,len(segments_info)):
    start = segments_info[id]['segment'][0]
    start_time_list.append(start)
    end = segments_info[id]['segment'][1]
    end_time_list.append(end)
    duration = end - start
    duration_list.append(duration)
    id_list.append(id)
    sentence = segments_info[id]['sentence']
    sentence_list.append(sentence)


print('Loading CLIP model')
preprocess = Preprocessing()
model, _ = clip.load("ViT-L/14", device="cuda:1")
model.eval()
model = model.cuda()

print('Loading ASR model')
# load the model on gpu 1
model_st = whisper.load_model('large', device='cuda:0')

print('Loading video')
h, w, fr = _get_video_dim(video_path)
if fr < 1:
    print("Corrupted Frame Rate: {}".format(video_path))
else:
    size = 224
    framerate = 1
    height, width =_get_output_dim(size,h, w)

    cmd = (
        ffmpeg.input(video_path)
        .filter("fps", fps= framerate)
        .filter("scale", width, height)
    )

    x = int((width - size) / 2.0)
    y = int((height - size) / 2.0)
    cmd = cmd.crop(x, y, size, size)

    out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
        capture_stdout=True, quiet=True
    )

    height, width = size, size
    raw_video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    raw_video = th.from_numpy(raw_video.astype("float32"))
    raw_video = raw_video.permute(0, 3, 1, 2)

print('Extracting features')
features_list = []
video_id_list = []
asr_string_list = []
asr_start_list = []
asr_end_list = []
for i in tqdm(range(0,len(id_list)), desc='Extracting features', total=len(id_list)):
    video = raw_video[int(start_time_list[i]*framerate):int(end_time_list[i]*framerate)]

    with th.no_grad():
        input_file = video
        video_id = video_path.split('/')[-1].split('.')[0]+'/'+str(id_list[i])
        video_id_list.append(video_id)
        asr_string, asr_start, asr_end = ASR_whisper(video_path, segs, i, model_st)
        asr_string_list.append(asr_string[0])
        asr_start_list.append(asr_start)
        asr_end_list.append(asr_end)
        

        video = preprocess(video)
        n_chunk = len(video)
        features = th.cuda.FloatTensor(n_chunk, feature_dim).fill_(0)
        n_iter = int(math.ceil(n_chunk / float(batch_size)))
        for i in tqdm(range(n_iter)):
            min_ind = i * batch_size
            max_ind = (i + 1) * batch_size
            video_batch = video[min_ind:max_ind].cuda()
            batch_features = model.encode_image(video_batch)
            if l2_normalize:
                batch_features = F.normalize(batch_features, dim=1)
            features[min_ind:max_ind] = batch_features
        features = features.cpu().numpy()
        if half_precision:
            features = features.astype("float16")
        features_list.append(features)

# fill the csv file
csv_df['video_id'] = video_id_list
csv_df['duration'] = duration_list
csv_df['caption'] = sentence_list
csv_df['start'] = start_time_list
csv_df['end'] = end_time_list
csv_df['asr_string'] = asr_string_list
csv_df['asr_start'] = asr_start_list
csv_df['asr_end'] = asr_end_list
csv_df['features'] = features_list

# save the csv file
csv_df.to_csv('youcookii_segments_inf.csv', index=False)
