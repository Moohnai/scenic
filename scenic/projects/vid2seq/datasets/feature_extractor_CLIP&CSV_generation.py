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
from moviepy.editor import AudioFileClip
import scipy.signal as sps



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
csv_df = pd.DataFrame(columns=[
    'video_id', 
    'duration', 
    'caption', 
    'start', 
    'end', 
    'asr_string', 
    'asr_start', 
    'asr_end', 
    # 'features'
])



video_path = ['../../mnt/welles/scratch/datasets/YouCook2/YouCookII/raw_videos/validation/405/fn9anlEL4FI.webm']
video_id = video_path[0].split('/')[-1].split('.')[0]
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
seg_list = []
sentence_list = []
segments_info = segs['database'][video_id]['annotations']
for id in range(0,len(segments_info)):
    start = segments_info[id]['segment'][0]
    # second to microsecond
    start = start * 1000000
    start_time_list.append(start)
    end = segments_info[id]['segment'][1]
    end = end * 1000000
    end_time_list.append(end)
    seg_list.append(id)
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
h, w, fr = _get_video_dim(video_path[0])
if fr < 1:
    print("Corrupted Frame Rate: {}".format(video_path[0]))
else:
    size = 224
    framerate = 1
    height, width =_get_output_dim(size,h, w)

    cmd = (
        ffmpeg.input(video_path[0])
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

features_list = []
video_id_list = []
asr_string_list = []
asr_start_list = []
asr_end_list = []
duration = (raw_video.shape[0]/framerate)* 1000000


pbar = tqdm(range(0,len(video_path)), total=len(video_path))
for i in pbar:
    # video = raw_video[int(start_time_list[i][0]*framerate):int(end_time_list[i][0]*framerate)]
    video = raw_video
    duration_list.append([int((video.shape[0]/framerate)* 1000000)])
    video_id = video_path[i].split('/')[-1].split('.')[0]
    video_id_list.append([video_id])

    npy_output_file = f'/home/mona/scenic/scenic/projects/vid2seq/data/youcook2/{video_id}.npy'

    # read audio data from video
    pbar.set_description("Loading Audio for {}".format(video_id))
    audio = AudioFileClip(video_path[i])
    # convert to numpy array
    audio_array = audio.to_soundarray(fps=audio.fps)
    # downsample to 16kHz
    new_rate = 16000
    number_of_samples = round(len(audio_array) * float(new_rate) / audio.fps)
    audio_array = sps.resample(audio_array, number_of_samples)
    # convert to single channel
    audio_array = np.mean(audio_array, axis=1)

    pbar.set_description("Processing Whisper for {}".format(video_id))
    # for j in tqdm(range(0,len(seg_list)), total=len(seg_list), leave=False, desc="Processing Whisper"):
    with th.no_grad():
        # input_file = video
        # video_id_list.append([video_id])
        # asr_string, asr_start, asr_end = ASR_whisper(audio_array, segs, j, model_st, video_id, new_rate)
        # # add start to asr_start and end to asr_end
        # asr_start = asr_start + int(start_time_list[j])
        # # second to microsecond
        # asr_start = asr_start 
        # asr_end = asr_end + int(start_time_list[j])
        # asr_end = asr_end 
        # floor the start and end times to the nearest second
        # asr_start = math.floor(asr_start)
        # asr_end = math.floor(asr_end)
        # asr_string_list.append(asr_string[0])
        # asr_start_list.append(asr_start)
        # asr_end_list.append(asr_end)
        asr_string_list, asr_start_list, asr_end_list = ASR_whisper(audio_array, model_st, new_rate)
        # second to microsecond
        asr_start_list = [int(x * 1000000) for x in asr_start_list]
        asr_end_list = [int(x * 1000000) for x in asr_end_list]
        
    pbar.set_description("Processing CLIP for {}".format(video_id))
    with th.no_grad():
        video = preprocess(video)
        n_chunk = len(video)
        features = th.cuda.FloatTensor(n_chunk, feature_dim).fill_(0)
        n_iter = int(math.ceil(n_chunk / float(batch_size)))
        for i in tqdm(range(n_iter), total=n_iter, leave=False, desc="Processing CLIP"):
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
        features_list.append(features.tolist())
        np.save(npy_output_file, features)

# fill the csv file
csv_df['video_id'] = video_id_list
csv_df['duration'] = duration_list
csv_df['caption'] = str(sentence_list)
csv_df['start'] = str(start_time_list)
csv_df['end'] = str(end_time_list)
csv_df['asr_string'] = str(asr_string_list)
csv_df['asr_start'] = str(asr_start_list)
csv_df['asr_end'] = str(asr_end_list)
# csv_df['features'] =  features_list

# save the csv file
csv_df.to_csv('/home/mona/scenic/scenic/projects/vid2seq/data/youcook2/youcookii_1vid_inf.csv', index=False)
