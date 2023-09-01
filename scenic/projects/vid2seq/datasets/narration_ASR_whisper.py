import json
import numpy as np
import torch
import moviepy
from moviepy.editor import AudioFileClip
from transformers import AutoProcessor, WhisperForConditionalGeneration
# from datasets import load_dataset
import scipy.signal as sps
import whisper
import math
from tqdm import tqdm
import torch.nn.functional as F


# # load the processor and model on gpus
# processor = AutoProcessor.from_pretrained("openai/whisper-large", torch_device="cuda")
# # processor.to("cuda")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
# model = model.to("cuda")



def ASR_whisper (audio_array, model, new_rate=16000):
    


    # audio_chunks = []

    # # get the segments from the json file
    # segments_info = segs['database'][video_id]['annotations']
    # for id in range(0,len(segments_info)):
    #     start = segments_info[id]['segment'][0]
    #     end = segments_info[id]['segment'][1]
    #     start_frame = int(start * new_rate)
    #     end_frame = int(end * new_rate)
    #     audio_chunks.append(audio_array[start_frame:end_frame])

    # get the whole audio but deviede it into 30s chunks
    audio_chunks = []
    chunk_size = 30 
    audio_len = len(audio_array) / new_rate
    n_iter = int(math.ceil(audio_len / float(chunk_size)))
    start_shift_list = []

    for i in range(n_iter):
        start_chunk = i * chunk_size * new_rate
        end_chunk = (i + 1) * chunk_size * new_rate
        audio_chunk = audio_array[start_chunk:end_chunk]
        audio_chunks.append(audio_chunk)
        start_shift_list.append(i * chunk_size)


    
    # audio_chunks = [audio_array]


    # ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    # inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
    transcription_list = []
    start_asr_list = []
    end_asr_list = []
    for j, audio_array in tqdm(enumerate(audio_chunks), total=len(audio_chunks), leave=False, desc="Processing audio Chunks"):
        # original huggingface whisper code
        # inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        # input_features = inputs.input_features
        # input_features = input_features.to("cuda")
        # generated_ids = model.generate(inputs=input_features)
        # transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        #####
        with torch.no_grad():
            result = model.transcribe(audio_array.astype(np.float32), language='en', max_initial_timestamp=None)
        # transcription = result['text']
        # start_asr = result['segments'][0]['start']
        # end_asr = result['segments'][-1]['end']
        #####
        for segment in result['segments']:
            transcription_list.append(segment['text'])
            start_asr_list.append(segment['start'] + start_shift_list[j])
            end_asr_list.append(segment['end'])
        
        # print(transcription)
        # transcription_list.append(transcription)
    return transcription_list, start_asr_list, end_asr_list



if __name__ == '__main__':
    # load youcook2 videoa
    video_path = '../../mnt/welles/scratch/datasets/YouCook2/YouCookII/raw_videos/validation/405/fn9anlEL4FI.webm'

    #load video segments

    seg_path = '/mnt/welles/scratch/datasets/YouCook2/YouCookII/annotations/youcookii_annotations_trainval.json'
    with open(seg_path) as f:
        segs = json.load(f)
        
    model_st = whisper.load_model('large', device='cuda:0')

    # read audio data from video
    audio = AudioFileClip(video_path)
    # convert to numpy array
    audio_array = audio.to_soundarray(fps=audio.fps)
    # downsample to 16kHz
    new_rate = 16000
    number_of_samples = round(len(audio_array) * float(new_rate) / audio.fps)
    audio_array = sps.resample(audio_array, number_of_samples)
    # convert to single channel
    audio_array = np.mean(audio_array, axis=1)


    ASR_whisper(audio_array, model_st, new_rate=16000)