import json
import numpy as np
import torch
import moviepy
from moviepy.editor import AudioFileClip
from transformers import AutoProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import scipy.signal as sps
import whisper

# # load the processor and model on gpus
# processor = AutoProcessor.from_pretrained("openai/whisper-large", torch_device="cuda")
# # processor.to("cuda")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
# model = model.to("cuda")



def ASR_whisper (video_path, segs, i, model):
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
    # break the audio into 30 second chunks
    total_seconds = len(audio_array) / 16000
    audio_chunks = np.array_split(audio_array, 2)
    audio_chunks = []

    # get the segments from the json file
    segments_info = segs['database']['0O4bxhpFX9o']['annotations']
    for id in range(0,len(segments_info)):
        start = segments_info[id]['segment'][0]
        end = segments_info[id]['segment'][1]
        start_frame = int(start * new_rate)
        end_frame = int(end * new_rate)
        audio_chunks.append(audio_array[start_frame:end_frame])


    # ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    # inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
    transcription_list = []
    audio_chunks = [audio_chunks[i]]
    for audio_array in audio_chunks:
        # original huggingface whisper code
        # inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        # input_features = inputs.input_features
        # input_features = input_features.to("cuda")
        # generated_ids = model.generate(inputs=input_features)
        # transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        #####
        with torch.no_grad():
            result = model.transcribe(audio_array.astype(np.float32), language='en', max_initial_timestamp=None)
        transcription = result['text']
        start_asr = result['segments'][0]['start']
        end_asr = result['segments'][-1]['end']
        #####
        
        # print(transcription)
        transcription_list.append(transcription)
    return transcription_list, start_asr, end_asr



if __name__ == '__main__':
    # load youcook2 videoa
    video_path = '../../mnt/welles/scratch/datasets/YouCook2/YouCookII/raw_videos/training/101/0O4bxhpFX9o.webm'

    #load video segments

    seg_path = '/mnt/welles/scratch/datasets/YouCook2/YouCookII/annotations/youcookii_annotations_trainval.json'
    with open(seg_path) as f:
        segs = json.load(f)
        
    model_st = whisper.load_model('large', device='cuda:0')
    ASR_whisper(video_path, segs, 1, model_st)