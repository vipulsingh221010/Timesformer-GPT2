import av
import os
import json
import numpy as np
import torch
from transformers import VideoMAEImageProcessor,AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# load pretrained processor, tokenizer, and model
image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-huge-finetuned-kinetics")
#image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

# load video
test_set_dir = "/home/intern/interndata/vipul/autotagger"
result={}
with open('/home/intern/interndata/vipul/autotagger_links.json') as file:
    data = json.load(file)
x=0
for class_folder in os.listdir(test_set_dir):
    class_folder_path=os.path.join(test_set_dir,class_folder)

    for video_file in os.listdir(class_folder_path):
        try:


            video_name=os.path.splitext(video_file)[0]
            if 1>0:
                print(x)
                x+=1
            #ground_truth_labels[video_name]=class_folder
                video_path = os.path.join(class_folder_path,video_file)

        # video clip consists of 300 frames (10 seconds at 30 FPS)

                container = av.open(video_path)


                # extract evenly spaced frames from video
                seg_len = container.streams.video[0].frames
                clip_len = model.config.encoder.num_frames
                indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
                frames = []
                container.seek(0)
                for i, frame in enumerate(container.decode(video=0)):
                    if i in indices:
                        frames.append(frame.to_ndarray(format="rgb24"))

                # generate caption
                gen_kwargs = {
                    "min_length": 10,
                    "max_length": 50,
                    "num_beams": 8,
                }
                pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
                tokens = model.generate(pixel_values, **gen_kwargs)
                caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
                print(data[video_name]['url'])
                print(caption)
                result[video_name]={'caption':caption,'video_link':data[video_name]['url'],'truth_label':class_folder}
        except Exception as e:
            print(f"Error processing video '{video_file}': {str(e)}")

with open('Videomae_TimeSformer-GPT2_adobe_stock.json', 'w') as file:
    # Write the dictionary to the file
    json.dump(result, file)

