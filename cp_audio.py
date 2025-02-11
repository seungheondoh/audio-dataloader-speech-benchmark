import os
import shutil
from pydub.utils import mediainfo
import numpy as np
from multiprocessing import Pool, Manager

def process_file(args):
    idx, filename, target_audio_dir, ssd_dir, hdd_dir, ssd2_dir = args
    src = os.path.join(target_audio_dir, filename)
    dst = os.path.join(ssd_dir, filename)
    hdd_dst = os.path.join(hdd_dir, filename)
    ssd2_dst = os.path.join(ssd2_dir, filename)
    info = mediainfo(src)
    duration = float(info['duration'])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.makedirs(os.path.dirname(hdd_dst), exist_ok=True)
    os.makedirs(os.path.dirname(ssd2_dst), exist_ok=True)
    shutil.copy(src, dst)
    shutil.copy(src, hdd_dst)
    shutil.copy(src, ssd2_dst)
    return duration

ssd_dir = "./audio_speed_test"
hdd_dir = "/mnt/seungheond/audio_speed_test" 
ssd2_dir = "/mnt2/audio_speed_test"
target_audio_dir = "/workspace/seungheon/dataset/fma/audio"

files = os.listdir(target_audio_dir)
args = [(idx, f, target_audio_dir, ssd_dir, hdd_dir, ssd2_dir) for idx, f in enumerate(files)]

with Pool() as pool:
    durations = pool.map(process_file, args)
    durations = [d for d in durations if d is not None]
    
# average audio duration:296.74471641
total_duration_hours = sum(durations) / 3600  # Convert seconds to hours
print(f"Total audio duration: {total_duration_hours:.2f} hours")

print(np.mean(durations))