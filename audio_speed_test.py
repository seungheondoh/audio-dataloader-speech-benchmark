import os
import time
import torch
import torchaudio
# import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import subprocess
from typing import List, Tuple, Dict
import pandas as pd
import multiprocessing
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable

def run_loader(args):
    loader_func, file_path = args
    start_time = time.time()
    _ = loader_func(file_path)
    return time.time() - start_time
    
class AudioDataset(Dataset):
    def __init__(self, root_dir: str, loader_func: Callable):
        self.root_dir = Path(root_dir)
        self.audio_files = list(os.listdir(self.root_dir))  # Adjust extension as needed
        self.loader_func = loader_func

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.root_dir, self.audio_files[idx])
        audio_data, sample_rate = self.loader_func(audio_path)
        return audio_data

class AudioLoadingTest:
    def __init__(self, storage_paths: Dict[str, str]):
        """
        storage_paths: Dictionary containing paths for different storage types
        e.g., {'hdd': '/path/to/hdd', 'ssd': '/path/to/ssd', 'nfs': '/path/to/nfs'}
        """
        self.storage_paths = storage_paths
        self.results = {}

    def load_audio_torchaudio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate

    def load_audio_ffmpeg(self, file_path: str) -> Tuple[np.ndarray, int]:
        cmd = [
            'ffmpeg',
            '-i', file_path,
            '-f', 'f32le',
            '-acodec', 'pcm_f32le',
            '-ar', '44100',
            '-ac', '1',
            '-'
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio_data, _ = proc.communicate()
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        return audio_array, 44100

    def load_audio_soundfile(self, file_path: str) -> Tuple[np.ndarray, int]:
        data, sample_rate = sf.read(file_path)
        return data, sample_rate

    def load_audio_librosa(self, file_path: str) -> Tuple[np.ndarray, int]:
        data, sample_rate = librosa.load(file_path, sr=None) # no resampling
        return data, sample_rate

    def test_single_file(self, file_path: str, n_iterations: int = 100) -> Dict:
        loaders = {
            'torchaudio': self.load_audio_torchaudio,
            'ffmpeg': self.load_audio_ffmpeg,
            'librosa': self.load_audio_librosa
        }
        results = {}
        
        with multiprocessing.Pool() as pool:
            for loader_name, loader_func in loaders.items():
                args = [(loader_func, file_path) for _ in range(n_iterations)]
                times = pool.map(run_loader, args)
                
                results[loader_name] = {
                    'mean': np.mean(times),
                    'std': np.std(times), 
                    'min': np.min(times),
                    'max': np.max(times)
                }
        
        return results

    def run_storage_comparison(self, filename: str):
        results = {}
        for storage_type, path in self.storage_paths.items():
            file_path = os.path.join(path, filename)
            results[storage_type] = self.test_single_file(file_path)
        self.results['single_file'] = results
        return results

    def visualize_results(self, test_name: str):
        results = self.results[test_name]
        
        # Create DataFrame for easier plotting
        data = []
        for storage_type, loader_results in results.items():
            for loader_name, metrics in loader_results.items():
                data.append({
                    'Storage': storage_type,
                    'Loader': loader_name,
                    'Time (s)': metrics['mean'],
                    'Std': metrics['std']
                })
        df = pd.DataFrame(data)

        # Create grouped bar plot
        plt.figure(figsize=(12, 6))
        bar_width = 0.2
        index = np.arange(len(df['Storage'].unique()))
        
        for i, loader in enumerate(df['Loader'].unique()):
            data = df[df['Loader'] == loader]
            bars = plt.bar(index + i * bar_width, 
                   data['Time (s)'],
                   bar_width,
                   label=loader,
                   yerr=data['Std'])
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')

        plt.xlabel('Storage Type')
        plt.ylabel('Time (seconds)')
        plt.title(f'Audio Loading Performance - {test_name}')
        plt.xticks(index + bar_width * 1.5, df['Storage'].unique())
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'audio_loading_performance_{test_name}.png')
        plt.close()

    def write_audio_test(self, file_path: str, data: np.ndarray, sample_rate: int):
        # Test writing with different libraries
        write_results = {}
        
        # Torchaudio
        start_time = time.time()
        torchaudio.save(file_path + '_torchaudio.wav', 
                       torch.from_numpy(data).unsqueeze(0), 
                       sample_rate)
        write_results['torchaudio'] = time.time() - start_time
        
        # Soundfile
        start_time = time.time()
        sf.write(file_path + '_soundfile.wav', data, sample_rate)
        write_results['soundfile'] = time.time() - start_time
        
        # Librosa
        start_time = time.time()
        librosa.output.write_wav(file_path + '_librosa.wav', data, sample_rate)
        write_results['librosa'] = time.time() - start_time
        
        return write_results

    def resample_audio_test(self, data: np.ndarray, orig_sr: int, target_sr: int):
        resample_results = {}
        
        # Torchaudio
        start_time = time.time()
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        _ = resampler(torch.from_numpy(data).unsqueeze(0))
        resample_results['torchaudio'] = time.time() - start_time
        
        # Librosa
        start_time = time.time()
        _ = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
        resample_results['librosa'] = time.time() - start_time
        
        return resample_results 