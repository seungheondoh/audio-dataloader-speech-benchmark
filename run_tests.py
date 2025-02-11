from audio_speed_test import AudioLoadingTest, AudioDataset
from torch.utils.data import DataLoader
import time
import torch
import argparse

def main(args):
    storage_paths = {
        'hdd': args.hdd,
        'ssd': args.ssd, 
        'ssd_hq': args.ssd_hq
    }

    # Initialize test
    test = AudioLoadingTest(storage_paths)

    # Test 1: Single file comparison
    print("Running single file tests...")
    test.run_storage_comparison('7481.mp3')
    test.visualize_results('single_file')
    # Test 2: DataLoadser comparison
    batch_size = 1
    num_workers = 8
    dataloader_results = {}
    for storage_type, path in storage_paths.items():
        print(f"\nTesting DataLoader with {storage_type}...")
        dataloader_results[storage_type] = {}
        
        for loader_name, loader_func in {
            'torchaudio': test.load_audio_torchaudio,
            'ffmpeg': test.load_audio_ffmpeg,
            'librosa': test.load_audio_librosa
        }.items():
            dataset = AudioDataset(path, loader_func)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True
            )
            # Time the entire epoch
            start_time = time.time()
            for batch in dataloader:
                pass
            elapsed_time = time.time() - start_time
            dataloader_results[storage_type][loader_name] = {
                'mean': float(elapsed_time),
                'std': 0.0  # Single run so std is 0
            }
    # Store results and visualize
    test.results['dataloader'] = dataloader_results
    test.visualize_results('dataloader')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio loading speed tests')
    parser.add_argument('--hdd', type=str, default='/mnt/seungheond/audio_speed_test')
    parser.add_argument('--ssd', type=str, default='./audio_speed_test')
    parser.add_argument('--ssd_hq', type=str, default='/mnt2/audio_speed_test')
    args = parser.parse_args()
    main(args) 