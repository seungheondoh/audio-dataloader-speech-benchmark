import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

class AudioTestVisualizer:
    def __init__(self, results_file: str):
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        # Create plots directory if it doesn't exist
        self.plots_dir = Path('plots')
        self.plots_dir.mkdir(exist_ok=True)

    def plot_single_file_comparison(self):
        results = self.results['single_file']
        
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
            plt.bar(index + i * bar_width, 
                   data['Time (s)'],
                   bar_width,
                   label=loader,
                   yerr=data['Std'])

        plt.xlabel('Storage Type')
        plt.ylabel('Time (seconds)')
        plt.title('Single File Audio Loading Performance')
        plt.xticks(index + bar_width * 1.5, df['Storage'].unique())
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'single_file_performance.png')
        plt.close()

    def plot_dataloader_comparison(self):
        results = self.results['dataloader']
        
        # Create DataFrame
        data = []
        for storage_type, loader_results in results.items():
            for loader_name, time_taken in loader_results.items():
                data.append({
                    'Storage': storage_type,
                    'Loader': loader_name,
                    'Time (s)': time_taken
                })
        df = pd.DataFrame(data)

        # Create grouped bar plot
        plt.figure(figsize=(12, 6))
        bar_width = 0.2
        index = np.arange(len(df['Storage'].unique()))
        
        for i, loader in enumerate(df['Loader'].unique()):
            data = df[df['Loader'] == loader]
            plt.bar(index + i * bar_width, 
                   data['Time (s)'],
                   bar_width,
                   label=loader)

        plt.xlabel('Storage Type')
        plt.ylabel('Time (seconds)')
        plt.title('DataLoader Batch Loading Performance')
        plt.xticks(index + bar_width * 1.5, df['Storage'].unique())
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'dataloader_performance.png')
        plt.close()

    def plot_all(self):
        self.plot_single_file_comparison()
        self.plot_dataloader_comparison()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file', help='Path to the JSON results file')
    args = parser.parse_args()

    visualizer = AudioTestVisualizer(args.results_file)
    visualizer.plot_all() 