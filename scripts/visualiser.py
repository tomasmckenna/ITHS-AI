import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# def get_file_paths(config):
#     if os.name == 'nt':  # Windows
#         data_path = config['output_file_path']['windows']
#         output_dir = os.path.dirname(data_path)
#     elif os.uname().sysname == 'Darwin':  # macOS
#         data_path = config['output_file_path']['mac']
#         output_dir = os.path.dirname(data_path)
#     else:  # Linux
#         data_path = config['output_file_path']['linux']
#         output_dir = os.path.dirname(data_path)

    return data_path, output_dir

def save_plot(plt_object, filename, output_dir):
    file_path = os.path.join(output_dir, filename)
    plt_object.savefig(file_path)
    print(f"Plot saved to {file_path}")

if __name__ == "__main__":
    config_path = (
        "/Users/tomas/Documents/GitHub/AImed/config/config.yaml"
        if os.uname().sysname == 'Darwin' else
        "/home/tomas/GitHub/AImed/config/config.yaml"
    )

    config = '/Users/tomas/Documents/GitHub/ITHS-AI-Project/config/config.yaml'
    data_path ='/Users/tomas/Documents/GitHub/ITHS-AI-Project/data/df.xlsx'
    output_dir = '/Users/tomas/Documents/GitHub/ITHS-AI-Project/data/'

    try:
        df = pd.read_excel(data_path, engine='openpyxl')  # Use openpyxl for reading Excel files
        print(f"Data successfully loaded from {data_path}")
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        exit()

    # Configure Seaborn
    sns.set(style="whitegrid")

    # 1. Distribution of Travel Durations
    plt.figure(figsize=(10, 6))
    sns.histplot(df['DurationMin'], bins=50, kde=True)
    plt.title('Distribution of Travel Durations')
    plt.xlabel('Travel Duration (minutes)')
    plt.ylabel('Frequency')
    save_plot(plt, "distribution_of_travel_durations.png", output_dir)
    plt.show()

    # 2. Box Plot to Identify Outliers in Travel Durations
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['DurationMin'])
    plt.title('Box Plot of Travel Durations')
    plt.xlabel('Travel Duration (minutes)')
    save_plot(plt, "box_plot_travel_durations.png", output_dir)
    plt.show()

    # 3. Scatter Plot: Distance vs. Duration
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='DistanceM', y='DurationMin', data=df)
    plt.title('Travel Duration vs. Travel Distance')
    plt.xlabel('Travel Distance (meters)')
    plt.ylabel('Travel Duration (minutes)')
    save_plot(plt, "scatter_distance_vs_duration.png", output_dir)
    plt.show()

    # 4. Bar Plot of Average Travel Duration by Care Episode
    plt.figure(figsize=(14, 8))
    average_duration_by_episode = df.groupby('CareEpisodeID')['DurationMin'].mean().reset_index()
    sns.barplot(x='CareEpisodeID', y='DurationMin', data=average_duration_by_episode)
    plt.xticks(rotation=90)
    plt.title('Average Travel Duration by Care Episode')
    plt.xlabel('Care Episode ID')
    plt.ylabel('Average Travel Duration (minutes)')
    save_plot(plt, "bar_plot_average_duration_by_episode.png", output_dir)
    plt.show()

    # 5. Distribution of Travel Distances
    plt.figure(figsize=(10, 6))
    sns.histplot(df['DistanceM'], bins=50, kde=True)
    plt.title('Distribution of Travel Distances')
    plt.xlabel('Travel Distance (meters)')
    plt.ylabel('Frequency')
    save_plot(plt, "distribution_of_travel_distances.png", output_dir)
    plt.show()
