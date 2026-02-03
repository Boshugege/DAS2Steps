# -*- coding: utf-8 -*-

# 使用 Airtag 获得的数据，找出每个人行走对应的时间段，提取对应时间段的 TDMS 数据，保存为 CSV 文件并绘图输出。

import os
import pandas as pd
from datetime import datetime, timedelta
from tdms_reader import *
from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt

def read_csv_start_end_times(folder_path):
    time_ranges = []
    # sort the file list by name
    filelist = sorted(os.listdir(folder_path))
    for file in filelist:
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            start_time = pd.to_datetime(df.iloc[0, 0])  # Beijing time
            end_time = pd.to_datetime(df.iloc[-1, 0])  # Beijing time
            time_ranges.append({
                'file': file,
                'start_time': start_time.strftime('%Y%m%d_%H%M%S'),
                'end_time': end_time.strftime('%Y%m%d_%H%M%S')
            })
    return time_ranges

# Example usage
folder_path = 'D:/tdms/AirtagAlex/AirtagAlex'
csv_time_ranges = read_csv_start_end_times(folder_path)
print(csv_time_ranges)



first_channel = 0
last_channel = 151 # 199
first_time_sample = 0
last_time_sample = 119999

def process_tdms_files(tdms_folder_path, csv_time_ranges):
    
    for time_range in csv_time_ranges:
        data_all = None  # Initialize an empty variable to store all data
        start_time = datetime.strptime(time_range['start_time'], '%Y%m%d_%H%M%S') - timedelta(hours=8)  # Convert to UTC
        end_time = datetime.strptime(time_range['end_time'], '%Y%m%d_%H%M%S') - timedelta(hours=8)  # Convert to UTC
        print(f"Processing CSV: {time_range['file']} from {start_time} to {end_time}")
        for file in sorted(os.listdir(tdms_folder_path)):
            if file.endswith(".tdms"):
                file_time_str = file.split('_UTC_')[1][:17]
                file_time = datetime.strptime(file_time_str, '%Y%m%d_%H%M%S.%f')
                if file_time > start_time - timedelta(minutes=1) and file_time <= end_time:
                    print(f"Matched TDMS file: {file}")
                    ## Process the TDMS file
                    # tdms = TdmsReader(os.path.join(tdms_folder_path, file))
                    # some_data = tdms.get_data(first_channel, last_channel, first_time_sample, last_time_sample)
                    ## cancatetat the data to data_all
                    ## Concatenate data

                    tdms_file = TdmsFile.read(os.path.join(tdms_folder_path, file))

                    # 获取所有通道列表
                    all_channels = []
                    for group in tdms_file.groups():
                        for channel in group.channels():
                            all_channels.append(channel)

                    # 保证通道索引不越界
                    last_ch_idx = min(last_channel, len(all_channels)-1)

                    # 拼接每个通道的数据
                    some_data_list = []
                    for ch_idx in range(first_channel, last_ch_idx+1):
                        ch_data = all_channels[ch_idx][first_time_sample:last_time_sample+1]
                        some_data_list.append(ch_data)

                    if len(some_data_list) == 0:
                        continue  # 没有数据就跳过

                    some_data = np.column_stack(some_data_list)

                    if data_all is None:
                        data_all = some_data
                    else:
                        data_all = np.vstack((data_all, some_data))
        # Save data to a file
        # Assuming each column in data_all corresponds to a fixed time interval
        total_time_samples = data_all.shape[0]
        time_interval = 1/2000.0  # Set the time interval between samples in minutes
        time_axis = np.linspace(0, total_time_samples * time_interval, total_time_samples)

        # Set up channel axis
        channel_axis = np.arange(data_all.shape[1])

        csv_filename = time_range['file'][:-4] + '.csv'
        df = pd.DataFrame(data_all, columns=[f"ch_{i}" for i in range(first_channel, last_ch_idx+1)])
        df.to_csv(csv_filename, index=False)
        print(f"Saved CSV: {csv_filename}")

        # Create the plot
        fig=plt.figure(figsize=(15, 10))
        plt.imshow(data_all.transpose(), aspect='auto', interpolation='none', extent=(time_axis[0], time_axis[-1], channel_axis[0], channel_axis[-1]),cmap='seismic',origin='lower')
        # set caxis
        plt.clim(-5000, 5000)        
        plt.colorbar(label='Amplitude')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Channel Number')
        plt.title(start_time.strftime('%Y-%m-%d %H:%M:%S') + ' to ' + end_time.strftime('%Y-%m-%d %H:%M:%S'))
        plt.show()    
        # save figure
        fig.savefig(time_range['file'][:-4] + '.png')              

# Example usage
tdms_folder_path = 'D:/tdms/20231015_B202'
process_tdms_files(tdms_folder_path, csv_time_ranges)
