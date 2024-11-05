# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from torch.utils.data import Dataset
from speakerlab.utils.fileio import load_data_csv
import os
import torch
from torch.utils.data import Dataset
import pickle



class BaseSVDataset(Dataset):
    def __init__(self, data_file: str, preprocessor: dict, 
                 feature_map_file='/home/wangjuntao_p/project/demo/dataset/feature_map.pkl', 
                 feature_folder='/home/wangjuntao_p/project/demo/dataset/feature'):
        self.data_points = self.read_file(data_file)
        self.preprocessor = preprocessor
        self.feature_folder = feature_folder
        self.feature_map_file = feature_map_file
        self.feature_map = self.load_feature_map()

    def __len__(self):
        return len(self.data_points)


    def get_feature_path(self, wav_path):
        # 根据 wav 路径生成 feat 路径
        return wav_path.replace('raw_data/voxceleb1/dev/wav', 'feature').replace('.wav', '.feat')

    def load_feature_map(self):
        # 从磁盘加载特征映射文件
        if os.path.exists(self.feature_map_file):
            with open(self.feature_map_file, 'rb') as f:
                return pickle.load(f)
        else:
            return {}

    def save_feature_map(self):
        # 将特征映射保存到磁盘
        with open(self.feature_map_file, 'wb') as f:
            pickle.dump(self.feature_map, f)

class WavSVDataset(BaseSVDataset):

    def __getitem__(self, index):
        data = self.get_data(index)
        wav_path = data['path']
        spk = data['spk']
        feature_path = self.get_feature_path(wav_path)

        # 检查特征是否已存在
        if feature_path in self.feature_map:
            data = torch.load(self.feature_map[feature_path])
            feat = data['feature']
            spkid = data['spkid']
        else:
            wav, speed_index = self.preprocessor['wav_reader'](wav_path)
            spkid = self.preprocessor['label_encoder'](spk, speed_index)
            # 加噪声
            wav = self.preprocessor['augmentations'](wav)
            # 提取特征
            feat = self.preprocessor['feature_extractor'](wav)
            # 保存特征到文件
            if not os.path.exists(os.path.dirname(feature_path)):
                os.makedirs(os.path.dirname(feature_path))
            torch.save({'feature': feat, 'spkid': spkid}, feature_path)
            # 更新特征映射
            self.feature_map[wav_path] = feature_path
            self.save_feature_map()

        return feat, spkid

    def get_data(self, index):
        if not hasattr(self, 'data_keys'):
            self.data_keys = list(self.data_points.keys())
        key = self.data_keys[index]
        return self.data_points[key]

    def read_file(self, data_file):
        return load_data_csv(data_file)


# class BaseSVDataset(Dataset):
#     def __init__(self, data_file: str,  preprocessor: dict):
#         self.data_points = self.read_file(data_file)
#         self.preprocessor = preprocessor

#     def __len__(self):
#         return len(self.data_points)


# class WavSVDataset(BaseSVDataset):

#     def __getitem__(self, index):
#         data = self.get_data(index)
#         wav_path = data['path']
#         print("-----------")
#         print(wav_path)
#         print("-----------")
#         spk = data['spk']
#         wav, speed_index = self.preprocessor['wav_reader'](wav_path)
#         spkid = self.preprocessor['label_encoder'](spk, speed_index)
#         # 随机选取混响和噪声进行加噪
#         wav = self.preprocessor['augmentations'](wav)
#         # 提取特征
#         feat = self.preprocessor['feature_extractor'](wav)

#         return feat, spkid

#     def get_data(self, index):
#         if not hasattr(self, 'data_keys'):
#             self.data_keys = list(self.data_points.keys())
#         key = self.data_keys[index]

#         return self.data_points[key]

#     def read_file(self, data_file):
#         return load_data_csv(data_file)





