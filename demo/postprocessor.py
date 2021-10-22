import pandas as pd
import numpy as np
import json

class LogReconstructer:
    def __init__(self, action_logs_path, emotion_logs_path):
        with open(action_logs_path, 'r') as f:
            self.action_logs = json.load(f)

        with open(emotion_logs_path, 'r') as f:
            self.emotion_logs = json.load(f)

        self.synchronized_action_emotion_timestamps = {}
        self.emotion_dataframes_by_id = {}
        self.action_dataframes_by_id = {}
        self.composed_dataframes_by_id = {}

    def get_emotions_id(self):
        pivot = 0
        action_timestamps = list(self.action_logs.keys())
        emotion_timestamps = list(self.emotion_logs.keys())

        start = ''
        for i, emotion_timestamp in enumerate(emotion_timestamps):
            if action_timestamps[0] <= emotion_timestamp:
                start = i
                break

        for emotion_timestamp in emotion_timestamps[start:]:
            for k, action_timestamp in enumerate(action_timestamps[pivot:]):
                if abs(int(action_timestamp)-int(emotion_timestamp)) < 15:
                    self.synchronized_action_emotion_timestamps[emotion_timestamp] = action_timestamp

                    osizes = []
                    fsizes = []

                    action_log = self.action_logs[action_timestamp] # {obj_id: {'pos':..., 'actions':[...]}, obj_id: ...}
                    emotion_log = self.emotion_logs[emotion_timestamp] # [{'pos':..., 'emotions':[...]}, ...]

                    for obj_id, info in action_log.items():
                        l, t, r, b = info['pos']
                        size = (r-l)*(b-t)
                        osizes.append((obj_id, size))

                    for j, face in enumerate(emotion_log):
                        l, t, r, b = face['pos']
                        size = (r-l)*(b-t)
                        fsizes.append((j, size))

                    osizes.sort(key=lambda x:x[1])
                    fsizes.sort(key=lambda x:x[1])

                    row_nums = len(osizes)
                    col_nums = len(fsizes)

                    matrix = [[0]*col_nums for _ in range(row_nums)]

                    for i in range(row_nums):
                        l, t, r, b = action_log[osizes[i][0]]['pos']

                        for j in range(col_nums):
                            fsize = fsizes[j] # (idx, size)
                            fl, ft, fr, fb = emotion_log[fsize[0]]['pos']

                            # overlapped grid
                            ovl = max(l, fl)
                            ovt = max(t, ft)
                            ovr = min(r, fr)
                            ovb = min(b, fb)

                            overlapped_size = (ovr-ovl)*(ovb-ovt)

                            matrix[i][j] = overlapped_size/fsize[1]
                    
                    for j in range(col_nums):
                        col = [matrix[i][j] for i in range(row_nums)]
                        overlapped_size_max = max(col)

                        for i, overlapped_size in enumerate(col):
                            if overlapped_size == overlapped_size_max and j <= i:
                                emotion_log[fsizes[j][0]]['id'] = osizes[i][0]
                                break

                    pivot += k
                    break

    def get_emotion_dataframes(self):
        emotion_dataframe_columns = ['timestamp', 'age_range', 'age_avg', 'best_emotion', 'best_emotion_conf']
        emotion_labels = ['CALM', 'HAPPY', 'SAD', 'ANGRY', 'FEAR', 'SURPRISED', 'DISGUSTED', 'CONFUSED']
        emotion_nums = len(emotion_labels)
        emotion_dataframe_columns += emotion_labels

        for timestamp, emotion_log in self.emotion_logs.items():
            for face in emotion_log: 
                if 'id' in face:
                    obj_id = face['id']

                    if obj_id not in self.emotion_dataframes_by_id:
                        self.emotion_dataframes_by_id[obj_id] = pd.DataFrame()

                    best_emotion = face['emotions'][0]['Type']
                    best_emotion_conf = face['emotions'][0]['Confidence']
                    emotion_confs = [np.nan]*emotion_nums

                    age_range = face['age_range']
                    age_avg = round(np.mean(age_range), 1)

                    for e in face['emotions']:
                        emotion_confs[emotion_labels.index(e['Type'])] = e['Confidence']

                    
                    data = [
                        self.synchronized_action_emotion_timestamps[timestamp],
                        age_range,
                        age_avg,
                        best_emotion,
                        best_emotion_conf
                    ]
                    data += emotion_confs
                    self.emotion_dataframes_by_id[obj_id] = self.emotion_dataframes_by_id[obj_id].append(
                        pd.DataFrame(data=[data], columns=emotion_dataframe_columns)
                    )

        for obj_id in self.emotion_dataframes_by_id.keys():
            self.emotion_dataframes_by_id[obj_id] = self.emotion_dataframes_by_id[obj_id].reset_index(drop=True)

    def get_action_dataframes(self):
        action_dataframe_columns = ['timestamp', 'pos', 'actions']

        for timestamp, action_log in self.action_logs.items():
            for obj_id, info in action_log.items():
                if obj_id not in self.action_dataframes_by_id:
                    self.action_dataframes_by_id[obj_id] = pd.DataFrame()

                data = [timestamp, info['pos'], info['actions']]
                self.action_dataframes_by_id[obj_id] = self.action_dataframes_by_id[obj_id].append(
                    pd.DataFrame(data=[data], columns=action_dataframe_columns)
                )

        for obj_id in self.action_dataframes_by_id:
            self.action_dataframes_by_id[obj_id] = self.action_dataframes_by_id[obj_id].reset_index(drop=True)

    def get_composed_dataframes(self):
        for obj_id, adf in self.action_dataframes_by_id.items():
            if obj_id in self.emotion_dataframes_by_id:
                edf = self.emotion_dataframes_by_id[obj_id]
                df = pd.merge(left=adf, right=edf, how='left', on='timestamp')

                self.composed_dataframes_by_id[obj_id] = df

    def save(self):
        appeared_by_role = {'A': False, 'C': False}

        for df in self.composed_dataframes_by_id.values():
            if 10 < len(df):
                obj_role = 'C' if df.age_avg.mean() < 10 else 'A'

                if not appeared_by_role[obj_role]:
                    df.to_csv(f'{obj_role}.csv')
                    appeared_by_role[obj_role] = True

                    if all(list(appeared_by_role.values())):
                        break