import logging, os, time
import json

class ActionLogger():
    def __init__(self, path):
        self.logger = logging.getLogger('ActionLogger')
        
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(fmt='%(message)s')

        # stream(console) handler 객체 생성
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # file handler 객체 생성
        os.makedirs(path, exist_ok=True)
        file_handler = logging.FileHandler(filename=path+'/action.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logs = {}
        self.personal_log = {}

    def add_action(self, input):
        timestamp, frame_num, id, action, x1, y1, x2, y2 = input

        if timestamp not in self.logs:
            self.logs[timestamp] = {}

        if id not in self.logs[timestamp]:
            self.logs[timestamp][id] = {'pos': [x1, y1, x2, y2], 'actions': []}

        splited = action.split(' ')

        self.logs[timestamp][id]['actions'].append({'action': ' '.join(splited[:-1]), 'confidence': float(splited[-1])})


    def save(self):
        # now = time.localtime()
        # self.personal_log['time'] = f'{now.tm_year:04d}-{now.tm_mon:02d}-{now.tm_mday:02d} {now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}'
        # self.personal_log['activities'] = self.activities
        # self.personal_log['fn'] = frame_num
        # self.personal_log['timestamp'] = timestamp
        self.logger.info(json.dumps(self.logs))

class EmotionLogger():
    def __init__(self, path):
        self.logger = logging.getLogger('EmotionLogger')
        
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(fmt='%(message)s')

        # stream(console) handler 객체 생성
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # file handler 객체 생성
        os.makedirs(path, exist_ok=True)
        file_handler = logging.FileHandler(filename=path+'/emotion.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log(self, face_boxes_and_emotions_by_timestamp):
        self.logger.info(json.dumps(face_boxes_and_emotions_by_timestamp))

class ActionPredictionLogger:
    def __init__(self, path):
        self.logger = logging.getLogger('ActionPredictionLogger')
        
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(fmt='%(message)s')

        # stream(console) handler 객체 생성
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # file handler 객체 생성
        os.makedirs(path, exist_ok=True)
        file_handler = logging.FileHandler(filename=path+'/action_prediction.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log(self, input):
        rounded_mills, frame_num, id, caption, x1, y1, x2, y2 = input

        output_log = f'timestamp: {rounded_mills}, frame_num: {frame_num}, id: {id}, action: {caption}'
        self.logger.info(output_log)
