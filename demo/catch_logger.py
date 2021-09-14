import logging, os, time
import json

class PersonalLogger():
    def __init__(self, path):
        self.logger = logging.getLogger('PersonalLogger')
        
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(fmt='%(message)s')

        # stream(console) handler 객체 생성
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # file handler 객체 생성
        os.makedirs(path, exist_ok=True)
        file_handler = logging.FileHandler(filename=path+'/person.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.activities = []
        self.personal_log = {}

    def add_person(self, input):
        name, action, x1, y1, x2, y2 = input
        person = {'name': name, 'action': action, 'pos': [x1, y1, x2, y2]}
        self.activities.append(person)

    def log(self, frame_num):
        now = time.localtime()
        self.personal_log['time'] = f'{now.tm_year:04d}-{now.tm_mon:02d}-{now.tm_mday:02d} {now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}'
        self.personal_log['activities'] = self.activities
        self.personal_log['fn'] = frame_num
        self.logger.info(json.dumps(self.personal_log))
        self.activities = []
