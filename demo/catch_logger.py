import logging, os

class PersonalLogger():
    def __init__(self, path):
        self.logger = logging.getLogger("PersonalLogger")
        
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(fmt="%(asctime)s - %(message)s")

        # stream(console) handler 객체 생성
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # file handler 객체 생성
        os.makedirs(path, exist_ok=True)
        file_handler = logging.FileHandler(filename=path+'/personal.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.persons = []
        self.personal_log = dict()

    def add_person(self, input):
        name, action, x1, y1, x2, y2 = input
        person = {'name': name, 'action': action, 'pos': [x1, y1, x2, y2]}
        self.persons.append(person)

    def log(self, frame_num):
        self.personal_log['person'] = self.persons
        self.personal_log['frame'] = frame_num
        self.logger.info(self.personal_log)
        self.persons = []
