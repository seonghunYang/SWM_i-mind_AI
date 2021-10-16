import boto3
import cv2

# AWS Rekognition
class AWSClient:
    def __init__(self):
        # self.S3_ACCESS_KEY_ID = '...'
        # self.S3_SECRET_ACCESS_KEY = '...'
        self.BUCKET_NAME = 'test.i-mind101.com'
        self.PREFIX = 'public/test/'
        self.FILE_NAME = 'double_single.mp4'
        self.FILE_KEY = self.PREFIX+self.FILE_NAME

        print('Creating AWS S3, Rekognition')
        self.s3 = boto3.resource(
            's3',
            # aws_access_key_id=self.S3_ACCESS_KEY_ID,
            # aws_secret_access_key=self.S3_SECRET_ACCESS_KEY
        )
        self.s3_client = boto3.client(
            's3',
            # aws_access_key_id=self.S3_ACCESS_KEY_ID,
            # aws_secret_access_key=self.S3_SECRET_ACCESS_KEY,
            region_name='ap-northeast-2'
        )
        self.bucket = self.s3.Bucket(name=self.BUCKET_NAME)
        self.rekog = boto3.client(
            'rekognition',
            # aws_access_key_id=self.S3_ACCESS_KEY_ID,
            # aws_secret_access_key=self.S3_SECRET_ACCESS_KEY,
            region_name='ap-northeast-2'
        )

        self.start_face_detection_response = None
        self.get_face_detection_response = None

    def download_file(self, file_key, file_name):
        """
        Request Syntax:
        boto3.resource.meta.client.download_file(
            Bucket,
            Key,
            Filename,
            ExtraArgs=None,
            Callback=None,
            Config=None
        )
        boto3.resource.Bucket.download_file(Key, ...)

        ex)
        self.s3.meta.client.download_file(
            self.BUCKET_NAME,
            self.FILE_KEY,
            file_name
        )
        """
        self.s3.meta.client.download_file(
            self.BUCKET_NAME,
            file_key,
            file_name
        )

    def upload_file(self, file_name, file_key):
        """
        Request Syntax:
        boto3.resource.meta.client.upload_file(
            Filename,
            Bucket,
            Remotepath
        )

        ex)
        self.s3.meta.client.upload_file(
            'processed.mp4',
            'test.i-mind101.com',
            'public/test/processed.mp4'
        )
        """
        self.s3.meta.client.upload_file(
            file_name,
            self.BUCKET_NAME,
            file_key
        )

    def start_face_detection(self, file_key):
        success = False
        request_token = 1
        while not success:
            try:
                self.start_face_detection_response = self.rekog.start_face_detection(
                    Video={
                        'S3Object': {
                            'Bucket': self.BUCKET_NAME,
                            'Name': file_key
                        }
                    },
                    ClientRequestToken=str(request_token),
                    FaceAttributes = 'ALL'
                )
                print(f'start face detection with request token: {request_token}')
                success = True

            except:
                request_token += 1

        print(f'start_face_detection_response: {self.start_face_detection_response}')

    def get_face_detection(self):
        self.get_face_detection_response = self.rekog.get_face_detection(
            JobId=self.start_face_detection_response['JobId'],
            MaxResults=123
        )
        print(f'get_face_detection_response: {self.get_face_detection_response}')

    def get_next(self):
        if 'NextToken' in self.get_face_detection_response:
            self.get_face_detection_response = self.rekog.get_face_detection(
                JobId=self.start_face_detection_response['JobId'],
                MaxResults=123,
                NextToken=self.get_face_detection_response['NextToken']
            )
            return True

        return False

    def get_all_rekog_results(self):
        all_rekog_results = []

        self.get_face_detection_response = self.rekog.get_face_detection(
            JobId=self.start_face_detection_response['JobId'],
            MaxResults=123
        )

        all_rekog_results.append(self.get_face_detection_response)
        self.width = self.get_face_detection_response['VideoMetadata']['FrameWidth']
        self.height = self.get_face_detection_response['VideoMetadata']['FrameHeight']

        while 'NextToken' in self.get_face_detection_response:
            self.get_face_detection_response = self.rekog.get_face_detection(
                JobId=self.start_face_detection_response['JobId'],
                MaxResults=123,
                NextToken=self.get_face_detection_response['NextToken']
            )
            all_rekog_results.append(self.get_face_detection_response)

        return all_rekog_results

    def get_face_boxes_and_emotions(self, all_rekog_results):
        face_boxes_and_emotions_by_timestamp = {}

        for rekog_result in all_rekog_results:
            for detection in rekog_result['Faces']:
                timestamp = detection['Timestamp']

                if timestamp not in face_boxes_and_emotions_by_timestamp:
                    face_boxes_and_emotions_by_timestamp[timestamp] = []

                box_set = detection['Face']['BoundingBox']

                l = int(self.width * box_set['Left'])
                t = int(self.height * box_set['Top'])
                r = int((self.width * box_set['Width']) + l)
                b = int((self.height * box_set['Height']) + t)

                face_boxes_and_emotions_by_timestamp[timestamp].append(
                    {
                        'pos': (l, t, r, b),
                        'emotions': detection['Face']['Emotions']
                    }
                )

        return face_boxes_and_emotions_by_timestamp

"""
탐지 가능한 표정 종류:
1. 놀람(SURPRISED)
2. 혼란(CONFUSED)
3. 평온(CALM)
4. 기쁨(HAPPY)
5. 화남(ANGRY)
6. 질색, 역겨움(DISGUSTED)
7. 슬픔(SAD)
8. 무서움, 공포(FEAR)

이목구비의 위치와 성별, 나이대 등을 예측할 수도 있음.
표정 인식은 영상 내에서 약 0.5초마다 얼굴 탐지 - 표정 인식을 진행하는데, 처리 속도는 영상 k분에 대하여 약 k분(1:1 비율) 소요되는 것으로 확인됨

결과값 예시:
{'Timestamp': 32232, 'Face': {'BoundingBox': {'Width': 0.2492465078830719, 'Height': 0.16330894827842712, 'Left': 0.4193752408027649, 'Top': 0.2872404456138611}, 'AgeRange': {'Low': 2, 'High': 8}, 'Smile': {'Value': False, 'Confidence': 99.233642578125}, 'Eyeglasses': {'Value': False, 'Confidence': 94.7647933959961}, 'Sunglasses': {'Value': False, 'Confidence': 98.55323028564453}, 'Gender': {'Value': 'Male', 'Confidence': 95.00914001464844}, 'Beard': {'Value': False, 'Confidence': 79.92568969726562}, 'Mustache': {'Value': False, 'Confidence': 97.12648010253906}, 'EyesOpen': {'Value': True, 'Confidence': 72.04719543457031}, 'MouthOpen': {'Value': False, 'Confidence': 89.42609405517578}, 'Emotions': [{'Type': 'CALM', 'Confidence': 92.77474212646484}, {'Type': 'CONFUSED', 'Confidence': 2.904121160507202}, {'Type': 'SAD', 'Confidence': 2.4802494049072266}, {'Type': 'SURPRISED', 'Confidence': 1.1435612440109253}, {'Type': 'ANGRY', 'Confidence': 0.2569330036640167}, {'Type': 'HAPPY', 'Confidence': 0.17666040360927582}, {'Type': 'DISGUSTED', 'Confidence': 0.14547398686408997}, {'Type': 'FEAR', 'Confidence': 0.11825743317604065}], 'Landmarks': [{'Type': 'eyeLeft', 'X': 0.45085737109184265, 'Y': 0.3478122651576996}, {'Type': 'eyeRight', 'X': 0.5550960302352905, 'Y': 0.3621358275413513}, {'Type': 'mouthLeft', 'X': 0.44908303022384644, 'Y': 0.4105801582336426}, {'Type': 'mouthRight', 'X': 0.536986768245697, 'Y': 0.4229136109352112}, {'Type': 'nose', 'X': 0.46927961707115173, 'Y': 0.39829984307289124}, {'Type': 'leftEyeBrowLeft', 'X': 0.4264124035835266, 'Y': 0.3247862756252289}, {'Type': 'leftEyeBrowRight', 'X': 0.4431769847869873, 'Y': 0.3271276652812958}, {'Type': 'leftEyeBrowUp', 'X': 0.4673379361629486, 'Y': 0.33617711067199707}, {'Type': 'rightEyeBrowLeft', 'X': 0.5266111493110657, 'Y': 0.34368446469306946}, {'Type': 'rightEyeBrowRight', 'X': 0.5626844167709351, 'Y': 0.34242111444473267}, {'Type': 'rightEyeBrowUp', 'X': 0.6070130467414856, 'Y': 0.3485867977142334}, {'Type': 'leftEyeLeft', 'X': 0.4363901913166046, 'Y': 0.3437507152557373}, {'Type': 'leftEyeRight', 'X': 0.47198277711868286, 'Y': 0.3511461615562439}, {'Type': 'leftEyeUp', 'X': 0.4499261975288391, 'Y': 0.3452407717704773}, {'Type': 'leftEyeDown', 'X': 0.4510535001754761, 'Y': 0.35062161087989807}, {'Type': 'rightEyeLeft', 'X': 0.5348142981529236, 'Y': 0.35965976119041443}, {'Type': 'rightEyeRight', 'X': 0.5775337219238281, 'Y': 0.3628527522087097}, {'Type': 'rightEyeUp', 'X': 0.5544503331184387, 'Y': 0.3593718409538269}, {'Type': 'rightEyeDown', 'X': 0.554295003414154, 'Y': 0.36466822028160095}, {'Type': 'noseLeft', 'X': 0.4660630524158478, 'Y': 0.3959483504295349}, {'Type': 'noseRight', 'X': 0.5046395063400269, 'Y': 0.40094470977783203}, {'Type': 'mouthUp', 'X': 0.48025184869766235, 'Y': 0.4139835834503174}, {'Type': 'mouthDown', 'X': 0.48094093799591064, 'Y': 0.43114763498306274}, {'Type': 'leftPupil', 'X': 0.45085737109184265, 'Y': 0.3478122651576996}, {'Type': 'rightPupil', 'X': 0.5550960302352905, 'Y': 0.3621358275413513}, {'Type': 'upperJawlineLeft', 'X': 0.43613141775131226, 'Y': 0.32516252994537354}, {'Type': 'midJawlineLeft', 'X': 0.4386405944824219, 'Y': 0.3965441584587097}, {'Type': 'chinBottom', 'X': 0.4873442053794861, 'Y': 0.4585384428501129}, {'Type': 'midJawlineRight', 'X': 0.6226316690444946, 'Y': 0.4200594425201416}, {'Type': 'upperJawlineRight', 'X': 0.6623634696006775, 'Y': 0.3542330265045166}], 'Pose': {'Roll': 14.216777801513672, 'Yaw': -20.475229263305664, 'Pitch': -23.94377326965332}, 'Quality': {'Brightness': 75.83385467529297, 'Sharpness': 26.1773681640625}, 'Confidence': 99.99693298339844}}
"""
