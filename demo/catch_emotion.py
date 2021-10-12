import cv2
import json

class EmotionPredictor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def predict_emotion(self, output_path):
        with open('../logs/emotion.log', 'r') as f:
            face_boxes_and_emotions_by_timestamp = json.load(f)

        # for timestamp, v in face_boxes_and_emotions_by_timestamp.items():
        #     print(f'timestamp: {timestamp}')
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000/fps)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(width), int(height)))

        timestamps = list(map(int, face_boxes_and_emotions_by_timestamp.keys()))
        pivot = 0
        prev_faces = []
        
        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)

            try:
                if abs(pos_msec-timestamps[pivot]) < 15:
                    prev_faces = faces = face_boxes_and_emotions_by_timestamp[str(timestamps[pivot])]
                    for face in faces:
                        l, t, r, b = face['pos']
                        emotion = face['emotions'][0]
                        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), thickness=2)
                        cv2.putText(frame, emotion['Type']+' '+str(round(emotion['Confidence'], 1)), (l, t), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)

                    pivot += 1

                    # cv2.imshow('output', frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                
            except IndexError:
                pass

            for face in prev_faces:
                l, t, r, b = face['pos']
                emotion = face['emotions'][0]
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), thickness=2)
                cv2.putText(frame, emotion['Type']+' '+str(round(emotion['Confidence'], 1)), (l, t), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)

            # cv2.imshow('output', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            out.write(frame)

        self.cap.release()
        cv2.destroyAllWindows()
