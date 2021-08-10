import cv2
import matplotlib.pyplot as plt

def showImg(img_path):
    img_arr  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_arr)
    plt.show()
    
    return img_arr

def drawFrameWithBbox(frame, detected_faces, labels):
    draw_frame = frame.copy()
    idx = 0
    for key in detected_faces.keys():
      face = detected_faces[key]
      face_area = face['facial_area']
      left = face_area[0]
      top = face_area[1]
      right = face_area[2]
      bottom = face_area[3]

      cv2.rectangle(draw_frame, (left, top), (right, bottom), (255, 0, 0), 1)
      cv2.putText(draw_frame, labels[idx], (int(left), int(top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
      landmarks = face['landmarks']
      idx += 1
      # landmark 
      for key in landmarks:
        cv2.circle(draw_frame, tuple(landmarks[key]), 1, (255, 0, 0), -1)
      
    return draw_frame
