import cv2
import numpy as np
from tracker import Tracker
from yolo_detector import YOLODetector
from room_counter import RoomCounter
import config

# โหลด YOLO และตัวติดตาม
yolo = YOLODetector()
tracker = Tracker()
room_counter = RoomCounter(config.areas)

# เปิดกล้อง
cap = cv2.VideoCapture(config.RTSP_URL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 15)  # ลด FPS ถ้ากระตุกมาก


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    detections = yolo.detect(frame)
    tracked_people = tracker.update(detections)
    
    room_data = room_counter.update_count(tracked_people)

    # แสดงข้อมูลแต่ละห้อง
    for i, (room_in, room_out) in enumerate(room_data, start=1):
        cv2.putText(frame, f"Room {i} IN: {room_in}, OUT: {room_out}", (50, 50 + (i * 50)), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

    # วาดกรอบห้อง
    for (area_in, area_out) in config.areas:
        cv2.polylines(frame, [np.array(area_in, np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.polylines(frame, [np.array(area_out, np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

    # เปิด-ปิดไฟในแต่ละห้อง
    for i, (room_in, room_out) in enumerate(room_data, start=1):
        light_status = "ON" if room_in > 1 and room_out < room_in else "OFF"
        color = (0, 255, 0) if light_status == "ON" else (0, 0, 255)
        cv2.putText(frame, f"Room {i} Light: {light_status}", (50, 70 + (i * 50)), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)

    cv2.imshow("People Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
