import json
import sys, os
from datetime import datetime

import cv2
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QMessageBox
from ux import main_ux
from deepface import DeepFace
import threading
import pandas as pd


class AttendanceApp(QMainWindow, main_ux.Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(AttendanceApp, self).__init__(*args, **kwargs)
        self.setupUi(self)
        os.makedirs('attendance_files', exist_ok=True)
        os.makedirs('dataset', exist_ok=True)
        try:
            self.Worker1 = Worker1()
            self.Worker1.start()
            self.pushButton.clicked.connect(self.Worker1.takePhoto)
            self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
            self.Worker1.FaceRecognation.connect(self.face_recognation)
        except Exception as e:
            pass

    def closeEvent(self, event):
        self.Worker1.closeCamera()

    def ImageUpdateSlot(self, Image):
        self.camer_label.setPixmap(QPixmap.fromImage(Image))

    def face_recognation(self, data):
        try:
            dt = data[0]
            data_json = json.loads(dt.to_json(orient='records'))
            name = data_json[0]['identity'].replace('dataset', "").replace(".jpg", "").replace(".png", "").replace(
                ".jpeg", "").replace("/", "").replace("\\", "")
            names = name.split('_')
            threading.Thread(target=self.excel_write, args=(names[1], names[0], )).start()
            self.fio_label.setText(names[1])
            self.hemis_id_label.setText(names[0])
            pixmap = QPixmap('unknown.jpg')
            new_size = pixmap.scaled(300, 200)
            self.student_image_label.setPixmap(new_size)

        except Exception as e:
            print(e, 'error')

    def excel_write(self, name, id):
        try:
            date_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            date = datetime.today().date()
            file_path = f"attendance_files/attendance_{date}.xlsx"
            if os.path.exists(file_path):
                data = {
                    "hemis id": id,
                    "O'quvchi": name,
                    "Sana": date_time
                }
                df = pd.read_excel(file_path)
                if not (df['hemis id'] == data['hemis id']).any():
                    pass
                else:
                    df = df._append(data, ignore_index=True)
                    df.to_excel(file_path, index=False)
            else:
                data = {
                    "hemis id": [id],
                    "O'quvchi": [name],
                    "Sana": [date_time]
                }
                df = pd.DataFrame(data)
                df.to_excel(file_path, index=False)
        except Exception as e:
            print(e)


class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    FaceRecognation = pyqtSignal(list)
    face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    Capture = cv2.VideoCapture(0)

    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            ret, frame = self.Capture.read()
            detected_frame = self.detect_faces(frame)
            if ret:
                Image = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0],
                                           QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)

    def stop(self):
        self.ThreadActive = False

    def closeCamera(self):
        self.ThreadActive = False
        self.Capture.release()

    def takePhoto(self):
        ret, frame = self.Capture.read()
        if ret:
            cv2.imwrite("unknown.jpg", frame)
            self.stop()
            self.facethread = threading.Thread(target=self.perform_face_recognition, args=("unknown.jpg",))
            self.facethread.start()


    def perform_face_recognition(self, img_path):
        obj = DeepFace.find(img_path=img_path, db_path="dataset", detector_backend='opencv')
        self.FaceRecognation.emit(obj)
        self.run()

    def detect_faces(self, image):
        try:
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            return image
        except Exception as e:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttendanceApp()
    window.show()
    sys.exit(app.exec_())
