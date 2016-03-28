from PIL import Image, ImageDraw2
from facepp import API, File
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sys
import time

# FACE++ API 
API_KEY = 'YOUR API KEY HERE'
API_SECRET = 'YOUR API SECERT HERE'

api = API(API_KEY, API_SECRET)


class FaceRec(QDialog):
    def __init__(self, parent=None):
        super(FaceRec, self).__init__(parent)
        self.setWindowTitle('SimpleFaceRecognition')
        self.img_label = QLabel("Choose A Picture!")
        self.info_label = QLabel('No Picture Selected!')
        self.changed = False
        self.path = ''
        self.img_width = 0
        self.img_height = 0
        self.result = {}
        self.information = ""


    def init_ui(self):
        select_img_button = QPushButton('Choose Picture..')
        rec_img_button = QPushButton('Recognize Face..')

        right_layout = QVBoxLayout()
        right_layout.addWidget(select_img_button)
        right_layout.addWidget(rec_img_button)
        right_layout.addWidget(self.info_label)

        main_layout = QGridLayout()
        main_layout.addWidget(self.img_label, 0, 0)
        main_layout.addLayout(right_layout, 0, 1)

        self.setLayout(main_layout)

        self.connect(select_img_button, SIGNAL("clicked()"), self.open_file)
        self.connect(rec_img_button, SIGNAL("clicked()"), self.draw_sign)

    def get_result(self):
        result = api.detection.detect(img=File(self.path))
        return result

    def get_information(self):
        self.img_width = self.result["img_width"]
        self.img_height = self.result["img_height"]
        # face:age, age_range, gender, race
        age = self.result["face"][0]["attribute"]["age"]["value"]
        age_range = self.result["face"][0]["attribute"]["age"]["range"]
        gender = self.result["face"][0]["attribute"]["gender"]["value"]
        race = self.result["face"][0]["attribute"]["race"]["value"]
        return (age, age_range, gender, race)

    def calculate_coordinate(self):
        self.result = self.get_result()
        information2 = self.get_information()
        self.information = "Age Range:%s~%s\nGender:%s\nRace:%s\n" % (information2[0]-information2[1], \
                        information2[0]+information2[1], information2[2], information2[3])
        # face_center coordinate
        face_center = (self.result["face"][0]["position"]["center"]["x"] \
               / 100 * self.img_width, self.result["face"][0]["position"]["center"]["y"] \
               / 100 * self.img_height)
        # face_percentage
        face_percentage = (self.result["face"][0]["position"]["width"] \
                / 100 * self.img_width, self.result["face"][0]["position"]["height"] \
                / 100 * self.img_height)
        # face left up coordinate
        face_left_up_corner = (face_center[0] - (face_percentage[0] / 2) \
                           , face_center[1] - (face_percentage[1] / 2))
        # face right down coordinate
        face_right_down_corner = (face_center[0] + (face_percentage[0] / 2) \
                              , face_center[1] + (face_percentage[1] / 2))
        # left_eye_coordinate
        left_eye_coordinate = (self.result["face"][0]["position"]["eye_left"]["x"] \
               / 100 * self.img_width, self.result["face"][0]["position"]["eye_left"]["y"] \
               / 100 * self.img_height)
        # right_eye_coordinate
        right_eye_coordinate = (self.result["face"][0]["position"]["eye_right"]["x"] \
               / 100 * self.img_width, self.result["face"][0]["position"]["eye_right"]["y"] \
               / 100 * self.img_height)
	# left_eye_draw_coordinate
        left_coordinate = (left_eye_coordinate[0]-5,left_eye_coordinate[1]-5, \
                   left_eye_coordinate[0], left_eye_coordinate[1])
	# right_eye_draw_coordinate
        right_coordinate = (right_eye_coordinate[0], right_eye_coordinate[1], \
                            right_eye_coordinate[0]+5, right_eye_coordinate[1]+5)
	# face_draw_coordiante
        face_coordinate = (face_left_up_corner[0], face_left_up_corner[1], \
                   face_right_down_corner[0], face_right_down_corner[1])
        return [face_coordinate, left_coordinate, right_coordinate]

    def open_file(self):
        self.path = str(QFileDialog.getOpenFileName(self, "Open File", "/"))
        self.changed = True
        img = QPixmap(self.path)
        self.img_label.setPixmap(img)
        self.info_label.setText("Click Reconize Button")

    def draw_sign(self):
        face_coordinate = self.calculate_coordinate()
        image = Image.open(self.path)
        draw = ImageDraw2.Draw(image)
        pen = ImageDraw2.Pen('red', 1.0,)
        draw.rectangle(face_coordinate[0], pen)
        draw.rectangle(face_coordinate[1], pen)
        draw.rectangle(face_coordinate[2], pen)
        image.save('code.jpg', 'jpeg')
        img2 = QPixmap('code.jpg')
        self.img_label.setPixmap(img2)
        self.info_label.setText(self.information)

def main():
	app = QApplication(sys.argv)
	run = FaceRec()
	run.resize(300, 400)
	run.move(400, 400)
	run.init_ui()
	run.show()
	while run.changed:
    		run.get_result()
    		run.get_information()
    		run.calculate_coordinate()
    		run.changed = False
	sys.exit(app.exec_())

if __name__ == '__main__':
	main()
