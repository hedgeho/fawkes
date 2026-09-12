# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog
from fawkes.protection import Fawkes
import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'

class Worker(QThread):
    signal = pyqtSignal('PyQt_PyObject')

    def __init__(self):
        QThread.__init__(self)
        self.image_paths = None
        self.target_dir = None
        self.my_fawkes = None

    def run(self):
        if self.my_fawkes is None or self.my_fawkes.target_dir != self.target_dir:
            self.my_fawkes = Fawkes(mode='low', target_dir=self.target_dir)
        status = self.my_fawkes.run_protection(self.image_paths, debug=True)
        self.signal.emit(status)


class FawkesAPP(object):
    def __init__(self, Form):
        Form.setObjectName("Form")
        Form.resize(220, 200)

        self.running = False
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(0, 20, 220, 36))

        self.targetButton = QtWidgets.QPushButton(Form)
        self.targetButton.setGeometry(QtCore.QRect(0, 70, 220, 36))

        self.cloakButton = QtWidgets.QPushButton(Form)
        self.cloakButton.setGeometry(QtCore.QRect(0, 120, 220, 36))

        self.img_paths = None
        self.target_dir = None

        self.labelA = QtWidgets.QLabel(Form)
        self.labelA.setText('Please select images to protect. ')
        self.labelA.move(10, 165)

        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        self.thread = Worker()
        self.thread.signal.connect(self.finished)

    def retranslateUi(self, Form):
        self.tr = QtCore.QCoreApplication.translate
        Form.setWindowTitle(self.tr("Form", "Fawkes"))
        self.pushButton.setText(self.tr("Form", "Select Images"))
        self.targetButton.setText(self.tr("Form", "Select Target Photos Folder"))
        self.cloakButton.setText(self.tr("Form", "Protect Selected Images"))
        self.pushButton.clicked.connect(self.pushButton_handler)
        self.targetButton.clicked.connect(self.target_handler)
        self.cloakButton.clicked.connect(lambda: self.protect_images())

    def pushButton_handler(self):
        print("Button pressed")
        self.open_dialog_box()

    def open_dialog_box(self):
        qfd = QFileDialog()
        path = "."
        filter = "Images (*.png *.xpm *.jpg *jpeg *.gif)"

        filename = QFileDialog.getOpenFileNames(qfd, "Select Image(s)", path, filter)
        self.img_paths = filename[0]
        print("Selected paths", self.img_paths)
        self.labelA.setText('Selected {} images'.format(len(self.img_paths)))

    def target_handler(self):
        path = QFileDialog.getExistingDirectory(None, "Folder with a few photos of the person to mimic", ".")
        if path:
            self.target_dir = path
            self.labelA.setText('Target: {}'.format(os.path.basename(path)))

    def finished(self, result):
        if result == 1:
            self.labelA.setText("Finished! Saved to original folder. ")
        elif result == 2:
            self.labelA.setText("Error: No face detected. ")
        elif result == 3:
            self.labelA.setText("Error: No image selected. ")
        self.cloakButton.setEnabled(True)
        self.pushButton.setEnabled(True)
        self.img_paths = None

    def protect_images(self):
        if self.img_paths is None:
            self.labelA.setText("Please select images first.")
            return
        if self.target_dir is None:
            self.labelA.setText("Please select a target photos folder.")
            return

        self.labelA.setText("Running Fawkes... ~{} minute(s)".format(int(len(self.img_paths) * 2)))
        self.labelA.repaint()

        self.thread.image_paths = self.img_paths
        self.thread.target_dir = self.target_dir
        self.cloakButton.setEnabled(False)
        self.pushButton.setEnabled(False)

        self.thread.start()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = FawkesAPP(Form)
    Form.show()
    sys.exit(app.exec_())
