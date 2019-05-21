import os
import sys
import logging
import pickle
from collections import namedtuple
from operator import itemgetter

import begin
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2 import QtUiTools
import qimage2ndarray
import numpy as np
import matplotlib

from generators import HangingBinaryDataGenerator


def load_ui(file_name, where=None):
    """
    Loads a .UI file into the corresponding Qt Python object
    :param file_name: UI file path
    :param where: Use this parameter to load the UI into an existing class (i.e. to override methods)
    :return: loaded UI
    """
    # Create a QtLoader
    loader = QtUiTools.QUiLoader()

    # Open the UI file
    ui_file = QtCore.QFile(file_name)
    ui_file.open(QtCore.QFile.ReadOnly)

    # Load the contents of the file
    ui = loader.load(ui_file, where)

    # Close the file
    ui_file.close()

    return ui

# Named tuple to store evaluation results
# Evaluation = namedtuple('Evaluation', 'name stage1 stage2 stage3')


class TextilesHangingEvaluationWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        # Widgets and UI
        self.hangedResultButton = None
        self.floorResultButton = None
        self.badResultButton = None
        self.infoLabel = None
        self.infoLabel_string = None
        self.graphicsView = None
        self.pixmap = QtGui.QPixmap()

        # Iteration over input data
        self.generator = None
        self.current_batch = None

        # Output data related
        self.output_data = []
        self.current_label = None
        self.output_data_path = os.path.abspath('.')

        self.setup_ui()

    def setup_ui(self):
        # Load UI and set it as main layout
        ui_file_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'TextilesHangingEvaluationWidget.ui')
        main_widget = load_ui(ui_file_path, self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(main_widget)
        self.setLayout(layout)

        # Get a reference to the widgets
        self.hangedResultButton = self.findChild(QtWidgets.QPushButton, 'hangedResultButton')
        self.floorResultButton = self.findChild(QtWidgets.QPushButton, 'floorResultButton')
        self.infoLabel = self.findChild(QtWidgets.QLabel, 'infoLabel')
        self.infoLabel.setText("Advertise yourself here! Call now!")
        self.graphicsView = self.findChild(QtWidgets.QGraphicsView, 'graphicsView')

        # Connect slots / callbacks
        self.hangedResultButton.clicked.connect(self.on_hanged_button_clicked)
        self.floorResultButton.clicked.connect(self.on_floor_button_clicked)

    def update_image(self):
        """
        Set pixmap in widget's graphics view
        """
        try:
            i, img = self.current_batch.__next__()
        except StopIteration:
            with open("out.pickle", "wb") as f:
                pickle.dump(self.output_data, f)
                self.get_next_batch()
                self.update_image()

        img = matplotlib.cm.get_cmap('gist_stern')(img)
        img = np.uint8(img * 255)
        qimage = qimage2ndarray.array2qimage(img[:, :, 0, :3])
        scene = QtWidgets.QGraphicsScene()
        scene.addItem(QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(qimage)))
        self.graphicsView.setScene(scene)
        self.graphicsView.show()
        self.infoLabel.setText(self.infoLabel_string.format(i+1))

    def on_hanged_button_clicked(self):
        self.current_label = 1
        self.on_button_clicked()

    def on_floor_button_clicked(self):
        self.current_label = 0
        self.on_button_clicked()

    def on_button_clicked(self):
        """
        Try to load next image, if there is no image left, load the next item
        """
        self.output_data.append(self.current_label)
        self.current_label = None
        self.update_image()

    def start(self):
        if not self.generator:
            logging.error("Generator has not been setup correctly")
            exit(-1)
        self.generator = enumerate(iter(self.generator))
        self.get_next_batch()
        self.update_image()

    def get_next_batch(self):
        try:
            index, data = self.generator.__next__()
        except StopIteration:
            logging.info("No more data left")
            self.close()
        self.current_batch = enumerate(iter(data[0]))
        self.infoLabel_string = "Current batch: {}".format(index) + " Current image: {}/100"


@begin.start(auto_convert=True)
@begin.logging
def main(test_files: 'Pickle file containing the names of the files to be labeled',
         input_folder: 'Folder containing the files to be labeled'):
    test_files = os.path.abspath(os.path.expanduser(test_files))
    input_folder = os.path.abspath(os.path.expanduser(input_folder))
    logging.info("Starting labeling utility with the following params:")
    logging.info("Test files file: {}".format(test_files))
    logging.info("Input folder: {}".format(input_folder))

    # Create data generator
    with open(test_files, 'rb') as f:
        test_files = pickle.load(f)
    params = {'batch_size': 100, 'resize': True,  'shuffle': False}
    training_generator = HangingBinaryDataGenerator(test_files, input_folder, **params)

    # Create Qt App
    app = QtWidgets.QApplication(sys.argv)
    gui = TextilesHangingEvaluationWidget()
    gui.generator = training_generator
    gui.start()
    gui.show()

    # Run the app
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
