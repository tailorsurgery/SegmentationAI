import os
import sys

from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

# Add the directory containing dicom2nrrd.py to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, "utils")  # Assuming dicom2nrrd.py is in a 'utils' folder
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from utils.dicom2nrrd import process_images  # Adjust the import to match the function in dicom2nrrd.py
from utils.align_image_multiclass_mask import align_image, load_multiclass_mask, viewer_with_colored_classes

import warnings
warnings.filterwarnings("ignore", message="A QApplication is already running with 1 event loop.*")
warnings.filterwarnings("ignore", message="color_dict did not provide a default color.*")


class SegmentationAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SegmentationAI Viewer")
        self.setGeometry(700, 200, 200, 300)

        layout = QVBoxLayout()

        # Add the logo to the main view
        logo_path = os.path.join(utils_dir, "logo/logo_with_name_inverted.png")
        if os.path.exists(logo_path):
            self.logo_label = QLabel(self)
            pixmap = QPixmap(logo_path)
            self.logo_label.setPixmap(pixmap)
            self.logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.logo_label)

        # Input box for case name
        self.case_name_input = QLineEdit(self)
        self.case_name_input.setPlaceholderText("Enter case name (e.g., 240028)")
        layout.addWidget(self.case_name_input)

        # Button to set case name
        self.set_case_name_button = QPushButton("Set Case Name")
        self.set_case_name_button.clicked.connect(self.set_case_name)
        layout.addWidget(self.set_case_name_button)
        # Add buttons
        self.select_dicom_button = QPushButton("Select DICOM Images")
        self.select_dicom_button.clicked.connect(self.select_dicom_folder)
        layout.addWidget(self.select_dicom_button)

        self.select_mask_button = QPushButton("AI Segmentation")
        self.select_mask_button.clicked.connect(self.select_mask_file)
        layout.addWidget(self.select_mask_button)

        self.view_button = QPushButton("Show Segmentation")
        self.view_button.clicked.connect(self.view_images)
        layout.addWidget(self.view_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.dicom_folder = None
        self.image_path = None
        self.mask_path = None

    def set_case_name(self):
        """
        Set the case name based on user input.
        """
        self.case_name = self.case_name_input.text().strip()
        if self.case_name:
            print(f"Case name set to: {self.case_name}")
        else:
            print("Case name is empty. Please enter a valid case name.")

    def select_dicom_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder", script_dir)
        if folder:
            self.dicom_folder = folder
            # Convert DICOM to NRRD
            output_dir = os.path.join(script_dir, "scripts/output")  # Define output directory for NRRD files
            os.makedirs(output_dir, exist_ok=True)

            #self.case_name = "240028"  # Adjust the case naming convention if needed
            process_images(self.dicom_folder, f"{output_dir}/images", self.case_name)
            self.image_path = os.path.join(output_dir, f"images/{self.case_name}_images.nrrd")
            self.mask_path = os.path.join(output_dir, f"multiclass_masks/{self.case_name}_multiclass_mask.nrrd")
            print(f"NRRD file saved correctly")

    def select_mask_file(self):
        # Call U-Net model to generate mask
        # output_dir = os.path.join(script_dir, "scripts/output")
        # os.makedirs(output_dir, exist_ok=True)
        # self.mask_path = os.path.join(output_dir, f"multiclass_masks/{self.case_name}-10_multiclass_mask.nrrd")

        print("Tailor Surgery AI Segmentation - Coming soon! :)")
        '''file, _ = QFileDialog.getOpenFileName(self, "Select Mask File", filter="NRRD Files (*.nrrd)")
        if file:
            self.mask_path = file
            print(f"Mask file selected: {self.mask_path}")'''

    def view_images(self):
        '''if not self.image_path or not self.mask_path:
            print("Please select both a DICOM folder and a mask file.")
            return'''

        image_array, image_spacing, _ = align_image(self.image_path, flip=False, save=False)
        multiclass_mask, labels = load_multiclass_mask(self.mask_path)
        viewer_with_colored_classes(image_array, multiclass_mask, image_spacing, labels)



if __name__ == "__main__":
    app = QApplication([])
    window = SegmentationAIApp()
    window.show()
    app.exec()


