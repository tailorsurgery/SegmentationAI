Below is a README file in English, summarizing the key points of your project. Feel free to modify or expand it according to your needs.

Automated Digital Surgery with 3D U-Net

Author: Samya Karzazi El Bachiri

Overview

Digital surgery has made a remarkable impact on the medical field by enabling highly accurate and personalized surgical planning through automatic segmentation of CT images. This repository presents a 3D U-Net-based system that can rapidly identify bone structures to minimize manual labor and enhance clinical decision-making. Additionally, a custom Graphical User Interface (GUI) has been developed to integrate the entire workflow, from DICOM file conversion to the 3D visualization of the segmented parts.

Key Features
	1.	3D U-Net Segmentation
	•	Volumetric Data Processing: Leverages convolutions in 3D for high-precision bone structure identification.
	•	Attention Module: Focuses on relevant image regions, improving metrics (IoU, Dice Score) for complex or noisy volumes.
	2.	Data Preprocessing
	•	DICOM to NRRD Conversion: Ensures anonymization and compatibility with medical imaging tools.
	•	Binary and Multiclass Masks: Supports both multi-class and binary configurations (background vs. bone).
	3.	GUI Integration
	•	User-Friendly Interface: Based on Python libraries (PyQt5, napari, SimpleITK) for streamlined image import, conversion, and result visualization.
	•	Real-Time Review: Displays multiple anatomical axes (axial, coronal, sagittal) in 2D and 3D volumes to check and refine segmentation.
	4.	Optimization Strategies
	•	Patch-Based Approach: Filters out irrelevant patches (e.g., background-only) to reduce training time.
	•	Attention-Driven Segmentation: Improves performance in smaller or more complex regions.

Methodology Summary
	1.	Data Acquisition
	•	CT scans (TAC) and corresponding manual segmentations provided by Tailor Surgery.
	•	Dataset includes upper and lower limb anatomies with varying resolution and complexity.
	2.	Preprocessing
	•	Conversion of DICOM and STL files into NRRD.
	•	Data Augmentation: Applied mirroring/flip to boost the dataset variability and address uneven distribution.
	3.	3D U-Net Architecture
	•	Based on Çiçek et al.’s approach for volumetric segmentation.
	•	Encoder-Decoder Structure with skip connections in 3D.
	•	Attention Module added to refine the focus on crucial bone structures.
	4.	Training & Validation
	•	Patch Extraction: Divides large 3D volumes into smaller subvolumes (e.g., 128×128×128) for efficient GPU usage.
	•	Loss Functions: Mostly cross-entropy or Dice-based loss.
	•	Evaluation Metrics: IoU (Intersection over Union) and Dice Score to gauge segmentation accuracy.
	•	Hardware: Trained primarily on an NVIDIA RTX 3080Ti GPU.
	5.	GUI Development
	•	Implemented in Python using PyQt5 (for interface), SimpleITK (for image processing), and napari (for visualization).
	•	Enables easy conversion, segmentation, and real-time inspection of results.

Results
	•	Reduced Training Time: Through patch filtering, training time dropped from up to 8 hours per epoch to around 1 hour and 40 minutes.
	•	Improved Accuracy: Incorporating Attention raised both IoU and Dice Score, at the expense of higher computational cost.
	•	Feasibility in Clinical Settings: The workflow significantly speeds up the segmentation process, reducing manual input and allowing clinicians to focus on decision-making.

Limitations and Future Work
	•	Metallic Implants: The presence of metallic objects can confuse the model, lowering IoU and Dice metrics. Pre-filtering strategies for these artifacts are a future goal.
	•	Broader Anatomical Coverage: Currently limited to specific regions (e.g., arms). Scaling to additional anatomies requires further data and adaptation.
	•	Multi-Class Segmentation: Although tested with up to 9 classes, more complex datasets may require balancing strategies and additional computational resources.
	•	Integration with SAM-Med3D: Adapting the dataset to work with advanced zero-shot models like SAM-Med3D (Meta AI) could enhance performance and generalization.

How to Use
	1.	Clone the Repository
```bash
git clone https://github.com/yourusername/digital-surgery-3dunet.git
cd digital-surgery-3dunet
```
	2.	Set Up the Environment
```bash
pip install -r requirements.txt
```
	3.	Run the GUI
```bash
python main_gui.py
```
	•	Select the DICOM directory when prompted.
	•	Review and convert the files to NRRD.
	•	Press “Segment” to run 3D U-Net on the chosen images.
	4.	Check Results
	•	Examine segmentation masks and 3D reconstructions in napari.

References
	1.	Ö. Çiçek et al., “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,” arXiv preprint arXiv:1606.06650, 2016.
	2.	A. Kirillov et al., “Segment Anything,” arXiv preprint arXiv:2304.02643, 2023.

Contact

For inquiries or support, please reach out to:
	•	Samya Karzazi El Bachiri (samya.uab@gmail.com)
