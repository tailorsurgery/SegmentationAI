<h1>Automated Digital Surgery with 3D U-Net</h1>

<h4>Author: Samya Karzazi El Bachiri</h4>

<h3>Overview</h3>

Digital surgery has made a remarkable impact on the medical field by enabling highly accurate and personalized surgical planning through automatic segmentation of CT images. This repository presents a 3D U-Net-based system that can rapidly identify bone structures to minimize manual labor and enhance clinical decision-making. Additionally, a custom Graphical User Interface (GUI) has been developed to integrate the entire workflow, from DICOM file conversion to the 3D visualization of the segmented parts.

<h3>Key Features</h3>

1.	3D U-Net Segmentation
	- Volumetric Data Processing: Leverages convolutions in 3D for high-precision bone structure identification.
	- Attention Module: Focuses on relevant image regions, improving metrics (IoU, Dice Score) for complex or noisy volumes.
  
2.	Data Preprocessing
   	- DICOM to NRRD Conversion: Ensures anonymization and compatibility with medical imaging tools.
	- Binary and Multiclass Masks: Supports both multi-class and binary configurations (background vs. bone).
  
3.	GUI Integration
	- User-Friendly Interface: Based on Python libraries (PyQt5, napari, SimpleITK) for streamlined image import, conversion, and result visualization.
	- Real-Time Review: Displays multiple anatomical axes (axial, coronal, sagittal) in 2D and 3D volumes to check and refine segmentation.
4.	Optimization Strategies
	- Patch-Based Approach: Filters out irrelevant patches (e.g., background-only) to reduce training time.
	- Attention-Driven Segmentation: Improves performance in smaller or more complex regions.

<h3>Methodology Summary</h3>

1.	Data Acquisition
   	- CT scans (TAC) and corresponding manual segmentations provided by Tailor Surgery.
	- Dataset includes upper and lower limb anatomies with varying resolution and complexity.
3.	Preprocessing
	- Conversion of DICOM and STL files into NRRD.
	- Data Augmentation: Applied mirroring/flip to boost the dataset variability and address uneven distribution.
4.	3D U-Net Architecture
	- Based on Çiçek et al.’s approach for volumetric segmentation.
	- Encoder-Decoder Structure with skip connections in 3D.
	- Attention Module added to refine the focus on crucial bone structures.
5.	Training & Validation
	- Patch Extraction: Divides large 3D volumes into smaller subvolumes (e.g., 128×128×128) for efficient GPU usage.
	- Loss Functions: Mostly cross-entropy or Dice-based loss.
	- Evaluation Metrics: IoU (Intersection over Union) and Dice Score to gauge segmentation accuracy.
	- Hardware: Trained primarily on an NVIDIA RTX 3080Ti GPU.
6.	GUI Development
	- Implemented in Python using PyQt5 (for interface), SimpleITK (for image processing), and napari (for visualization).
	- Enables easy conversion, segmentation, and real-time inspection of results.

<h3>Results</h3>

- Reduced Training Time: Through patch filtering, training time dropped from up to 8 hours per epoch to around 1 hour and 40 minutes.
- Improved Accuracy: Incorporating Attention raised both IoU and Dice Score, at the expense of higher computational cost.
- Feasibility in Clinical Settings: The workflow significantly speeds up the segmentation process, reducing manual input and allowing clinicians to focus on decision-making.

<h3>Limitations and Future Work</h3>

- Metallic Implants: The presence of metallic objects can confuse the model, lowering IoU and Dice metrics. Pre-filtering strategies for these artifacts are a future goal.
- Broader Anatomical Coverage: Currently limited to specific regions (e.g., arms). Scaling to additional anatomies requires further data and adaptation.
- Multi-Class Segmentation: Although tested with up to 9 classes, more complex datasets may require balancing strategies and additional computational resources.
- Integration with SAM-Med3D: Adapting the dataset to work with advanced zero-shot models like SAM-Med3D (Meta AI) could enhance performance and generalization.

<h3>How to Use</h3>

1.	Clone the Repository
	```bash
	git clone https://github.com/tailorsurgery/SegmentationAI.git
	cd SegmentationAI
	```
2.	Set Up the Environment
	```bash
	pip install -r requirements.txt
	```
3.	Run the model for training
	 ```bash
	python unet3d.ipynb
	```

4.	Run the GUI
	 ```bash
	python app.py
	```
	- Select the DICOM directory when prompted.
	- Review and convert the files to NRRD.
	- Press “Segment” to run 3D U-Net on the chosen images.

5.	Check Results
	- Examine segmentation masks and 3D reconstructions in napari.</li>

<h3>References</h3>

+ [1]	GitHub amb el codi: https://github.com/tailorsurgery/SegmentationAI .
+ [2]	O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” arXiv preprint arXiv:1505.04597, 2015. [Online]. Available: https://arxiv.org/pdf/1505.04597 .
+ [3]	F. Milletari, N. Navab, and S.-A. Ahmadi, “V-Net: Fully Convo-lutional Neural Networks for Volumetric Medical Image Seg-mentation,” arXiv preprint arXiv:1606.04797, 2016. [Online]. Available: https://arxiv.org/pdf/1606.04797.
+ [4]	Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, i Olaf Ronneberger, 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, arXiv, vol. 1606.06650, 2016. [Online]. Available: https://arxiv.org/abs/1606.06650 .
+ [5]	W. Xie, N. Willems, S. Patil, Y. Li, i M. Kumar, "SAM Fewshot Finetuning for Anatomical Segmentation in Medical Images", arXiv, vol. 2407.04651, Jul. 2024. [Online]. Available: https://arxiv.org/abs/2407.04651 .
+ [6]	Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, i Ross Girshick, Segment Anything, arXiv, vol. 2304.02643, 2023. [Online]. Avai-lable: https://arxiv.org/abs/2304.02643 .
+ [7]	S. Pandey, "Harvard’s MedAI 'KGARevion': A Game-Changing Knowledge Graph-based Agent of Medical QA," Medium, Unlocking AI, Oct. 16, 2024. [Online]. Available: https://medium.com/unlocking-ai/harvards-medai-kgarevion-09848282a1a4 .
+ [8]	H. Gu, R. Colglazier, H. Dong, J. Zhang, Y. Chen, Z. Yildiz, Y. Chen, L. Li, J. Yang, J. Willhite, A. M. Meyer, B. Guo, Y. A. Shah, E. Luo, S. Rajput, S. Kuehn, C. Bulleit, K. A. Wu, J. Lee, B. Ra-mirez, D. Lu, J. M. Levin, M. A. Mazurowski, "SegmentAnyBo-ne: A Universal Model that Segments Any Bone at Any Locati-on on MRI", arXiv, vol. 2401.12974, Jan. 2024. doi: 10.48550/arXiv.2401.12974. [Online]. Available: https://arxiv.org/abs/2401.12974 .
+ [9]	J. Zhou, P. F. Damasceno, R. Chachad, J. R. Cheung, A. Ballatori, J. C. Lotz, A. A. Lazar, T. M. Link, A. J. Fields, i R. Krug, "Auto-matic Vertebral Body Segmentation Based on Deep Learning of Dixon Images for Bone Marrow Fat Fraction Quantification", Frontiers in Endocrinology, vol. 11, p. 612, Sep. 2020. doi: 10.3389/fendo.2020.00612. [Online]. Available: https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2020.00612/full .
+ [10]	Y. Liu, R. Xie, L. Wang, H. Liu, C. Liu, Y. Zhao, S. Bai, i W. Liu, "Fully automatic AI segmentation of oral surgery-related tissues based on cone beam computed tomography images",  Internati-onal Journal of Oral Science, vol. 16, article 34, May 2024. doi: 10.1038/s41368-024-00123-8. [Online]. Available: https://www.nature.com/articles/s41368-024-00294-z .
+ [11]	O. Oktay et al., “Attention U-Net: Learning Where to Look for the Pancreas,” arXiv preprint arXiv:1804.03999, 2018. [Online]. Available: https://arxiv.org/abs/1804.03999 .
+ [12]	Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, “UNet++: A Nested U-Net Architecture for Medical Image Segmentation,” arXiv preprint arXiv:1807.10165, 2018. [Online]. Available: https://arxiv.org/abs/1807.10165.
+ [13]	S. F. Abbas, N. T. Duc, Y. Song, K. Kim, E. Srivastava, and B. Lee, “CV-Attention UNet: Attention-based UNet for 3D Cerebro-vascular Segmentation of Enhanced TOF-MRA Images,” arXiv preprint arXiv:2311.10224, Jun. 2024. [Online]. Available: https://arxiv.org/html/2311.10224v3 .
+ [14]	D. Müller, I. Soto-Rey, and F. Kramer, “Towards a Guideline for Evaluation Metrics in Medical Image Segmentation,” arXiv preprint arXiv:2202.05273, 2022. [Online]. Available: https://arxiv.org/pdf/2202.05273.


<h3>Contact</h3>

For inquiries or support, please reach out to: Samya Karzazi El Bachiri (samya.uab@gmail.com)
