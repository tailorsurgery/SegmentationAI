import os.path as osp
import medim
from medim.utils import data_preprocess, data_postprocess, sam_model_infer

''' 1. read and pre-process your input data '''
img_path = "./test_data/kidney_right/AMOS/imagesVal/amos_0013.nii.gz"
gt_path = "./test_data/kidney_right/AMOS/labelsVal/amos_0013.nii.gz"
category_index = 3  # the index of your target category in the gt annotation
output_dir = "./test_data/kidney_right/AMOS/pred/"
roi_image, roi_label, meta_info = data_preprocess(img_path, gt_path, category_index=category_index)

''' 2. prepare the pre-trained model with local path or huggingface url '''
ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
# or you can use the local path like: ckpt_path = "./ckpt/sam_med3d_turbo.pth"
model = medim.create_model("SAM-Med3D",
                           pretrained=True,
                           checkpoint_path=ckpt_path)

''' 3. infer with the pre-trained SAM-Med3D model '''
roi_pred = sam_model_infer(model, roi_image, roi_gt=roi_label)

''' 4. post-process and save the result '''
output_path = osp.join(output_dir, osp.basename(img_path).replace(".nii.gz", "_pred.nii.gz"))
data_postprocess(roi_pred, meta_info, output_path, img_path)

print("result saved to", output_path)