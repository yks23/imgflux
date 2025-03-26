import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from src.dataset.hoivideodataset import get_train_set,get_valid_set
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)
def show_masks(image, masks, scores, path,point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(path+str(i)+'.png')
        plt.close()
        # plt.show()
def main():
    checkpoint = "/data115/video-diff/Kaisen.Yang/workspace/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    dataset1=get_train_set()
    output_dir='./body'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    for i,data in enumerate(dataset1):
        print(i)
        image=data['video']
        image=((image.permute(1,2,0).numpy()+1)*127.5).astype(np.uint8)
        predictor.set_image(image)
        if not data['hand_mask'][0].any():
            mask_img=Image.fromarray(np.zeros_like(image).astype(np.uint8))
        else:
            mask = data['hand_mask'][0]  # 去除批次维度，得到 [H, W]
            y_indices, x_indices = torch.where(mask)
            center_y = torch.mean(y_indices.float())  # y 坐标均值
            center_x = torch.mean(x_indices.float())  # x 坐标均值
            center_y_int = torch.round(center_y).int().item()
            center_x_int = torch.round(center_x).int().item()
            input_point=np.array([[center_x_int,center_y_int]])
            input_label=np.array([1])
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
                
            )
            img_name_1="0"*(5-len(str(i)))+str(i)+'tt.png'
            sorted_ind = np.argsort(scores)
            masks=masks[sorted_ind]
            scores=scores[sorted_ind]
            show_masks(image, masks, scores, path=os.path.join(output_dir,img_name_1),point_coords=input_point, input_labels=input_label, borders=True)
            mask=(masks[0]*255).astype(np.uint8)
            mask_img=Image.fromarray(mask)
            
        img_name="0"*(5-len(str(i)))+str(i)+'handmask.png'
        img_name_1="0"*(5-len(str(i)))+str(i)+'segout.png'
        img2=Image.fromarray(data['hand_mask'][0].numpy().astype(np.uint8)*255)
        img2.save(os.path.join(output_dir,img_name_1))
        mask_img.save(os.path.join(output_dir,img_name))
if __name__=='__main__':
    print(1)
    main()