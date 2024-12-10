import sys
sys.path.append("..")
from flask import Flask, request, jsonify
import numpy as np
import joblib
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import LCMScheduler
import math
import cv2
import os
import torch
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import draw_kps
from pipeline_stable_diffusion_xl_instantid_inpaint import StableDiffusionXLInstantIDInpaintPipeline
import json
import base64
import io
import datetime


global pipe

# 初始化Flask应用
app = Flask(__name__)

# init FaceAnalysis
app_face = FaceAnalysis(name='antelopev2', root='/mnt/pfs-ssai-cv/DCQ/tmp_20240402/models/FaceAnysis', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))
 
# 加载预测模型
# Path to InstantID models
checkpoint_dir = '/mnt/pfs-ssai-cv/DCQ/tmp_20240402/models/InstantID'
face_adapter = f'{checkpoint_dir}/ip-adapter.bin'
controlnet_path = f'{checkpoint_dir}/ControlNetModel'

# Load pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

# LCM Lora path ( https://huggingface.co/latent-consistency/lcm-lora-sdxl )
lora = f'/mnt/pfs-ssai-cv/DCQ/tmp_20240402/models/lcm-lora-sdxl/pytorch_lora_weights.safetensors'

# You can use any base XL model (do not use models for inpainting!)
base_model_path = '/mnt/pfs-ssai-cv/DCQ/tmp_20240402/models/RealVisXL_V3.0'

pipe = StableDiffusionXLInstantIDInpaintPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16
)

pipe.cuda()
# pipe.enable_sequential_cpu_offload()
# pipe.enable_model_cpu_offload()


pipe.load_ip_adapter_instantid(face_adapter)
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

pipe.enable_xformers_memory_efficient_attention()


def base64_to_pil_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img_io = io.BytesIO(img_data)
    return Image.open(img_io)

def numpy_to_base64(image_np):
    import cv2
    import numpy as np
    import base64
    # 将图像编码为JPEG格式的字节
    _, img_encoded = cv2.imencode('.jpg', image_np)
    # 转换为Base64字符串
    image_b64 = base64.b64encode(img_encoded.tobytes()).decode()
    return image_b64

def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def prepare_average_embeding(face_list):
    face_emebdings = []
    for face_path in face_list:
      face_image = load_image(face_path)
      face_image = resize_img(face_image)
      face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
      face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
      face_emb = face_info['embedding']
      face_emebdings.append(face_emb)

    return np.concatenate(face_emebdings)

def prepareMaskAndPoseAndControlImage(pose_image, face_info, padding = 50, mask_grow = 20, resize = True):
    if padding < mask_grow:
        raise ValueError('mask_grow cannot be greater than padding')

    kps = face_info['kps']
    width, height = pose_image.size

    x1, y1, x2, y2 = face_info['bbox']
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # check if image can contain padding & mask
    m_x1 = max(0, x1 - mask_grow)
    m_y1 = max(0, y1 - mask_grow)
    m_x2 = min(width, x2 + mask_grow)
    m_y2 = min(height, y2 + mask_grow)

    m_x1, m_y1, m_x2, m_y2 = int(m_x1), int(m_y1), int(m_x2), int(m_y2)

    p_x1 = max(0, x1 - padding)
    p_y1 = max(0, y1 - padding)
    p_x2 = min(width, x2 + padding)
    p_y2 = min(height,y2 + padding)

    p_x1, p_y1, p_x2, p_y2 = int(p_x1), int(p_y1), int(p_x2), int(p_y2)

    # mask
    mask = np.zeros([height, width, 3])
    mask[m_y1:m_y2, m_x1:m_x2] = 255
    mask = mask[p_y1:p_y2, p_x1:p_x2]
    mask = Image.fromarray(mask.astype(np.uint8))

    image = np.array(pose_image)[p_y1:p_y2, p_x1:p_x2]
    image = Image.fromarray(image.astype(np.uint8))

    # resize image and KPS
    original_width, original_height = image.size
    kps -= [p_x1, p_y1]
    if resize:
        mask = resize_img(mask)
        image = resize_img(image)
        new_width, new_height = image.size
        kps *= [new_width / original_width, new_height / original_height]
    control_image = draw_kps(image, kps)

    # (mask, pose, control PIL images), (original positon face + padding: x, y, w, h)
    return (mask, image, control_image), (p_x1, p_y1, original_width, original_height)

# 定义AI服务接口
@app.route('/predict', methods=['POST'])
def predict(is_saved=True):

    # 获取请求数据
    data = request.data
    data = json.loads(data)
    
    name = data.get('name', 'no_recv')
    age = data.get('age', 'no_recv')
    print("======= recv args: ", name, str(age))

    person_images_b64 = data['person_images']
    model_image_b64 = request.json['model_image']

    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")


    face_emebdings = []
    count = 0
    for image_b64 in person_images_b64:
        person_image = base64_to_pil_image(image_b64)

        if is_saved:
            filename = "asset/upload/person_upload_" + f"{time_str}.png"
            person_image.save(filename)

        person_image = resize_img(person_image)
        person_face_info = app_face.get(cv2.cvtColor(np.array(person_image), cv2.COLOR_RGB2BGR))
        person_face_info = sorted(person_face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
        face_emb = person_face_info['embedding']
        face_emebdings.append(face_emb)
        count+=1

    # prepare face_emb
    face_emb = np.concatenate(face_emebdings)

    # get model_image face_info
    model_image = base64_to_pil_image(model_image_b64)
    model_face_info = app_face.get(cv2.cvtColor(np.array(model_image), cv2.COLOR_RGB2BGR))
    model_face_info = sorted(model_face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face

    images, position = prepareMaskAndPoseAndControlImage(
        model_image,
        model_face_info,
        60,  # padding
        40,  # grow mask
        True # resize
    )
    mask, model_image_preprocessed, control_image = images

    prompt = 'a female, look at the camera'
    # negative_prompt is used only when guidance_scale > 1
    # https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl
    negative_prompt = '(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured'
    steps = 30
    mask_strength = 0.7 # values between 0 - 1

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        control_image=control_image,
        image=model_image_preprocessed,
        mask_image=mask,
        controlnet_conditioning_scale=0.8,
        strength=mask_strength,
        ip_adapter_scale=0.3, # keep it low
        num_inference_steps=int(math.ceil(steps / mask_strength)),
        guidance_scale=7
    ).images[0]


    torch.cuda.empty_cache()

    # # processed face with padding
    # image.save('face.jpg')

    # integrate cropped result into the pose image
    x, y, w, h = position

    image = image.resize((w, h))
    model_image.paste(image, (x, y))

    if is_saved:
        filename = "asset/results/model_res_" + f"{time_str}.png"
        model_image.save(filename)

    model_image_res = numpy_to_base64(cv2.cvtColor(np.array(model_image), cv2.COLOR_RGB2BGR))
    return jsonify({'model_image_res': model_image_res})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)