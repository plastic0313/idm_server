#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/09/02 19:06
# @Author  : lishanlu
# @File    : post_request_samples.py
# @Software: PyCharm
# @Discription: 上传图片及参数的不同方式

from __future__ import absolute_import, print_function, division
import requests
import cv2
import base64
import json
from PIL import Image
import io
import time



person_image_path = '../examples/hf_test/test_input.png'
model_image_path = '../examples/hf_test/test_person_4.jpg'


def img2b64(img):
    retval, buffer = cv2.imencode('.bmp', img)
    # buffer = img.tostring()
    pic_str = base64.b64encode(buffer)
    pic_str = pic_str.decode()
    return pic_str

def base64_to_pil_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img_io = io.BytesIO(img_data)
    return Image.open(img_io)

def post_json():

    person_image = cv2.imread(person_image_path)
    person_image_str = img2b64(person_image)
    
    model_image = cv2.imread(model_image_path)
    model_image_str = img2b64(model_image)

    data = {'person_images': [person_image_str], 'model_image': model_image_str, 'name': 'xyz', 'age':33}
    response = requests.post('http://43.128.133.3:80/predict', json=data).json()

    res_image = base64_to_pil_image(response['model_image_res'])

    res_image.save('res_test.png')



if __name__ == '__main__':
    start_time = time.time()
    post_json()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"耗时: {elapsed_time} 秒")



