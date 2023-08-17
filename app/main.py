import cv2
import numpy as np
import requests
import base64
from fastapi import FastAPI, HTTPException, Request



# img = cv2.imread("image\\23.jpg")
# base64_image = img2vec(img)

app = FastAPI()

def img2vec(img_str):
    data = str.split(str(img_str),",")[1]
    return data

def calculate_hog_features(base64_image):
    image_data = base64.b64decode(base64_image)
    nparr = np.frombuffer(image_data, np.uint8)
    img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    img_new = cv2.resize(img_gray, (128, 128), cv2.INTER_AREA)
    win_size = img_new.shape
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    hog_descriptor = hog.compute(img_new)
    
    return hog_descriptor.flatten().tolist()


@app.get("/api")
def root():
    return {"message": "This is my api"}

@app.get("/api/genhog")
async def upload_image(request : Request): 
    try:
        item =  await request.json()
        item_str = item['img']
        base64_image =img2vec(item_str)
        hog = calculate_hog_features(base64_image)
        return {"Hog": hog}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


  

