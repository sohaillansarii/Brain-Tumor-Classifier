from fastapi import FastAPI,UploadFile,File,HTTPException,Query
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow .keras.models import load_model
from PIL import Image
import cv2
import numpy as np
import io,json,os,base64

app=FastAPI(title="Brain Tumor Classification API ",version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]

)

model_dir =os.path.join(os.path.dirname(__file__), "models")

with open(os.path.join(model_dir,"metadata.json")) as f:
    metadata = json.load(f)

img_size = metadata["img_size"]
class_names = metadata["class_names"]
last_conv_layer = metadata.get("last_conv_layer","Conv_1")

model = load_model(os.path.join(model_dir,"mv2_model.keras"))

grad_model =  tf.keras.models.Model(
            model.inputs,
            [model.get_layer(last_conv_layer).output,model.output]
)

def preprocess_image (image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((img_size,img_size))
    arr = np.array(img,dtype=np.float32)/255.0
    input_arr = (arr*2)-1
    return arr,input_arr[np.newaxis,...]


def make_gradcam_heatmap(input_arr,pred_idx):
    with tf.GradientTape() as tape:
        conv_outputs,predictions = grad_model(input_arr)
        class_channel = predictions[:, pred_idx]  

    grads = tape.gradient(class_channel,conv_outputs)
    pooled_grads = tf.reduce_mean(grads,axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[...,tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap,0)/tf.math.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()


def overlay_gradcam(img_01,heatmap,alpha=0.4):
    img_uint8 = np.uint8(255 * img_01)

    heatmap = cv2.resize(heatmap , (img_01.shape[1],img_01.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
    
    overlayed = cv2.addWeighted(img_uint8,1-alpha, heatmap,alpha,0)
    return overlayed

def encode_image_base64(img_array):
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf,format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.get("/health")
def health():
    return{"status":"ok","model_loaded": model is not None}

@app.post("/predict")
async def predict(file:UploadFile= File(...),gradcam:bool= Query(False)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400,detail="Please Uplaod a JPEG or PNG image file")
    
    image_bytes = await file.read()
    try:
        img_01,input_arr = preprocess_image(image_bytes)
    except Exception :
        raise HTTPException(status_code=400,detail="Could not  read the image file.")

    probs =  model.predict(input_arr, verbose=0)[0]
    pred_idx =  int(np.argmax(probs))

    response ={
        "prediction": class_names[pred_idx],
        "confidence":float(probs[pred_idx]),
        "all_probabilities":{name: float(p) for name , p in zip(class_names,probs)},
        "disclaimer":"Educational purpose only. Not a medical advice. Please consult a medical professional for any health concerns."
    }
    if gradcam:
        heatmap = make_gradcam_heatmap(input_arr,pred_idx)
        overlayed = overlay_gradcam(img_01,heatmap)
        response["gradcam_image_base64"] = encode_image_base64(overlayed)
    
    return response


