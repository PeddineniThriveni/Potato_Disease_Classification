from fastapi import FastAPI,File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.staticfiles import StaticFiles
app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
MODEL= None
MODEL= tf.keras.models.load_model("..\Saved_models\my_model.keras")

CLASS_NAMES=["Early Blight","Late Blight","Healthy"]
@app.get("/ping")
async def ping():
    return "Hello ,I am Alive"
    

def read_file_as_image(data) ->np.ndarray:
    image=np.array(Image.open(BytesIO(data))) 
    return image
@app.post("/predict")
async def predict(
    file:UploadFile =File(...)
):
    image=read_file_as_image(await  file.read())  
    img_batch=np.expand_dims(image,0)
    print(type(MODEL)) 
    predictions=MODEL.predict(img_batch)
    predicted_class=CLASS_NAMES[np.argmax(predictions[0])]
    confidence=float(np.max(predictions[0]))
    return {
        'class' : predicted_class,
        'prediction' :confidence
    }
if __name__ =="__main__":
    app.run()