from fastapi import FastAPI, File, UploadFile
from tensorflow import keras
from PIL import Image
import io
from tensorflow import keras, nn, expand_dims
import numpy as np

app = FastAPI()
model = keras.models.load_model("./model.h5")

def predict(image):
    #resize, predict and return prediction
    class_names = [ f'name_to_replace_{i}' for i in range(179)]
    img_array = np.asarray(image.resize((224, 224)))[..., :3]
    img_array = np.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    probabilities = nn.softmax(predictions).numpy()[0]
    index_names = probabilities.argsort()[-5:][::-1]
    top_5_names = [class_names[i] for i in index_names if i < len(class_names)]
    top_5_probas =  [float(probabilities[i]) for i in index_names if i < len(class_names)]
    
    mushrooms = []
    
    for index, i in enumerate(top_5_names):
        mushroom = {}
        mushroom['Species'] = i
        mushroom['Probability'] = top_5_probas[index]
        mushrooms.append(mushroom)
    print(mushrooms)    
        

    
    #for k,v in dict_mushrooms.items():
     #   proba_substrate = file.get_proba_species_criteria(name,'Substrate',user_input_substrate)
       # proba_month = file.get_proba_species_criteria(name,'Month',user_input_month)
      #  proba_habitat = file.get_proba_species_criteria(name,'Habitat',user_input_habitat)
        
        #Substrate.append(proba_substrate)
        #Habitat.append(proba_habitat)
        #Month.append(proba_month)
    
    
    
    return mushrooms

def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)
    return prediction