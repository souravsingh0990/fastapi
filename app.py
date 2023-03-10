from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import spacy

app = FastAPI()
nlp = spacy.load("./NER")

class Item(BaseModel):
    text : str
    

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def read_item(data:Item):
    data = data.dict()
    text_pred = data['text']
    doc = nlp(text_pred)

    data_response = [{'text':text_pred}]

    for ent in doc.ents:

        name = ent.text
        start = ent.start_char
        end = ent.end_char
        label = ent.label_

        resp_body = [{'name': name,"start": start,"end": end, "label": label}]
        data_response.append(resp_body[0])
        
    return {'data': data_response}

   

# if __name__=="__main__":
#     uvicorn.run(app,host="127.0.0.1",port= 5000)
