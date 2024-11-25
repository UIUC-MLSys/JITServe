import asyncio
from joblib import load
from typing import List, Tuple

def load_model(model_path: str, tokenizer_path: str) -> Tuple:
    try:
        model = load(model_path)
        tokenizer = load(tokenizer_path)
    except:
        raise Exception("Model or tokenizer not found")
    return model, tokenizer

def predict(vectorizer, model, input):
    X_test_vec = vectorizer.transform(input)
    y_pred_i = model.predict(X_test_vec.toarray(), quantiles=[0.05, 0.5, 1])
    y_pred_low = y_pred_i[:, 0]
    y_pred = y_pred_i[:, 1]
    y_pred_upp = y_pred_i[:, 2]
    return int(y_pred_upp)

async def async_predict(vectorizer, model, input):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, predict, vectorizer, model, input)