import asyncio
from joblib import load
from typing import List, Tuple

# model_path
# llama_qrf_model = load(r'qrf_model/0_qrf_lmsys_chat_llama3_8b.pkl')
# qwen_qrf_model = load(r'qrf_model/0_qrf_lmsys_chat_qwen25_7b.pkl')
# 
# llama_tokenizer = load(r'qrf_vectorizer/0_qrf_lmsys_chat_llama3_8b.pkl')
# qwen_tokenizer = load(r'qrf_vectorizer/0_qrf_lmsys_chat_qwen25_7b.pkl')

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

# async def handle_user_input(input):
#     _, _, upp = await async_predict(llama_tokenizer, llama_qrf_model, [input])
#     print(f"Prediction: {upp[0]}")

# async def main():
#     user_inputs = ["hello", "world", "asyncio why short", "example"]
# 
#     tasks = [handle_user_input(input) for input in user_inputs]
# 
#     await asyncio.gather(*tasks)
# 
# if __name__ == "__main__":
#     asyncio.run(main())