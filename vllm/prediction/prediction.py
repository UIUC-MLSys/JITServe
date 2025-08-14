import asyncio
import socket
import numpy as np
import pickle
from joblib import load
from typing import List, Tuple
from vllm.request_info import RequestInfo

# Load the model and tokenizer
def load_model(model_path: str, tokenizer_path: str) -> Tuple:
    try:
        model = load(model_path)
        tokenizer = load(tokenizer_path)
    except:
        raise Exception("Model or tokenizer not found")
    return model, tokenizer

# Perform prediction on the input using the loaded model and vectorizer
def predict(vectorizer, model, input):
    X_test_vec = vectorizer.transform(input)
    y_pred_i: np.array = model.predict(X_test_vec.toarray(), quantiles=[0.95])
    return [int(element) for element in y_pred_i]

# Asynchronous function to handle prediction and return result
async def async_predict(vectorizer, model, input, request_info: RequestInfo) -> None:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, predict, vectorizer, model, input)
    request_info.output_len = result[0]


# Handle incoming connections and process requests
def handle_request(connection: socket.socket, model, vectorizer):
    try: 
        with connection:
            data = connection.recv(4096)
            if data:
                request_info = pickle.loads(data)

                result = predict(vectorizer, model, request_info['prompt'])
                result = {"output_len": result}

                # Serialize and send the result back to xx.py
                data_to_send = pickle.dumps(result)
                connection.sendall(data_to_send)
    except Exception as e:
        print(f"Error during request handling: {e}")
    finally:
        connection.close()

def start_client(model_path: str, tokenizer_path: str):
    model, tokenizer = load_model(model_path, tokenizer_path)

    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 65433))
    server_socket.listen(5)

    print("Server is listening...")

    while True:
        connection, client_address = server_socket.accept()
        print(f"Connection established with {client_address}")
        
        # Handle each request
        handle_request(connection, model, tokenizer)

model_path = '/home/jovyan/workspace/qrf_model/0_qrf_lmsys_chat_llama3_8b.pkl'
tokenizer_path = '/home/jovyan/workspace/qrf_vectorizer/0_qrf_lmsys_chat_llama3_8b.pkl'

if __name__ == "__main__":
    start_client(model_path, tokenizer_path)
