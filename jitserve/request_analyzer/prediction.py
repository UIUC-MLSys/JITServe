"""QRF response-length predictor (Request Analyzer, paper §4.1).

Runs as a separate process so QRF inference never blocks the serving engine
(§5). Messages are length-prefixed pickles over a local TCP socket, so long
prompts are never truncated.

Online refinement: a request is first bounded by the q-quantile of the QRF's
predictive distribution for its prompt. As it generates tokens the bound is
refined to the q-quantile of that same distribution conditioned on
"length > tokens generated so far". (The paper re-invokes QRF on prompt +
generated text; the shipped QRF is trained on prompts only, and feeding it
generated text made its bounds less accurate on our traces, so conditioning
is the default. Send {'prompt': [prompt + generated_text]} without 'decoded'
to get the literal variant.)

    python -m jitserve.request_analyzer.prediction [--port 65433]
"""
import argparse
import pickle
import queue
import socket
import struct
import threading
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import numpy as np
from joblib import load

DEFAULT_PORT = 65433
DEFAULT_QUANTILE = 0.95
# Below this many forest samples above the decoded length, the conditional
# quantile is not meaningful; the caller then falls back to decoded + interval.
MIN_TAIL_SAMPLES = 5
DEFAULT_MODEL_PATH = 'assets/qrf/qrf_model/0_qrf_lmsys_chat_llama3_8b.pkl'
DEFAULT_TOKENIZER_PATH = 'assets/qrf/qrf_vectorizer/0_qrf_lmsys_chat_llama3_8b.pkl'
_HEADER = struct.Struct("!I")


# Load the model and tokenizer
def load_model(model_path: str, tokenizer_path: str) -> Tuple:
    try:
        model = load(model_path)
        tokenizer = load(tokenizer_path)
    except Exception as e:
        raise Exception(f"Model or tokenizer not found: {e}")
    return model, tokenizer


# Perform prediction on the input using the loaded model and vectorizer
def predict(vectorizer, model, input: List[str],
            quantile: float = DEFAULT_QUANTILE) -> List[int]:
    X_test_vec = vectorizer.transform(input)
    y_pred_i: np.array = model.predict(X_test_vec.toarray(), quantiles=[quantile])
    return [int(element) for element in np.ravel(y_pred_i)]


class LengthDistribution:
    """Per-prompt predictive distribution of a quantile regression forest.

    With max_samples_leaf=1 every tree contributes the training target stored
    in the leaf the prompt falls into, so the forest's prediction is the
    quantile over these n_estimators samples (identical to model.predict).
    """

    def __init__(self, vectorizer, model, quantile: float = DEFAULT_QUANTILE,
                 cache_size: int = 20000) -> None:
        self.vectorizer = vectorizer
        self.model = model
        self.quantile = quantile
        forest = model.forest_
        self._y = np.asarray(forest.y_train)[0]
        self._y_leaves = np.asarray(forest.y_train_leaves)[:, :, 0, 0]
        self._tree_idx = np.arange(self._y_leaves.shape[0])[None, :]
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_size = cache_size

    def _samples(self, prompts: Sequence[str]) -> List[np.ndarray]:
        missing = [p for p in prompts if p not in self._cache]
        if missing:
            X = self.vectorizer.transform(missing).toarray()
            leaves = self.model.apply(X)
            # y_train_leaves holds 1-based indices into y_train.
            samples = self._y[self._y_leaves[self._tree_idx, leaves] - 1]
            for prompt, ys in zip(missing, samples):
                self._cache[prompt] = np.sort(ys)
        out = []
        for prompt in prompts:
            self._cache.move_to_end(prompt)
            out.append(self._cache[prompt])
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out

    def upper_bounds(self, prompts: Sequence[str],
                     decoded: Optional[Sequence[int]] = None) -> List[int]:
        """q-quantile of the output length given the prompt and, if given,
        given that `decoded` tokens were already generated (-1 when there
        are too few samples beyond `decoded`)."""
        decoded = decoded or [0] * len(prompts)
        bounds = []
        for ys, k in zip(self._samples(prompts), decoded):
            tail = ys[ys > k] if k > 0 else ys
            if len(tail) < MIN_TAIL_SAMPLES:
                bounds.append(-1)
            else:
                bounds.append(int(np.quantile(tail, self.quantile)))
        return bounds


def send_msg(sock: socket.socket, obj) -> None:
    data = pickle.dumps(obj)
    sock.sendall(_HEADER.pack(len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket):
    header = _recv_exact(sock, _HEADER.size)
    if header is None:
        return None
    body = _recv_exact(sock, _HEADER.unpack(header)[0])
    return None if body is None else pickle.loads(body)


# Handle one persistent connection; each message is
# {'prompt': [str, ...]} or {'prompt': [...], 'decoded': [int, ...]}.
def handle_connection(connection: socket.socket, dist: LengthDistribution,
                      lock: threading.Lock) -> None:
    try:
        with connection:
            while True:
                request = recv_msg(connection)
                if request is None:
                    return
                with lock:
                    result = dist.upper_bounds(request['prompt'],
                                               request.get('decoded'))
                send_msg(connection, {"output_len": result})
    except Exception as e:
        print(f"Error during request handling: {e}")


def serve(model_path: str, tokenizer_path: str, port: int = DEFAULT_PORT,
          quantile: float = DEFAULT_QUANTILE) -> None:
    model, tokenizer = load_model(model_path, tokenizer_path)
    dist = LengthDistribution(tokenizer, model, quantile)
    lock = threading.Lock()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', port))
    server_socket.listen(128)
    print(f"Prediction server is listening on port {port} (q={quantile})",
          flush=True)

    while True:
        connection, _ = server_socket.accept()
        threading.Thread(target=handle_connection,
                         args=(connection, dist, lock),
                         daemon=True).start()


class PredictionClient:
    """Thread-safe client keeping a small pool of persistent connections."""

    def __init__(self, host: str = 'localhost', port: int = DEFAULT_PORT,
                 timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._pool: "queue.SimpleQueue[socket.socket]" = queue.SimpleQueue()

    def is_alive(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=1):
                return True
        except OSError:
            return False

    def predict(self, texts: List[str],
                decoded: Optional[List[int]] = None) -> Optional[List[int]]:
        try:
            sock = self._pool.get_nowait()
        except queue.Empty:
            try:
                sock = socket.create_connection((self.host, self.port),
                                                timeout=self.timeout)
            except OSError:
                return None
        try:
            message = {'prompt': texts}
            if decoded is not None:
                message['decoded'] = decoded
            send_msg(sock, message)
            reply = recv_msg(sock)
        except OSError:
            sock.close()
            return None
        if reply is None:
            sock.close()
            return None
        self._pool.put(sock)
        return reply['output_len']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer-path", type=str,
                        default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    args = parser.parse_args()
    serve(args.model_path, args.tokenizer_path, args.port, args.quantile)
