# Benchmarking vLLM

## Downloading the BurstGPT dataset
Link: https://github.com/HPMLL/BurstGPT
Download the BurstGPT_1.csv file and place it under ./trace

## Construct the trace
`
python trace_construction.py --config-key test_long --save-path ./example-long.json
`

## Run benchmark
`
python benchmark_scheduler.py --model meta-llama/Llama-3.1-8B-Instruct --save-result --request-rate 0.2,0.8
`
