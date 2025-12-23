### Code Structure

Our implementation is based on vLLM. We vendor a snapshot of vLLM under
`third_party/vllm/` and make minimal modifications to its scheduler to
enable pluggable scheduling policies.

All JITServe-specific logic (SLO tracking, service gain computation,
and scheduling policy) is implemented in the `jitserve/` directory,
which is independent of the vLLM codebase.

⚠️ Please ensure no preinstalled vllm exists:
pip uninstall vllm -y
