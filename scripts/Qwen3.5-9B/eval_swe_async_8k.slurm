#!/bin/bash
##SBATCH --partition=<your-partition>
#SBATCH --job-name=qwen35_swe_8k_a
#SBATCH --output=logs/qwen35_eval_swe_8k_a.log
#SBATCH --error=logs/qwen35_eval_swe_8k_a.log
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
##SBATCH --constraint=gpu80

source ~/.bashrc

# Load your cluster's CUDA toolkit (e.g., `module load cuda/12.8`).

PROJ_DIR="${PROJECT_ROOT:-/path/to/project}"
source "$PROJ_DIR/.venv/bin/activate"

export MSWEA_SINGULARITY_EXECUTABLE=apptainer
export TMPDIR="${SCRATCH:-/tmp}/.apptainer/tmp_root"
export APPTAINER_TMPDIR="${SCRATCH:-/tmp}/.apptainer/tmp"
export APPTAINER_CACHEDIR="${SCRATCH:-/tmp}/.apptainer/cache"

pwd
nvidia-smi

wait_for_server() {
    { set +x; } 2>/dev/null
    local port=$1
    local vllm_pid=$2
    while true; do
        # Fail fast if vLLM has died — we'd otherwise poll forever on a dead server.
        if [ -n "$vllm_pid" ] && ! kill -0 "$vllm_pid" 2>/dev/null; then
            echo "ERROR: vLLM process $vllm_pid is no longer running. Aborting." >&2
            exit 1
        fi
        curl -sf http://localhost:$port/v1/models > /dev/null
        if [ $? -eq 0 ]; then
            echo "Server on port $port is ready!"
            break
        else
            echo "Waiting for server on port $port to start..."
            sleep 20
        fi
    done
}

VLLM_PORT=$(( 10000 + (SLURM_JOB_ID % 50000) ))
echo "Assigned dynamic port: VLLM_PORT=$VLLM_PORT"

CUDA_VISIBLE_DEVICES=0 vllm serve \
    Qwen/Qwen3.5-9B --reasoning-parser qwen3 \
    --enable-prefix-caching --mamba-cache-mode=align \
    --port $VLLM_PORT --max-model-len 131072 --tensor-parallel-size 1 --seed 42 &

VLLM_PID=$!

wait_for_server $VLLM_PORT $VLLM_PID

echo "Warming guided-decoding cache..."
date
curl -s -o /dev/null -X POST "http://localhost:$VLLM_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen/Qwen3.5-9B\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with a single digit.\"}],\"max_tokens\":4,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false},\"guided_choice\":[\"0\",\"1\",\"2\",\"3\",\"4\",\"5\",\"6\",\"7\",\"8\",\"9\",\"10\"]}"
echo "Warmup done."
date

nvidia-smi

cd "$PROJ_DIR"


exec mini-extra swebench \
    --config swebench_functioncall.yaml \
    --config swebench_functioncall_async.yaml \
    --config my_config.yaml \
    --config "model.model_name=Qwen/Qwen3.5-9B" \
    --config "model.model_kwargs.api_base=http://localhost:$VLLM_PORT/v1" \
    --config "model.model_kwargs.temperature=0" \
    --config "agent.step_limit=100" \
    --config "agent.summary_context_tokens=8192" \
    --config "agent.context_full_trigger=1.0" \
    --config "agent.enable_speculation=true" \
    --config "agent.speculation_length=8192" \
    --config "agent.speculation_threshold=7" \
    --config "agent.judge_model=Qwen/Qwen3.5-9B" \
    --config "agent.judge_base_url=http://localhost:$VLLM_PORT/v1" \
    --output ./results/Qwen/Qwen3.5-9B/eval_swe_8k_a/ \
    --subset verified \
    --split test \
    --workers 16 \
    --filter "(pytest|pylint|scikit-learn|astropy|sympy|matplotlib).*"

echo 'Cleaning up...'
kill $VLLM_PID
