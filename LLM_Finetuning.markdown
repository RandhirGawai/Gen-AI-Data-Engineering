# LLM Finetuning

## Overview
You take a pre-trained model (already knows general language) and teach it your style or task (FAQs, support tone, domain jargon) using your examples.

## PEFT (Parameter-Efficient Fine-Tuning)
Instead of updating all model weights (expensive), you add a few tiny “adapters” (like LoRA) and only train those. You get ~95% of the quality for a tiny fraction of the cost/GPU.

## RLHF (Reinforcement Learning from Human Feedback)
You first do normal finetuning (SFT) on good examples. Then you teach the model your preferences by:
- Training a small reward model that scores answers as “good/bad” based on human ratings.
- Optimizing the chatbot to produce outputs that the reward model likes (e.g., PPO or simpler DPO).
- **Result**: The model follows instructions and aligns with your brand/tone better than SFT alone.

## Data Annotation Workflow (The Human Loop)
1. Define the task + output format (e.g., JSONL with `{"prompt":..., "response":...}`).
2. Write clear guidelines + edge cases.
3. Label a small pilot set, measure agreement, refine rules.
4. Scale labeling (tooling, QA, spot checks).
5. Version your dataset, split into train/val/test.
6. Keep collecting new tricky examples and iterate.

## GPU Utilization (Make Training Efficient)
- Use mixed precision (fp16/bf16), gradient checkpointing, and gradient accumulation.
- Use LoRA/PEFT or QLoRA (4-bit) to fit larger models.
- Keep GPUs busy: bigger effective batch, fast dataloaders, avoid CPU bottlenecks.
- Monitor with `nvidia-smi` and logs; profile if throughput is low.

## How to Do It with Code (Local or Any VM)

### A) Quick PEFT (LoRA) Finetune with Hugging Face
```python
# pip install transformers datasets peft accelerate bitsandbytes trl
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch

model_name = "meta-llama/Llama-3-8b-instruct"   # or any compatible CausalLM
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Your data should be text pairs—here we mock a simple dataset with "text"
dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

def format_example(ex):
    # turn prompt/response into a single training string
    prompt = ex.get("prompt","")
    answer = ex.get("response","")
    ex["text"] = f"### Instruction:\n{prompt}\n\n### Response:\n{answer}"
    return ex

dataset = dataset.map(format_example)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True
)

# LoRA config
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine"
)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=2048, padding="max_length")

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=collator
)
trainer.train()

# Save LoRA adapters (small files)
model.save_pretrained("lora_out")
tokenizer.save_pretrained("lora_out")
```

### B) Simple RLHF via DPO (Easier than PPO) with TRL
You need pairs: for each prompt, a chosen (better) and rejected (worse) response.
```python
# pip install trl transformers datasets peft accelerate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

base_model = "meta-llama/Llama-3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# dataset jsonl fields: prompt, chosen, rejected
dataset = load_dataset("json", data_files={"train":"dpo_train.jsonl","eval":"dpo_eval.jsonl"})

def to_fields(ex):
    return {
        "prompt": ex["prompt"],
        "chosen": ex["chosen"],
        "rejected": ex["rejected"]
    }
dataset = dataset.map(to_fields)

model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lora_cfg)

config = DPOConfig(
    output_dir="dpo_out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    num_train_epochs=1,
    bf16=True,
    beta=0.1   # strength of preference
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,   # if None, trainer will create a frozen reference from base_model
    args=config,
    beta=config.beta,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    tokenizer=tokenizer
)

trainer.train()
trainer.model.save_pretrained("dpo_lora_out")
tokenizer.save_pretrained("dpo_lora_out")
```

**Typical RLHF Pipeline**: SFT → (optional) reward model training → policy optimization (PPO/DPO). DPO skips explicit reward modeling, making it simpler.

## How to Do It on Azure (GUI + Quick Starts)

### PEFT/Finetune (GUI-first)
1. Open Azure AI Studio (or Azure ML Studio).
2. Model catalog → pick an open-source model that supports managed fine-tuning (e.g., Llama/Phi where available).
3. Click Fine-tune → choose LoRA/QLoRA if offered.
4. Attach dataset (Blob/Datastore). Format usually JSONL with prompt/response.
5. Pick Compute (GPU: e.g., A10/A100) and hyperparameters (epochs, lr, max length).
6. Submit. Track logs/metrics in the UI; when done, Deploy as an endpoint.

### RLHF on Azure (Practical Path)
- There isn’t a one-click RLHF GUI for arbitrary OSS models. Do it with Azure ML jobs or notebooks:
  1. Create a GPU Compute Cluster.
  2. Use a notebook (or a Command job) with the `trl` examples (DPO/PPO).
  3. Store data in Azure Blob; mount in the job.
  4. Log metrics to Azure ML; register the LoRA adapter as a Model; Deploy.
- (Optional) Orchestrate multi-step SFT→DPO with Azure ML Pipelines; for human feedback collection and review UIs, use Azure AI Studio data labeling or external tools.

### Data Annotation on Azure (GUI)
- Azure AI Studio has a data labeling experience for classification, NER, and custom tasks.
- For free-form chat preferences, teams often build a small web app/form (or use third-party tools), then store results in Blob/Dataverse and import into Azure ML.

## How to Do It on AWS (GUI + Quick Starts)

### PEFT/Finetune (GUI-first)
1. Open SageMaker Studio → JumpStart.
2. Choose a model (e.g., Llama 3, Mistral).
3. Click Fine-tune. Many JumpStart recipes support LoRA/QLoRA out of the box.
4. Point to your S3 dataset (JSONL).
5. Choose Instance type (g5/g6, p4/p5) and hyperparameters; Launch.
6. Monitor training in the Studio UI/CloudWatch; Deploy an endpoint (real-time or serverless inference where supported).

### RLHF on AWS (Practical Path)
- Use SageMaker Studio Notebooks with `trl` (PPO or DPO).
- (Optional) Use SageMaker RLHF containers or JumpStart examples if available for your chosen model.
- For human feedback, use SageMaker Ground Truth to collect pairwise preferences (chosen vs rejected) with a labeling job. Export JSONL to S3, then run DPO training.

### Data Annotation on AWS (GUI)
- SageMaker Ground Truth: create a Labeling job, define instructions, templates, and workforce (internal or Mechanical Turk/private).
- Use the custom task for pairwise ranking to gather RLHF preference data.
- Outputs land in S3; you can run quality metrics and iterate.

## Cloud SDK “Starter” Snippets

### SageMaker (Hugging Face + LoRA)
```python
# pip install sagemaker
import sagemaker
from sagemaker.huggingface import HuggingFace
import os

role = sagemaker.get_execution_role()
sess = sagemaker.Session()
bucket = sess.default_bucket()

# your training script (train.py) should accept args and run PEFT training like above
hyperparameters = {
  "model_name": "meta-llama/Llama-3-8b-instruct",
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "learning_rate": 2e-4,
  "num_train_epochs": 3,
  "use_lora": "true"
}

estimator = HuggingFace(
    entry_point="train.py",
    source_dir="src",
    instance_type="ml.g5.2xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.41",
    pytorch_version="2.3",
    py_version="py311",
    hyperparameters=hyperparameters
)

estimator.fit(
    inputs={
      "train": f"s3://{bucket}/datasets/train.jsonl",
      "validation": f"s3://{bucket}/datasets/val.jsonl"
    }
)
# After training: estimator.deploy(...)
```

### Azure ML (v2 SDK) Command Job for LoRA
```python
# pip install azure-ai-ml azure-identity
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import CommandJob, Environment, BuildContext

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="SUB_ID", resource_group_name="RG", workspace_name="WS"
)

env = Environment(
    build=BuildContext(path="."),  # docker context with requirements.txt
    name="lora-env"
)

job = CommandJob(
    code="./src",
    command=("python train.py "
             "--model_name meta-llama/Llama-3-8b-instruct "
             "--train_path ${{inputs.train}} --val_path ${{inputs.val}} "
             "--use_lora true --epochs 3"),
    environment=env,
    compute="gpu-cluster",
    inputs={
        "train": ml_client.data.get("train_jsonl").path,  # or azureml:// URIs
        "val": ml_client.data.get("val_jsonl").path
    },
    experiment_name="lora-finetune"
)

returned_job = ml_client.jobs.create_or_update(job)
print(returned_job.name)
```

**Note**: Your `train.py` is the same PEFT training loop shown earlier.