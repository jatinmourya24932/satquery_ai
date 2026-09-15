# Model weights

## BigEarthNet.txt LoRA adapter

1. After Colab Phase-1 training, download `final_adapter/` (zip).
2. Extract its contents into this folder:

```text
weights/bigearthnet_lora/
  adapter_config.json
  adapter_model.safetensors
  (optional tokenizer / processor files)
```

3. In `.env` set:

```text
SATQUERY_USE_HF=1
SATQUERY_HF_MODEL_ID=Qwen/Qwen3-VL-4B-Instruct
SATQUERY_LORA_ADAPTER=weights/bigearthnet_lora
SATQUERY_LOAD_4BIT=1
```

The app loads base Qwen3-VL once, then attaches this adapter via PEFT.
