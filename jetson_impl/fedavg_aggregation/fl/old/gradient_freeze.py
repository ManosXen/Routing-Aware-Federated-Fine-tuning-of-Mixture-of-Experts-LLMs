import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from prompt_formation import prompt_formation_and_tokenize_rosetta, prompt_formation_and_tokenize_meta_math, prompt_formation_and_tokenize_code
from python_filter import is_python_example

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

batch_size=8

def expert_gradient_sorted(model, dataset, prompt_formation_and_tokenize, rank, experts_per_layer=64, num_layers=16):
    sample_size = 300

    sampled_dataset = dataset.select(range(sample_size))

    sampled_dataset = sampled_dataset.map(prompt_formation_and_tokenize, remove_columns=sampled_dataset.column_names , batched=False)
    sampled_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorForSeq2Seq(tokenizer, padding="longest")
    loader = DataLoader(sampled_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True, pin_memory=True, num_workers=4)

    model.train()
    num_batches = len(loader)

    for epoch in range(1):  # epochs
        for batch_idx, batch in enumerate(loader):

            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            del outputs
            del loss

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{num_batches} (epoch {epoch+1})")

    num_layers = num_layers
    num_experts = experts_per_layer
    num_trainable_params_per_expert=0
    if rank==0:
        for p in model.base_model.layers[0].mlp.experts[0].parameters():
            if p.requires_grad:
                num_trainable_params_per_expert+=p.grad.numel()
    else:
        for p in model.base_model.model.model.layers[0].mlp.experts[0].parameters():
            if p.requires_grad:
                num_trainable_params_per_expert+=p.grad.numel()        

    concat_grad=torch.zeros(num_layers*num_experts*num_trainable_params_per_expert, dtype=torch.bfloat16, device=DEVICE) #flatten tensor equivelant to a 3D layers*expers*trainable_params per expert
    concat_grad.requires_grad=False

    idx=0
    if rank==0:
        for layer_idx, layer in enumerate(model.base_model.layers):
            for expert_idx, expert in enumerate(layer.mlp.experts):
                for param in expert.parameters():
                    if param.requires_grad and param.grad is not None:
                        n = param.grad.numel()
                        concat_grad[idx:idx+n]=torch.abs(param.grad).view(-1)
                        idx += n
    else:
        for layer_idx, layer in enumerate(model.base_model.model.model.layers):
            for expert_idx, expert in enumerate(layer.mlp.experts):
                for param in expert.parameters():
                    if param.requires_grad and param.grad is not None:
                        n = param.grad.numel()
                        concat_grad[idx:idx+n]=torch.abs(param.grad).view(-1)
                        idx += n        

    #sum all param gradients of an expert by grouping them
    grads=torch.reshape(concat_grad, (num_layers, num_experts, num_trainable_params_per_expert))
    grad_sum=torch.sum(grads, dim=2)

    del concat_grad
    del grads
    torch.cuda.empty_cache()
    return grad_sum

def get_frozen_experts(model, percentage, dataset_path, prompt_formation_and_tokenize, rank, experts_per_layer=64, num_layers=16):
    grads=expert_gradient_sorted(model, dataset_path, prompt_formation_and_tokenize, rank, experts_per_layer, num_layers)
    grads=grads.cpu().view(-1)
    grads_sorted, grads_indx = grads.sort()
    trainable = [[] for _ in range(num_layers)]
    for i in range(int(percentage * grads_indx.shape[0]) + 1, 1024):
            layer_idx = grads_indx[i] // experts_per_layer
            expert_idx = grads_indx[i] % experts_per_layer
            trainable[layer_idx].append(expert_idx.item())

    torch.cuda.empty_cache()
    return trainable, grads_indx

def freeze_experts_gradients(model, percentage, dataset_path, prompt_formation_and_tokenize, rank, current_round, experts_per_layer=64, num_layers=16):
    if rank == 0:
        for name, param in model.named_parameters():
            if "mlp.experts" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    if current_round!=0:
        for name, param in model.named_parameters():
            if "mlp.experts" in name and "lora" in name:
                param.requires_grad=True
                
    trainable, sorted_indx = get_frozen_experts(model, percentage, dataset_path, prompt_formation_and_tokenize, rank, experts_per_layer, num_layers)
    print(trainable)
    for _, param in model.named_parameters():
        param.requires_grad = False

    if current_round==0:
        if rank == 0:
            for layer_idx, layer in enumerate(model.base_model.layers):
                if layer_idx >= len(trainable):
                    break
                for expert_idx in trainable[layer_idx]:
                    for _, param in layer.mlp.experts[expert_idx].named_parameters():
                        param.requires_grad = True
        else:
            for layer_idx, layer in enumerate(model.base_model.model.model.layers):
                if layer_idx >= len(trainable):
                    break
                for expert_idx in range(experts_per_layer):
                    if expert_idx not in trainable[layer_idx]:
                        expert = layer.mlp.experts[expert_idx]
                        for proj_name in ("gate_proj", "up_proj", "down_proj"):
                            proj = getattr(expert, proj_name, None)
                            if proj is None:
                                continue
                            base = proj.base_layer
                            setattr(expert, proj_name, base)
                    else:
                        for name, param in layer.mlp.experts[expert_idx].named_parameters():
                            if "lora" in name:
                                param.requires_grad = True

    return trainable, sorted_indx