import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from prompt_formation import prompt_formation_and_tokenize_rosetta, prompt_formation_and_tokenize_meta_math
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def activations_counter_per_batch(router_logits):
  layers=len(router_logits)
  experts_per_layer=router_logits[0].shape[1]
  activated_experts=8

  #print(torch.stack(router_logits, dim=0).shape)
  router_logits_stacked=torch.stack(router_logits, dim=0).to(DEVICE) #Layers x Tokens x Experts
  selected=router_logits_stacked.topk(activated_experts, dim=2).indices
  del router_logits_stacked
  selected=selected.reshape(layers, -1)

  activations=torch.zeros(layers, experts_per_layer, device=DEVICE, dtype=torch.int16)
  one=torch.ones(layers, selected.shape[1], device=DEVICE, dtype=torch.int16)
  #print(one.shape, selected.shape)
  activations.scatter_add_(dim=1, src=one, index=selected)
  return activations

def expert_activations_sorted(model, dataset, prompt_formation_and_tokenize, batch_size):
  sample_size = 300
  sampled_dataset = dataset.select(range(sample_size))

  sampled_dataset = sampled_dataset.map(prompt_formation_and_tokenize, remove_columns=sampled_dataset.column_names , batched=False)
  sampled_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

  collator = DataCollatorForSeq2Seq(tokenizer)
  loader = DataLoader(sampled_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True, pin_memory=True, num_workers=4)

  activations=torch.zeros(16, 64, device=DEVICE, dtype=torch.int32)

  num_batches = len(loader)
  total_tokens=0
  model.eval()
  with torch.no_grad():
      for batch_idx, batch in enumerate(loader):
          total_tokens+=batch['input_ids'].shape[1] * batch['input_ids'].shape[0]
          batch = {k: v.to(DEVICE) for k, v in batch.items()}
          outputs = model(**batch, output_router_logits=True)
          actv=activations_counter_per_batch(outputs.router_logits)
          activations+=actv
          del outputs
          if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{num_batches}")


  prop=activations/(total_tokens*8)
  del activations
  torch.cuda.empty_cache()  
  return prop

def get_frozen_experts(model, threshold, dataset_path, prompt_formation_and_tokenize, batch_size, experts_per_layer=64, num_layers=16):
    prop=expert_activations_sorted(model, dataset_path, prompt_formation_and_tokenize, batch_size)
    prop_sorted, indx_sorted = prop.sort(descending=True)
    trainable = [[] for _ in range(num_layers)]
    for l in range(prop_sorted.shape[0]):
        score = 0
        for e in range(prop_sorted.shape[1]):
            score += prop_sorted[l][e].item()
            trainable[l].append(indx_sorted[l][e].item())
            if score >= threshold:
                break

    torch.cuda.empty_cache()
    return trainable, indx_sorted

def freeze_experts_esft(model, percentage, dataset_path, prompt_formation_and_tokenize, rank, batch_size, current_round, experts_per_layer=64, num_layers=16):
    trainable, sorted_indx = get_frozen_experts(model, percentage, dataset_path, prompt_formation_and_tokenize, batch_size, experts_per_layer, num_layers)
    
    # for i in range(num_layers):
    #    print(f"Layer {i}:", end=" ")
    #    for e in trainable[i]:
    #       print(e, end=", ")
    #    print()    
        
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
                            # attach base in place of the wrapper
                            setattr(expert, proj_name, base)
                    else:
                        for name, param in layer.mlp.experts[expert_idx].named_parameters():
                            if "lora" in name:
                                param.requires_grad = True

    return trainable, sorted_indx