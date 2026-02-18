# FedMoECap: Routing-Aware Federated Fine-tuning of Mixture-of-Experts LLMs

**FedMoECap** is a novel Federated Learning (FL) framework designed for the efficient fine-tuning of sparse Mixture-of-Experts (MoE) models. While MoE architectures decouple model capacity from inference costs, they present significant challenges in federated settings due to their massive parameter footprint and the resource constraints (bandwidth, energy, and VRAM) of participating clients. FedMoECap bridges this gap by leveraging the inherent sparsity of experts for resource optimization.



---

## 📂 Repository Structure

This repository contains the implementation of the FedMoECap framework for two distinct hardware tiers:

* **`/jetson_impl`**: Optimized for the **NVIDIA Jetson AGX Orin** edge device.
* **`/a100_impl`**: Optimized for **NVIDIA A100/DGX** server-grade hardware.

---

## 🛠 Key Contributions

The framework introduces several novel components to optimize federated MoE fine-tuning:

* **Activation Freeze Criterion**: A novel selection criterion that pools all experts across all layers of a MoE network into a single global set. It selects the top X% based on activation rates, allowing entire layers to be frozen if necessary and providing more efficient resource allocation than traditional per-layer threshold methods.
* **Selective LoRA**: Low-Rank Adaptation (LoRA) is applied selectively only to the trainable experts identified by a manually specified per-client freeze rate. This minimizes trainable parameters and communication overhead during aggregation.
* **Heterogeneity Handling**: Each client can independently specify a freeze rate based on its local hardware capacity (e.g., available VRAM), allowing diverse devices to participate in the same training round.
* **Hybrid Aggregation**: Specifically for the "Rolling" strategy, this rule utilizes a soft-weighting approach to preserve the weights of converged clients while integrating updates from active clients, effectively preventing catastrophic forgetting.

---

## 🚀 Adaptive Convergence Management

FedMoECap provides three complementary strategies to manage expert convergence based on relative parameter updates:

1.  **FedMoECap-S (Static)**: Expert selection is performed once during the first round and remains fixed, making it ideal for strict, unchanging bandwidth constraints.
2.  **FedMoECap-P (Pruning)**: Experts nearing convergence are progressively frozen during training, continuously reducing active parameters, communication costs, and energy consumption.
3.  **FedMoECap-R (Rolling)**: Converged experts are swapped out for previously frozen, untrained experts to maximize the total number of trained experts within a fixed resource budget.

---

## 📊 Performance & Efficiency

Evaluated using the **OLMOE-1B-7B** model on **PIQA**, **BoolQ**, and **CommonsenseQA** benchmarks, FedMoECap demonstrates significant gains over standard methods:

* **Communication Overhead**: Reduced by up to **95%**.
* **Energy Consumption**: Reduced by **53%** on edge devices.
* **Accuracy**: Maintains competitive performance while operating under extreme resource constraints.

---

## 📝 Citation
If you use this work, please cite the original thesis:
