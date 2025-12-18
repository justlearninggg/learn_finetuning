# Qwen3 Fine-tuning Projects 🚀

**Medical Reasoning & LaTeX OCR with Qwen3 / Qwen3-VL**

本仓库包含基于 **Qwen3 系列模型** 的两类微调与部署实践，分别聚焦于：

* 🩺 **医疗推理场景的文本模型微调（Qwen3-1.7B）**
* 📄 **LaTeX OCR 场景的多模态模型微调（Qwen3-VL）**

项目覆盖从 **手写 Transformers 训练流程** 到 **工程化微调框架（LLaMA-Factory）**，并提供 **FastAPI 推理部署示例**。

---

## 📌 项目一：Qwen3 医疗推理模型微调

### 项目简介

在医疗问答与推理场景下，对 **Qwen3-1.7B** 进行监督微调（SFT），系统对比：

* **全量微调（Full Fine-tuning）**
* **参数高效微调（LoRA Fine-tuning）**

并分析不同训练范式在 **显存占用、训练稳定性和生成行为** 上的差异。

---

### 技术要点

* **模型**：Qwen3-1.7B
* **数据集**：`delicate_medical_r1_data`
* **训练框架**：

  * Hugging Face Transformers（手写 Trainer）
  * LLaMA-Factory（零代码 LoRA 微调）
* **训练方式**：

  * 全量微调（bf16 + Adam + gradient checkpointing）
  * LoRA 微调（r=8, α=16, dropout=0.05）
* **部署**：FastAPI + Transformers + PEFT

---

### 关键实践与结论

* 分析发现 **思维链（CoT）在 token-level 强监督下会导致输出退化**
* 调整训练策略，仅监督最终回答，显著提升生成质量
* 使用 LoRA 微调后，显存占用从 **36GB 降至 27GB（约降低 28%）**
* 在保持模型性能的前提下，大幅降低训练与部署成本

---

### 医疗项目目录结构（示例）

```text
qwen3-medical/
├── train_transformers/     # 手写 Transformers 全量微调
├── train_llamafactory/     # LLaMA-Factory LoRA 微调
├── deploy_fastapi/         # FastAPI 推理部署
├── dataset/                # 医疗推理数据
└── README.md
```

---

## 📌 项目二：Qwen3-VL LaTeX OCR 多模态微调

### 项目简介

基于 **Qwen3-VL（视觉-语言模型）**，对 LaTeX OCR 场景进行参数高效微调，实现：

> **图像 → LaTeX 公式文本** 的端到端生成

---

### 技术要点

* **模型**：Qwen3-VL
* **任务类型**：多模态 OCR（图像 + 文本生成）
* **数据集**：LaTeX OCR 数据集
* **训练方式**：LoRA 微调
* **框架**：Hugging Face Transformers

---

### 关键实践

* 构建图像与文本的多模态输入对齐
* 使用 LoRA 避免全量微调 VL 模型的高显存开销
* 支持长序列 LaTeX 生成，保证训练稳定性

---

### OCR 项目目录结构（示例）

```text
qwen3-vl-ocr/
├── train_lora/             # Qwen3-VL LoRA 微调代码
├── dataset/                # LaTeX OCR 数据
├── inference/              # 推理脚本
└── README.md
```

---

## 🚀 FastAPI 推理部署

本仓库提供基于 **FastAPI** 的推理服务示例，支持：

* Base Model + LoRA Adapter 加载
* RESTful API 调用
* 医疗问答 / OCR 推理接口

启动方式：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 🧠 项目亮点总结

* ✅ 同时覆盖 **文本模型 + 多模态模型** 微调
* ✅ 对比 **Full Fine-tuning vs LoRA Fine-tuning**
* ✅ 深入理解 **SFT 训练机制与 CoT 行为**
* ✅ 显存、效率、工程可落地性的系统分析
* ✅ 提供从训练到部署的完整链路

---

## 📚 技术栈

* Python
* PyTorch
* Hugging Face Transformers
* PEFT (LoRA)
* LLaMA-Factory
* FastAPI
* CUDA / bf16

---

## 📎 免责声明

本项目仅用于 **研究与学习目的**，不构成任何医学诊断或医疗建议。
在真实医疗场景中请遵循专业医生与监管要求。

---

你直接说要哪一个，我马上给你。
