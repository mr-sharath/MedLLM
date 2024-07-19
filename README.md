# MedLLM: Domain-Specific Language Model Using GPT-2 🤖📚

Welcome to **MedLLM**! This project showcases how to train a small language model using the powerful distilgpt2 transformer to better understand diseases and symptoms. 🌟

## 🚀 Project Highlights

- **Dataset Loading**: Load a relevant dataset on diseases and symptoms from Hugging Face datasets.
- **Tokenization & Model Setup**: Tokenize the data using GPT-2's tokenizer and initialize the language model.
- **Training Loop**: Execute the training loop, monitor training and validation losses, and ensure effective learning.
- **Hyperparameter Tuning**: Fine-tune batch sizes, learning rates, and more for optimal model performance.
- **Text Generation**: Generate meaningful, context-aware text based on input strings using the trained model.

## 🌟 Why MedLLM?

Training a language model like MedLLM enhances understanding of relationships between diseases and symptoms, enabling the generation of informative and context-aware responses, which can be crucial for medical insights and diagnostics. 🩺💡

## 📋 Prerequisites

- Python 3.6 or higher
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- tqdm

## ⚙️ Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/mr-sharath/medllm.git
cd medllm
pip install -r requirements.txt
```

## 📂 Usage

### 1. Dataset Loading

```python
from datasets import load_dataset
dataset = load_dataset("your_dataset_name")
```

### 2. Tokenization and Model Setup

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
```

### 3. Training Loop

```python
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)
trainer.train()
```

### 4. Text Generation

```python
inputs = tokenizer("Sample input text", return_tensors="pt")
outputs = model.generate(inputs["input_ids"])
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## ©️ Copyright

© 2024 Sharath Kumar Reddy. All rights reserved.

---

Feel free to contribute, open issues, or suggest improvements! 😊✨

---

**Contact:**
Sharath Kumar Reddy  
Email: skreddykapu@uh.edu  
LinkedIn: [Sharath Kumar Reddy](https://linkedin.com/in/sharath-kumar-reddy)  
Portfolio: [mr-sharath.github.io](https://mr-sharath.github.io)
