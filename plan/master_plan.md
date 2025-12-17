# 16-Week Deep Learning Mastery Plan: From Beginner to Transformer Expert

This roadmap is designed to build a rock-solid foundation in Deep Learning mechanics before catapulting you into the cutting edge of Transformers, Large Language Models (LLMs), and Multimodal AI.</br>

**Note:** Addtional topics and suggestions have been added under the sections **How to Showcase Progress (Career Strategy)** and **Suggestions for Success** so this plan can be used to prepare for **interviews** and improving one's chances of **employability**.


---

## Phase 1: The Foundations (Weeks 1-4)
**Goal:** Master the "Old Guard" (MLPs, CNNs, RNNs). You cannot understand why Transformers are revolutionary if you don't understand the architectures they replaced.

| Week | Focus | Concepts | Project & Dataset | Resources |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **The Neural Circuit** | Perceptrons, Backpropagation, Activation Functions (ReLU, Sigmoid), Loss Functions (MSE, Cross-Entropy). | **"Build a Neural Net from Scratch"**<br>_Task:_ Implement a Multi-Layer Perceptron using only NumPy (no PyTorch/TensorFlow) to classify digits.<br>_Dataset:_ [MNIST](http://yann.lecun.com/exdb/mnist/) | **Course:** Andrew Ng’s _Neural Networks and Deep Learning_ (Coursera).<br>**Read:** [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) (Michael Nielsen) - Ch 1 & 2. |
| **2** | **Computer Vision (CNNs)** | Convolutions, Pooling, Strides, Batch Normalization, Dropout.<br><br>

[Image of CNN architecture diagram]
 | **"The Eye"**<br>_Task:_ Build a Convolutional Neural Network (CNN) to classify objects. Experiment with deeper layers vs. wider layers.<br>_Dataset:_ [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | **Course:** Stanford CS231n (YouTube - 2017 version is still great for basics). |
| **3** | **Sequences (RNNs & LSTMs)** | Recurrent Neural Networks, Vanishing Gradient Problem, LSTMs/GRUs, Sequence-to-Sequence models. | **"The Time Traveler"**<br>_Task:_ Predict the next day's stock price or classify text sentiment. Note how slow training is compared to CNNs.<br>_Dataset:_ [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) | **Blog:** Colah’s Blog - _"Understanding LSTM Networks"_. |
| **4** | **The Modern Toolkit** | PyTorch/TensorFlow Mastery, Optimizers (Adam, SGD), Learning Rate Schedulers, Data Augmentation. | **"Refinement"**<br>_Task:_ Re-write your Week 2 project using modern PyTorch/Lightning practices. Achieve >85% accuracy on CIFAR-10. | **Doc:** PyTorch "Blitz" Tutorial.<br>**Video:** Andrej Karpathy - _"Building makemore Part 1-3"_. |

---

## Phase 2: The Transformer Revolution (Weeks 5-8)
**Goal:** Deep dive into the "Attention" mechanism and the architecture that changed everything.

| Week | Focus | Concepts | Project & Dataset | Resources |
| :--- | :--- | :--- | :--- | :--- |
| **5** | **Attention Mechanisms** | Seq2Seq with Attention (Bahdanau), Key/Query/Value analogy, Alignment scores.<br><br> | **"Focus"**<br>_Task:_ Build a translator (English to French) using an RNN + Attention to visualize alignment maps.<br>_Dataset:_ [Anki Bilingual Sentence Pairs](https://www.manythings.org/anki/) | **Paper:** _"Neural Machine Translation by Jointly Learning to Align and Translate"_. |
| **6** | **The Transformer (Encoder-Decoder)** | Self-Attention, Multi-Head Attention, Positional Encodings, Residual Connections, LayerNorm.<br><br> | **"The Engine"**<br>_Task:_ Implement the `MultiHeadAttention` class from scratch in PyTorch. Verify output shapes.<br>_Dataset:_ Random dummy tensors. | **Paper:** _"Attention Is All You Need"_.<br>**Video:** Karpathy’s _"Let's build GPT: from scratch"_. |
| **7** | **Encoders (BERT Family)** | Masked Language Modeling (MLM), Fine-tuning, Tokenization (WordPiece), Embeddings. | **"The Understanding Bot"**<br>_Task:_ Fine-tune a `DistilBERT` model for Named Entity Recognition (NER) or Question Answering.<br>_Dataset:_ [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Q&A). | **Course:** Hugging Face Course (Chapters 1-3).<br>**Paper:** _"BERT: Pre-training of Deep Bidirectional Transformers..."_ |
| **8** | **Decoders (GPT Family)** | Causal Masking, Autoregressive generation, Temperature, Top-K/Top-P sampling. | **"The Creator"**<br>_Task:_ Train a mini-GPT on code to auto-complete Python functions.<br>_Dataset:_ [GitHub Code Clean](https://huggingface.co/datasets/codeparrot/github-code-clean) (subset). | **Blog:** Jay Alammar’s _"The Illustrated GPT-2"_. |

---

## Phase 3: The LLM & Production Era (Weeks 9-12)
**Goal:** Move from "training small models" to "efficiently tuning massive models" using modern techniques (PEFT, RAG).

| Week | Focus | Concepts | Project & Dataset | Resources |
| :--- | :--- | :--- | :--- | :--- |
| **9** | **Efficient Fine-Tuning (PEFT)** | LoRA (Low-Rank Adaptation), QLoRA, Quantization (4-bit/8-bit), Catastrophic Forgetting. | **"Consumer-Grade LLM"**<br>_Task:_ Fine-tune a 7B model (Llama 3 or Mistral) on a free Google Colab T4 GPU using LoRA.<br>_Dataset:_ [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) or [Databricks Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k). | **Paper:** _"LoRA: Low-Rank Adaptation of Large Language Models"_. |
| **10** | **Instruction Tuning & RLHF** | Supervised Fine-Tuning (SFT) vs. RLHF, PPO, DPO (Direct Preference Optimization). | **"The Aligned Assistant"**<br>_Task:_ Use DPO to align a model to prefer polite answers over rude ones.<br>_Dataset:_ [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf). | **Blog:** Hugging Face Blog on _"RLHF"_ and _"DPO"_. |
| **11** | **RAG (Retrieval Augmented Generation)** | Vector Databases (Pinecone/Chroma), Embeddings, Semantic Search, LangChain/LlamaIndex. | **"Talk to Your Data"**<br>_Task:_ Build a Chatbot that answers questions based on a specific documentation PDF (e.g., a company handbook).<br>_Dataset:_ Custom PDF. | **Course:** DeepLearning.AI _"LangChain for LLM Application Development"_. |
| **12** | **MLOps & Deployment** | Serving models, ONNX, Docker, API wrappers (FastAPI), Gradio/Streamlit. | **"Production Ready"**<br>_Task:_ Package your Week 11 RAG bot into a Docker container and deploy it to Hugging Face Spaces.<br>_Tools:_ Docker, FastAPI. | **Resource:** _"Full Stack Deep Learning"_ (course). |

---

## Phase 4: The Frontier & Mastery (Weeks 13-16)
**Goal:** Emerging architectures and combining modalities (Images + Text).

| Week | Focus | Concepts | Project & Dataset | Resources |
| :--- | :--- | :--- | :--- | :--- |
| **13** | **Vision Transformers (ViT)** | Patching images, Linear Projections, CLS tokens for images. How Transformers conquered Vision.<br><br> | **"New Vision"**<br>_Task:_ Fine-tune a ViT for classifying X-rays or Satellite imagery.<br>_Dataset:_ [Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). | **Paper:** _"An Image is Worth 16x16 Words"_. |
| **14** | **Multimodal (CLIP & Diffusion)** | Contrastive Learning, Joint Embeddings, Stable Diffusion basics (UNet + Attention). | **"Image Search Engine"**<br>_Task:_ Build a search engine where you type text ("a dog in snow") to find matching images in a folder using CLIP.<br>_Dataset:_ [Unsplash Lite](https://github.com/unsplash/datasets). | **Paper:** _"Learning Transferable Visual Models From Natural Language Supervision"_ (CLIP). |
| **15** | **Agents & Tool Use** | ReAct pattern, Chain-of-Thought (CoT), Function Calling. | **"The Agent"**<br>_Task:_ Build an LLM agent that can use a "Calculator" tool and a "Search" tool to answer math/current event questions.<br>_Tools:_ LangGraph or AutoGen. | **Paper:** _"ReAct: Synergizing Reasoning and Acting in Language Models"_. |
| **16** | **Capstone Project** | End-to-End mastery. | **"The Masterpiece"**<br>Combine 2+ concepts (e.g., A Multimodal RAG agent that reads charts in PDFs and summarizes them). | **Showcase:** Post on Product Hunt or Reddit r/MachineLearning. |

---

## How to Showcase Progress (Career Strategy)

### 1. The "Learning in Public" Log
* **Strategy:** Every Sunday, post a "Week X Update" on LinkedIn or Twitter.
* **Content:** Don't just say "I learned X." Say "I struggled with X concept, and here is the analogy that finally made it click."
* **Why:** Employers hire problem solvers, not just coders. Showing your struggle and solution proves resilience.

### 2. The "Interactive" Portfolio
* Static GitHub code is dead. You need **Live Demos**.
* Host every project from Weeks 9-16 on **Hugging Face Spaces** or **Vercel**.
* **Resume Bullet Point:** _"Designed and deployed a RAG-based legal document analyzer using Llama-2 and Pinecone, reducing information retrieval time by 90%."_

### 3. Contributing to Open Source
* In Weeks 10-12, you will likely encounter bugs in libraries like `transformers`, `peft`, or `langchain`.
* **Action:** Fix a typo in the docs, improve an example script, or raise a well-documented issue.
* **Impact:** Having "Contributor to HuggingFace/Transformers" on your CV is a massive signal of competence.

---

## Suggestions for Success
* **Don't get stuck on Math:** You need the intuition of the math (dot products measure similarity, gradients point up hill), but you don't need to solve partial differential equations by hand.
* **Read the Papers:** For Weeks 5, 6, 7, 13, and 14, actually try to read the abstract and architecture sections of the original papers. It separates "engineers" from "researchers."
* **Join a Discord:** The "Hugging Face" or "EleutherAI" Discords are full of people building similar things.
```
