### **Section 1: Industry Frameworks**

**1\. Which framework is considered the "industry standard" for fine-tuning because it hosts almost every open-weight model?**

A) Unsloth

B) [Hugging Face (PEFT/Transformers)](https://huggingface.co/docs/peft/index)

C) Axolotl

D) Keras

**2\. If a startup has a very limited GPU budget and needs the fastest possible iteration for a single-GPU setup, which framework is currently the best choice?**

A) [Unsloth](https://unsloth.ai/)

B) Vertex AI

C) Hugging Face

D) XTuner

**3\. What is the primary advantage of using Axolotl for an enterprise AI team?**

A) It has a no-code Web UI.

B) It is faster than Unsloth on a single GPU.

C) It uses [YAML configuration files](https://www.google.com/search?q=https://github.com/OpenAccess-AI-Collective/axolotl) for reproducible, multi-GPU pipelines.

D) It is only compatible with Google TPUs.

**4\. Why would a developer choose Keras/JAX over PyTorch-based frameworks for fine-tuning Gemma?**

A) To use a Web UI for training.

B) To optimize performance for [Google Cloud TPUs](https://ai.google.dev/gemma/docs/tune#choose-framework).

C) Because PyTorch doesn't support Gemma.

D) Because Keras uses less VRAM than Unsloth.

---

### **Section 2: LoRA & PEFT Mechanics**

**5\. In LoRA (Low-Rank Adaptation), what happens to the original weights ($W$) of the model during training?**

A) They are randomly re-initialized.

B) They are updated alongside the new weights.

C) They are [frozen](https://huggingface.co/blog/gemma-peft) and never change.

D) They are deleted to save space.

**6\. How are the two new LoRA matrices ($A$ and $B$) initialized at the start of fine-tuning?**

A) Both are initialized as all zeros.

B) Both are initialized with random Gaussian noise.

C) $A$ is random noise, and $B$ is [zeros](https://huggingface.co/blog/gemma-peft).

D) $A$ is zeros, and $B$ is random noise.

**7\. If you increase the Rank ($r$) from 8 to 64, what is the most direct consequence?**

A) The model will train faster.

B) The number of [trainable parameters](https://huggingface.co/blog/gemma-peft) increases.

C) The base model weights ($W$) become smaller.

D) The model automatically switches to full fine-tuning.

**8\. Mathematically, what is the maximum possible value for the Rank ($r$)?**

A) 8

B) 64

C) The number of training examples.

D) The [Hidden Dimension ($d$)](https://www.google.com/search?q=https://ai.google.dev/gemma/docs/tune%23parameter-efficient-tuning) of the model (e.g., 2048).

**9\. What does the term "Merging" refer to after LoRA training is complete?**

A) Combining two different datasets.

B) Mathematically [adding the LoRA weights](https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune) back into the main model weights ($W\_{new} \= W \+ BA$).

C) Uploading the model to the Hugging Face Hub.

D) Running the model on both a GPU and a TPU at the same time.

---

### **Section 3: Model Architecture & Data**

**10\. In Gemma 2B, every word (token) is converted into a vector of how many numbers?**

A) 512

B) [2048](https://www.google.com/search?q=https://ai.google.dev/gemma/docs/tune%23parameter-efficient-tuning)

C) 7 billion

D) 18

**11\. Which part of a Transformer layer is responsible for understanding the context and relationships between words?**

A) The [Attention Mechanism](https://huggingface.co/blog/gemma-peft) ($Q, K, V$ matrices).

B) The MLP (Feed-Forward) matrices.

C) The Activation Function.

D) The Tokenizer.

**12\. Why do we apply LoRA to "Linear Layers" rather than "Non-linear Layers"?**

A) Non-linear layers are too fast to train.

B) Linear layers contain the [massive weight matrices](https://huggingface.co/blog/gemma-peft) that can be factorized into $A$ and $B$.

C) Linear layers are only found in the first layer.

D) Only Google models have linear layers.

**13\. What is the purpose of the "Attention Mask" during padding?**

A) It hides the correct answers from the model.

B) It tells the model to [ignore the zero-value padding tokens](https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune) so they don't affect training.

C) It compresses the 2048 vector down to 8\.

D) It encrypts the data for security.

**14\. If a Gemma 2B model has 18 layers, how many of those layers do we typically target with LoRA?**

A) Only the first and last layers.

B) Only Layer 1\.

C) [All 18 layers](https://huggingface.co/blog/gemma-peft) (usually by targeting specific module names across all layers).

D) Layers are not used in LoRA.

---

### **Section 4: Training & Hyperparameters**

**15\. If your GPU has very low VRAM but you want to train with a larger effective batch size, which parameter should you increase?**

A) learning\_rate

B) [gradient\_accumulation\_steps](https://www.google.com/search?q=%5Bhttps://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune%5D\(https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune\))

C) num\_train\_epochs

D) max\_length

**16\. What is "Catastrophic Forgetting"?**

A) When a model's file is accidentally deleted.

B) When a model [overwrites its general knowledge](https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune) because it was fine-tuned too aggressively on a small dataset.

C) When the GPU runs out of memory.

D) When the tokenizer fails to recognize a word.

**17\. If you want a model to learn a specific tone or style (like a "Martian NPC") without changing its factual knowledge, which modules are best to target?**

A) The MLP/Feed-Forward matrices (up\_proj, down\_proj).

B) Only the Embedding layer.

C) The [Attention matrices](https://huggingface.co/blog/gemma-peft) (q\_proj, v\_proj).

D) All layers except Layer 1\.

**18\. What does a "Learning Rate" that is too high usually cause?**

A) The training will be extremely slow.

B) The model will [become unstable](https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune) and output gibberish.

C) The VRAM usage will double.

D) The model will refuse to load the dataset.

**19\. What does the "Forward Pass" math look like for a LoRA-adapted layer?**

A) $h \= Wx$

B) $h \= (B \+ A)x$

C) [h \= Wx \+ (BA)x](https://huggingface.co/blog/gemma-peft)

D) $h \= W(B+A)$

**20\. According to the Google and Hugging Face guides, what is the generally recommended approach for the highest quality fine-tune?**

A) Target only the $Q$ matrix in the first layer.

B) Target [all linear layers](https://huggingface.co/blog/gemma-peft) across the entire model.

C) Target only the tokenizer.

D) Use a Rank of 1 for everything.

---

### **Answer Key**

1. **B** | 2\. **A** | 3\. **C** | 4\. **B** | 5\. **C** | 6\. **C** | 7\. **B** | 8\. **D** | 9\. **B** | 10\. **B** | 11\. **A** | 12\. **B** | 13\. **B** | 14\. **C** | 15\. **B** | 16\. **B** | 17\. **C** | 18\. **B** | 19\. **C** | 20\. **B**

