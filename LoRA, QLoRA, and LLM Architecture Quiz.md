**LoRA, QLoRA, and LLM Architecture Quiz**

### **Part 1: Low-Rank Matrices & LoRA Basics**

**1\. If a weight matrix $W$ is $4096 \\times 4096$, and we apply LoRA with a rank ($r$) of 16, what are the dimensions of the two low-rank matrices ($A$ and $B$) created?**

A) $16 \\times 16$ and $4096 \\times 4096$

B) $4096 \\times 16$ and $16 \\times 4096$ 

C) $2048 \\times 16$ and $16 \\times 2048$

D) $4096 \\times 8$ and $8 \\times 4096$

**2\. In LoRA, what happens to the original $4096 \\times 4096$ weight matrices during training?**

A) They are updated alongside the adapters.

B) They are deleted to save space.

C) They are frozen and remain unchanged.

D) They are compressed into 4-bit and then updated.

**3\. Why is the rank parameter ($r$) the primary factor in determining the final file size of your LoRA adapter?**

A) It determines the number of layers in the model.

B) It sets the precision of the weights (e.g., 4-bit vs 16-bit).

C) It directly determines the number of trainable parameters in the adapter matrices.

D) It controls the learning rate of the optimizer.

---

### **Part 2: LoRA vs. QLoRA vs. Quantization**

**4\. Why is QLoRA considered an extension of LoRA?**

A) It uses more layers than LoRA.

B) It applies LoRA adapters specifically to a model that has been pre-quantized (usually to 4-bit).

C) It is faster than regular LoRA.

D) It only works on 8-bit models.

**5\. Despite using smaller 4-bit weights, why is QLoRA roughly 39% slower to train than regular LoRA?**

A) It uses a larger learning rate.

B) It requires a constant de-quantization step (4-bit to 16-bit) during every calculation.

C) It has to update the base model weights.

D) It only works on CPUs.

**6\. Why doesn't converting 4-bit weights back to 16-bit for calculations "defeat the point" of quantization?**

A) The conversion only happens temporarily in the GPU's registers, keeping the main VRAM usage low.

B) The 16-bit version is actually smaller than the 4-bit version.

C) The conversion only happens once at the very end of training.

D) It makes the model more accurate than a full 16-bit model.

---

### **Part 3: Llama 3 Architecture & Memory**

**7\. In a Llama 3 8B model, how many transformer layers are typically present?**

A) 12

B) 32

C) 70

D) 96

**8\. Llama 3 uses a SwiGLU activation function. How many matrices does this require for the Feed-Forward (MLP) block in each layer?**

A) 1

B) 2

C) 3 (gate\_proj, up\_proj, down\_proj)

D) 4

**9\. For a 20B parameter model, approximately how much VRAM is needed just to load the model in 4-bit?**

A) 5 GB

B) 10 GB

C) 20 GB

D) 40 GB

---

### **Part 4: Training Dynamics**

**10\. What is a "sequence" in the context of LLM training?**

A) The number of layers in the model.

B) A single block of tokens (text) processed by the model at once.

C) The order in which matrices are updated.

D) The time it takes to finish one epoch.

**11\. Which new GPU architecture introduces native support for 4-bit (FP4) math, potentially eliminating the need for de-quantization?**

A) NVIDIA Ampere (A100)

B) NVIDIA Hopper (H100)

C) NVIDIA Blackwell (B200)

D) NVIDIA Turing (RTX 2080\)

**12\. If you "merge" your 4-bit QLoRA adapters back into a 16-bit (BF16) base model after training, what happens to the final file size?**

A) It stays at the 4-bit size (\~5.4GB for 8B).

B) It becomes smaller than the 4-bit version.

C) It increases to the full 16-bit size (\~16GB for 8B).

D) The file size becomes zero because it is saved in the cloud.

Key

1. B  
2. C  
3. C  
4. B  
5. B  
6. A  
7. B  
8. C  
9. B  
10. B  
11. C  
12. C

