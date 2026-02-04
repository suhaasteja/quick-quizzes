  
      **LLM Fine-Tuning & Inference Quiz**

**1\. In the LLM lifecycle, where does "Fine-tuning" technically sit?**

a) Pre-training

b) Post-training

c) Architecture Design

d) Data Collection

**2\. Why is Unsloth often preferred over standard Hugging Face code for QLoRA fine-tuning?**

a) It uses fewer parameters in the model.

b) It replaces standard kernels with manually written Triton kernels for better speed and VRAM efficiency.

c) It is the only way to download models.

d) It automatically writes the training data for you.

**3\. What is the primary purpose of using an inference engine like vLLM or SGLang instead of just querying a raw model?**

a) To change the model's weights after training.

b) To increase the model's parameter count.

c) To enable high-throughput serving through features like Continuous Batching and PagedAttention.

d) To fix spelling errors in the training dataset.

**4\. How does SGLang’s "Radix Tree" (RadixAttention) improve performance in multi-turn reasoning evals?**

a) It deletes old conversation history to save space.

b) It automatically caches and reuses the KV cache of shared prompt prefixes.

c) It makes the weight matrices smaller.

d) It allows the model to search the internet faster.

**5\. If you are loading a 7B parameter model in 16-bit (BF16/FP16) precision, roughly how much VRAM do you need just for the weights?**

a) 7 GB

b) 14 GB

c) 28 GB

d) 3.5 GB

**6\. Which statement best describes the difference between FP16 and BF16?**

a) FP16 has a larger dynamic range, making it safer for training.

b) BF16 has the same dynamic range as FP32, making it more stable against "overflow" during training.

c) FP16 takes up 4 bytes per parameter, while BF16 takes 2\.

d) BF16 is only used for inference, never for fine-tuning.

**7\. What is the fundamental difference between "Static Batching" (Hugging Face) and "Continuous Batching" (vLLM/SGLang)?**

a) Static batching runs on the CPU, while continuous batching runs on the GPU.

b) Static batching waits for the longest prompt in a group to finish before starting anything new; Continuous batching inserts new requests at the token level as soon as a slot opens.

c) Continuous batching lowers the quality of the answers to generate them faster.

d) Static batching is only for fine-tuning, while continuous batching is only for training.

**8\. You want to run a massive evaluation of your model involving complex, multi-turn reasoning where the model must reference a long uploaded PDF repeatedly. Which engine is best?**

a) vLLM, because it has PagedAttention for high throughput.

b) Unsloth, because it makes the model smarter.

c) SGLang, because its Radix Tree automatically caches and reuses the PDF's prefixes across turns.

d) Hugging Face, because it is the easiest to set up.

**9\. We established that a 1B parameter model needs 2GB of VRAM in 16-bit precision. If you use 4-bit quantization (INT4), roughly how much VRAM do the weights alone require?**

a) 0.5 GB

b) 1 GB

c) 4 GB

d) It stays the same (2 GB).

**10\. What is the correct "Pro" workflow for building a custom AI application?**

a) Fine-tune in vLLM \-\> Serve with Unsloth.

b) Fine-tune in Unsloth \-\> Export \-\> Serve with vLLM or SGLang.

c) Train in SGLang \-\> Fine-tune in Hugging Face \-\> Serve in Unsloth.

d) Do everything in Google Colab Free tier forever.

**11\. Why does vLLM use "PagedAttention"?**

a) To break the KV cache into non-contiguous memory blocks (pages), preventing memory fragmentation and allowing larger batch sizes.

b) To allow the model to read pages from the internet.

c) To compress the model weights into a smaller file format.

d) To make the attention mechanism pay more attention to the end of the sentence.

**12\. If you increase the "Context Length" (the amount of text history) you feed into the model, what happens to your VRAM usage?**

a) It stays exactly the same; only model weights matter.

b) It increases, because the KV Cache grows linearly (or quadratically) with the number of tokens stored.

c) It decreases, because the model understands the context better.

d) It only increases if you are using LoRA.

**13\. Why is 16-bit precision defined as "2 bytes"?**

a) Because computer engineers like the number 2\.

b) Because 1 byte \= 4 bits, so 16 bits \= 4 bytes.

c) Because 1 byte \= 8 bits. Therefore, 16 / 8 \= 2\.

d) It is a trick question; it is actually 32 bits.

**Answers Key**

1. b  
2. b  
3. c  
4. b  
5. b  
6. b  
7. b  
8. c  
9. a  
10. b  
11. a  
12. b  
13. c