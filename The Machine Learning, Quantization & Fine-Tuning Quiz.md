### The Machine Learning, Quantization & Fine-Tuning Quiz

---

#### Section 1: Data Types & Floating Point Precision

1\. Why does FP32 provide "7 decimal digits of accuracy" even though it can display 30+ digits after the decimal?

* A) The computer randomly generates the extra digits.  
* B) The 23-bit mantissa can only uniquely represent about 7 decimal significant figures; everything after is "noise" from binary conversion.  
* C) FP32 is designed to only be used for whole numbers.  
* D) The GPU intentionally hides the other digits to save VRAM.

2\. How does BF16 (Bfloat16) achieve the same "range" as FP32 while using only half the memory?

* A) It uses a secret compression algorithm.  
* B) It removes the sign bit.  
* C) It keeps the same 8-bit exponent as FP32 but trims the precision (mantissa) bits.  
* D) It only stores positive numbers.

3\. What is the primary cause of an "overflow" crash when using standard FP16?

* A) The GPU gets too hot.  
* B) The exponent only has 5 bits, so it cannot represent numbers higher than 65,504, turning them into NaN.  
* C) The model has too many parameters for the CPU to count.  
* D) The internet connection is too slow.

---

#### Section 2: GPU Architecture & Hardware

4\. Why does an older GPU like the Tesla T4 crash when forced to use BF16?

* A) It doesn't have enough VRAM.  
* B) It lacks the physical hardware (Compute Capability 8.0+) to perform BF16 math natively.  
* C) The T4 only supports 4-bit quantization.  
* D) Google Colab disables it to save electricity.

5\. Which of the following GPUs would be the most efficient for training a 20B model using BF16?

* A) NVIDIA Tesla T4  
* B) NVIDIA GTX 1080  
* C) NVIDIA A100  
* D) NVIDIA K80

---

#### Section 3: Quantization & Memory Math

6\. If a 20B parameter model is loaded in 4-bit (NF4), what is the "raw" weight size in VRAM?

* A) 40 GB  
* B) 20 GB  
* C) \~10 GB (Calculation: 20B parameters × 0.5 bytes per parameter)  
* D) 5 GB

7\. In your notebook, why was 19.354 GB of memory "Reserved" if the model only takes \~10 GB?

* A) The system is downloading a second model.  
* B) The extra memory acts as a buffer for Activations, KV Cache, and Optimizer States.  
* C) It is a bug in the Python code.  
* D) The GPU is reserving space for the user's browser tabs.

8\. What does "NF4" stand for, and why is it used?

* A) New Format 4; it's just the newest version.  
* B) NormalFloat 4; it's optimized for the bell-curve (normal) distribution of neural network weights.  
* C) No-Fail 4; it prevents the model from giving wrong answers.  
* D) Next-Frame 4; it's for video processing.

---

#### Section 4: Fine-Tuning (LoRA & QLoRA)

9\. In LoRA, what is the "Rank" ($r$)?

* A) The model's position on the leaderboard.  
* B) The width of the tiny "adapter" matrices that learn new information while the base model stays frozen.  
* C) The number of layers in the model.  
* D) The precision of the 4-bit weights.

10\. How does QLoRA differ from standard LoRA?

* A) QLoRA is faster but uses more memory.  
* B) QLoRA applies LoRA adapters to a model that has been quantized (usually to 4-bit).  
* C) QLoRA only works on 32-bit models.  
* D) There is no difference; they are the same thing.

11\. Why do we perform calculations in BF16/FP16 during QLoRA even though the model is stored in 4-bit?

* A) Because 4-bit math is impossible.  
* B) 4-bit storage saves space, but 16-bit precision is needed for the "math" to be accurate enough for the model to learn.  
* C) Because the A100 GPU doesn't understand 4-bit.  
* D) To make the training take longer.

---

#### Section 5: Scaling & Inference

12\. How do companies like Google run massive models (like Gemini Ultra) for millions of users?

* A) They have one giant GPU the size of a building.  
* B) They split the model across thousands of GPUs using Tensor and Pipeline Parallelism.  
* C) They convert the whole model to 1-bit.  
* D) They only run the model when a user asks a question.

13\. What is the role of the "KV Cache"?

* A) It stores the user's credit card info for billing.  
* B) It remembers the context/history of the current chat so the model doesn't re-read the whole prompt for every new word.  
* C) It speeds up the initial download of the model.  
* D) It is used to clear the GPU's "thinking" space.

---

Answer Key:

1-B, 2-C, 3-B, 4-B, 5-C, 6-C, 7-B, 8-B, 9-B, 10-B, 11-B, 12-B, 13-B.

