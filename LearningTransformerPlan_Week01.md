# Week 1: The "Mechanics" Phase (Architecture & Theory)

**Goal:** Deconstruct the Transformer Architecture.
**Deliverable:** A Python script that manually calculates and visualizes Attention scores (no "black box" functions).

---

## Day 1: The Paradigm Shift (RNNs to Transformers)
**Focus:** Understand *why* Transformers exist.

* **The Problem:** Read about Recurrent Neural Networks (RNNs) and LSTMs. Understand their two fatal flaws:
    1.  **Sequential Processing:** You can't parallelize training (slow).
    2.  **Long-term Dependency:** They "forget" the beginning of a long sentence by the time they reach the end.
* **The Solution:** Transformers process the entire sequence at once (Parallelization) and use "Attention" to look at all words simultaneously.
* **Resource:** Read the first 2 pages of "Attention Is All You Need".
* **Check:** Can you explain why an RNN struggles with the sentence: *"The clouds are in the sky"* if there is a paragraph of text between "clouds" and "sky"?

## Day 2: The Heart - Self-Attention (The Math)
**Focus:** The single most important concept in modern AI.

* **Concept:** The "Database Analogy" for **Query (Q), Key (K), and Value (V)** vectors.
    * **Query:** What I'm looking for.
    * **Key:** What I define myself as.
    * **Value:** What information I hold.
* **The Formula:** Study this equation until you can write it from memory:</br>
    $$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
* **Deep Dive:**
    * Why $QK^T$? (This is a dot product, which measures *similarity* between vectors).
    * Why divide by $\sqrt{d_k}$? (To scale the values so gradients don't vanish/explode during Softmax).
* **Resource:** [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Focus specifically on the "Self-Attention" section).

## Day 3: Multi-Head Attention & Positional Encoding
**Focus:** Solving the nuances (Context & Order).

* **Multi-Head Attention:** One "head" might focus on grammar (subject-verb), another on relationships (pronouns). We need multiple heads to capture different types of context simultaneously.
* **Positional Encoding:** Since Transformers process words in parallel, they have no concept of "order" (unlike RNNs).
    * Learn why we add Sine and Cosine waves to the input embeddings to give the model a sense of position ($t=1, t=2...$).
* **Resource:** Jay Alammar’s blog (continued) or the Stanford CS25 Intro lecture.

## Day 4: The Supporting Cast (Norms & FFNs)
**Focus:** Stability and Non-linearity.

* **Residual Connections (Add):** The concept of "Skip Connections" (inherited from ResNet) allows gradients to flow through deep networks easily.
* **Layer Normalization (Norm):** Distinct from Batch Norm. Why do we normalize across the feature dimension for text?
* **Feed Forward Network (FFN):** After Attention mixes the information *between* words, the FFN processes information *within* each word individually.
* **Task:** Draw the full Transformer Block on a piece of paper: `Input -> MultiHead Attn -> Add & Norm -> FFN -> Add & Norm`.

## Day 5: The Full Architecture (Encoder vs. Decoder)
**Focus:** Differentiating the family tree.

* **The Encoder (e.g., BERT):** Uses bidirectional attention (looks at future and past). Good for understanding/classification.
* **The Decoder (e.g., GPT):** Uses **Masked** attention (can only look at the past). Good for generation.
* **The Encoder-Decoder (e.g., T5, Original Transformer):** Used for translation (Read English, Generate French).
* **Resource:** Watch Andrej Karpathy's "Let's build GPT" (First 20-30 mins for the theory recap).

## Day 6: Project Implementation - The Math Engine
**Focus:** Coding the Self-Attention mechanism using NumPy.

1.  **Step 1:** Create 3 random vectors for a "word" (Q, K, V) with dimension 4.
2.  **Step 2:** Create a sequence of 3 words (matrix shape: 3x4).
3.  **Step 3:** Compute the dot product $QK^T$.
4.  **Step 4:** Apply the mask (if simulating a decoder) - set future positions to `-infinity`.
5.  **Step 5:** Apply Softmax.

## Day 7: Project Implementation - Visualization & Review
**Focus:** Making the math visible.

1.  **Step 1:** Use `matplotlib` or `seaborn` to plot the Softmax output from Day 6 as a Heatmap.
2.  **Step 2:** Observe how high numbers in the heatmap correspond to words "paying attention" to each other.
3.  **Review:** Re-watch the Karpathy video. Now that you've coded the math, the video will make much more sense.
