# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness 

Tri Dao[†] , Daniel Y. Fu[†] , Stefano Ermon[†] , Atri Rudra[‡] , and Christopher Ré[†] 

> †Department of Computer Science, Stanford University 

> ‡Department of Computer Science and Engineering, University at Buffalo, SUNY 

`{trid,danfu}@cs.stanford.edu` , `ermon@stanford.edu` , `atri@buffalo.edu` , `chrismre@cs.stanford.edu` 

June 24, 2022 

## **Abstract** 

Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms _IOaware_ —accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3× speedup on GPT-2 (seq. length 1K), and 2.4× speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy). 

## **1 Introduction** 

Transformer models [82] have emerged as the most widely used architecture in applications such as natural language processing and image classification. Transformers have grown larger [5] and deeper [83], but equipping them with longer context remains difficult [80], since the self-attention module at their heart has time and memory complexity quadratic in sequence length. An important question is whether making attention faster and more memory-efficient can help Transformer models address their runtime and memory challenges for long sequences. 

Many approximate attention methods have aimed to reduce the compute and memory requirements of attention. These methods range from sparse-approximation [51, 74] to low-rank approximation [12, 50, 84], and their combinations [3, 9, 92]. Although these methods reduce the compute requirements to linear or near-linear in sequence length, many of them do not display wall-clock speedup against standard attention and have not gained wide adoption. One main reason is that they focus on FLOP reduction (which may not correlate with wall-clock speed) and tend to ignore overheads from memory access (IO). 

In this paper, we argue that a missing principle is making attention algorithms _IO-aware_ [1]—that is, carefully accounting for reads and writes to different levels of fast and slow memory (e.g., between fast GPU on-chip SRAM and relatively slow GPU high bandwidth memory, or HBM [45], Figure 1 left). On modern 

1 

**==> picture [396 x 152] intentionally omitted <==**

**----- Start of picture text -----**<br>
Outer Loop<br>K [T] :  d x N<br>Attention on GPT-2<br>Copy Block to  SRAM<br>Q:  N x d Outer Loop V:  N X d 15 Matmul<br>GPU SRAM: 19 TB/s (20 MB)<br>SRAM Dropout<br>GPU HBM: 1.5 TB/s (40 GB) Copy 10<br>HBM Compute  Block Softmax<br>on  SRAM<br>Main Memory(CPU DRAM) DRAM                (>1 TB): 12.8 GB/s Copy 5 Mask KernelFused<br>Matmul<br>Memory Hierarchy with Output to HBM 0<br>Bandwidth & Memory Size sm (QK [T] )V:  N x d PyTorch FlashAttention<br>Inner Loop<br>FlashAttention<br>TQK N x N :  Time (ms)<br>Inner Loop Outer Loop<br>Inner Loop<br>**----- End of picture text -----**<br>


Figure 1: **Left:** FlashAttention uses tiling to prevent materialization of the large _𝑁_ × _𝑁_ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the **K** and **V** matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of **Q** matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. **Right:** Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large _𝑁_ × _𝑁_ attention matrix to HBM, resulting in an 7.6× speedup on the attention computation. 

GPUs, compute speed has out-paced memory speed [61, 62, 63], and most operations in Transformers are bottlenecked by memory accesses [43]. IO-aware algorithms have been critical for similar memory-bound operations, when reading and writing data can account for a large portion of the runtime—such as database joins [71], image processing [70], numerical linear algebra [4], and more [40, 85]. However, common Python interfaces to deep learning such as PyTorch and Tensorflow do not allow fine-grained control of memory access. 

We propose FlashAttention, a new attention algorithm that computes exact attention with far fewer memory accesses. Our main goal is to avoid reading and writing the attention matrix to and from HBM. This requires (i) computing the softmax reduction without access to the whole input (ii) not storing the large intermediate attention matrix for the backward pass. We apply two well-established techniques to address these challenges. (i) We restructure the attention computation to split the input into blocks and make several passes over input blocks, thus incrementally performing the softmax reduction (also known as **tiling** ). (ii) We store the softmax normalization factor from the forward pass to quickly **recompute** attention on-chip in the backward pass, which is faster than the standard approach of reading the intermediate attention matrix from HBM. We implement FlashAttention in CUDA to achieve fine-grained control over memory access and fuse all the attention operations into one GPU kernel. Even with the increased FLOPs due to recomputation, our algorithm both **runs faster** (up to 7.6x on GPT-2 [67], Figure 1 right) and **uses less memory** —linear in sequence length—than standard attention, thanks to the massively reduced amount of HBM access. 

We analyze the IO complexity [1] of FlashAttention, proving that it requires _𝑂_ ( _𝑁_[2] _𝑑_[2] _𝑀_[−][1] ) HBM accesses where _𝑑_ is the head dimension and _𝑀_ is the size of SRAM, as compared to Ω( _𝑁𝑑_ + _𝑁_[2] ) of standard attention. For typical values of _𝑑_ and _𝑀_ , FlashAttention requires many times fewer HBM accesses compared to standard attention (up to 9× fewer, as shown in Fig. 2). Moreover, we provide a lower bound, showing that no exact attention algorithm can asymptotically improve on the number of HBM accesses over all SRAM sizes. 

We also show that FlashAttention can serve as a useful primitive for realizing the potential of approximate attention algorithms by overcoming their issues with memory access overhead. As a proof of concept, we implement block-sparse FlashAttention, a sparse attention algorithm that is 2-4× faster than even FlashAttention, scaling up to sequence length of 64k. We prove that block-sparse FlashAttention has better IO complexity than FlashAttention by a factor proportional to the sparsity ratio. We discuss further extensions to other operations (attention on multi-GPU, kernel regression, block-sparse matrix 

2 

multiply) in Section 5. We open-source FlashAttention to make it easier to build on this primitive.[1] 

We empirically validate that FlashAttention speeds up model training and improves model quality by modeling longer context. We also benchmark the runtime and memory footprint of FlashAttention and block-sparse FlashAttention compared to prior attention implementations. 

- **Faster Model Training.** FlashAttention trains Transformer models faster in wall-clock time. We train BERT-large (seq. length 512) 15% faster than the training speed record in MLPerf 1.1 [58], GPT2 (seq. length 1K) 3× faster than baseline implementations from HuggingFace [87] and Megatron-LM [77], and long-range arena (seq. length 1K-4K) 2.4× faster than baselines. 

- **Higher Quality Models.** FlashAttention scales Transformers to longer sequences, which improves their quality and enables new capabilities. We observe a 0.7 improvement in perplexity on GPT-2 and 6.4 points of lift from modeling longer sequences on long-document classification [13]. FlashAttention enables the first Transformer that can achieve better-than-chance performance on the Path-X [80] challenge, solely from using a longer sequence length (16K). Block-sparse FlashAttention enables a Transformer to scale to even longer sequences (64K), resulting in the first model that can achieve better-than-chance performance on Path-256. 

- **Benchmarking Attention.** FlashAttention is up to 3× faster than the standard attention implementation across common sequence lengths from 128 to 2K and scales up to 64K. Up to sequence length of 512, FlashAttention is both faster and more memory-efficient than any existing attention method, whereas for sequence length beyond 1K, some approximate attention methods (e.g., Linformer) start to become faster. On the other hand, block-sparse FlashAttention is faster than all existing approximate attention methods that we know of. 

## **2 Background** 

We provide some background on the performance characteristics of common deep learning operations on modern hardware (GPUs). We also describe the standard implementation of attention. 

## **2.1 Hardware Performance** 

We focus here on GPUs. Performance on other hardware accelerators are similar [46, 48]. 

**GPU Memory Hierarchy.** The GPU memory hierarchy (Fig. 1 left) comprises multiple forms of memory of different sizes and speeds, with smaller memory being faster. As an example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s [44, 45]. The on-chip SRAM is an order of magnitude faster than HBM but many orders of magnitude smaller in size. As compute has gotten faster relative to memory speed [61, 62, 63], operations are increasingly bottlenecked by memory (HBM) accesses. Thus exploiting fast SRAM becomes more important. 

**Execution Model.** GPUs have a massive number of threads to execute an operation (called a kernel). Each kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM. 

**Performance characteristics.** Depending on the balance of computation and memory accesses, operations can be classified as either compute-bound or memory-bound. This is commonly measured by the _arithmetic intensity_ [85], which is the number of arithmetic operations per byte of memory access. 

1. Compute-bound: the time taken by the operation is determined by how many arithmetic operations there are, while time accessing HBM is much smaller. Typical examples are matrix multiply with large inner dimension, and convolution with large number of channels. 

2. Memory-bound: the time taken by the operation is determined by the number of memory accesses, while time spent in computation is much smaller. Examples include most other operations: elementwise (e.g., activation, dropout), and reduction (e.g., sum, softmax, batch norm, layer norm). **Kernel fusion.** The most common approach to accelerate memory-bound operations is kernel fusion: if 

there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of multiple times for each operation. Compilers can automatically fuse many elementwise operations [53, 65, 75]. 

> 1FlashAttention code is available at `https://github.com/HazyResearch/flash-attention` 

3 

However, in the context of model training, the intermediate values still need to be written to HBM to save for the backward pass, reducing the effectiveness of naive kernel fusion. 

## **2.2 Standard Attention Implementation** 

Given input sequences **Q** _,_ **K** _,_ **V** ∈ R _[𝑁]_[×] _[𝑑]_ where _𝑁_ is the sequence length and _𝑑_ is the head dimension, we want to compute the attention output **O** ∈ R _[𝑁]_[×] _[𝑑]_ : 

**==> picture [279 x 12] intentionally omitted <==**

where softmax is applied row-wise. 

Standard attention implementations materialize the matrices **S** and **P** to HBM, which takes _𝑂_ ( _𝑁_[2] ) memory. Often _𝑁_ ≫ _𝑑_ (e.g., for GPT2, _𝑁_ = 1024 and _𝑑_ = 64). We describe the standard attention implementation in Algorithm 0. As some or most of the operations are memory-bound (e.g., softmax), the large number of memory accesses translates to slow wall-clock time. 

This problem is exacerbated by other elementwise operations applied to the attention matrix, such as masking applied to **S** or dropout applied to **P** . As a result, there have been many attempts to fuse several elementwise operations, such as fusing masking with softmax [77]. 

In Section 3.2, we will show that the standard attention implementation performs HBM accesses quadratic in the sequence length _𝑁_ . We also compare the number of FLOPs and number of HBM accesses of standard attention and of our method (FlashAttention). 

**Algorithm 0** Standard Attention Implementation 

**Require:** Matrices **Q** _,_ **K** _,_ **V** ∈ R _[𝑁]_[×] _[𝑑]_ in HBM. 

- 1: Load **Q** _,_ **K** by blocks from HBM, compute **S** = **QK**[⊤] , write **S** to HBM. 

- 2: Read **S** from HBM, compute **P** = softmax( **S** ), write **P** to HBM. 

- 3: Load **P** and **V** by blocks from HBM, compute **O** = **PV** , write **O** to HBM. 

- 4: Return **O** . 

## **3 FlashAttention: Algorithm, Analysis, and Extensions** 

We show how to compute exact attention with fewer HBM reads/writes and without storing large intermediate matrices for the backward pass. This yields an attention algorithm that is both memory efficient and faster in wall-clock time. We analyze its IO complexity, showing that our method requires much fewer HBM accesses compared to standard attention. We further show that FlashAttention can serve as a useful primitive by extending it to handle block-sparse attention. 

We focus here on the forward pass for ease of exposition; Appendix B contains details for the backward. 

## **3.1 An Efficient Attention Algorithm With Tiling and Recomputation** 

Given the inputs **Q** _,_ **K** _,_ **V** ∈ R _[𝑁]_[×] _[𝑑]_ in HBM, we aim to compute the attention output **O** ∈ R _[𝑁]_[×] _[𝑑]_ and write it to HBM. Our goal is to reduce the amount of HBM accesses (to sub-quadratic in _𝑁_ ). 

We apply two established techniques (tiling, recomputation) to overcome the technical challenge of computing exact attention in sub-quadratic HBM accesses. We describe this in Algorithm 1. The main idea is that we split the inputs **Q** _,_ **K** _,_ **V** into blocks, load them from slow HBM to fast SRAM, then compute the attention output with respect to those blocks. By scaling the output of each block by the right normalization factor before adding them up, we get the correct result at the end. 

**Tiling.** We compute attention by blocks. Softmax couples columns of **K** , so we decompose the large softmax with scaling [51, 60, 66]. For numerical stability, the softmax of vector _𝑥_ ∈ R _[𝐵]_ is computed as: 

**==> picture [414 x 26] intentionally omitted <==**

4 

**==> picture [450 x 66] intentionally omitted <==**

Therefore if we keep track of some extra statistics ( _𝑚_ ( _𝑥_ ) _, ℓ_ ( _𝑥_ )), we can compute softmax one block at a time.[2] We thus split the inputs **Q** _,_ **K** _,_ **V** into blocks (Algorithm 1 line 3), compute the softmax values along with extra statistics (Algorithm 1 line 10), and combine the results (Algorithm 1 line 12). 

**Recomputation.** One of our goals is to not store _𝑂_ ( _𝑁_[2] ) intermediate values for the backward pass. The backward pass typically requires the matrices **S** _,_ **P** ∈ R _[𝑁]_[×] _[𝑁]_ to compute the gradients with respect to **Q** _,_ **K** _,_ **V** . However, by storing the output **O** and the softmax normalization statistics ( _𝑚, ℓ_ ), we can recompute the attention matrix **S** and **P** easily in the backward pass from blocks of **Q** _,_ **K** _,_ **V** in SRAM. This can be seen as a form of selective gradient checkpointing [10, 34]. While gradient checkpointing has been suggested to reduce the maximum amount of memory required [66], all implementations (that we know off) have to trade speed for memory. In contrast, even with more FLOPs, our recomputation speeds up the backward pass due to reduced HBM accesses (Fig. 2). The full backward pass description is in Appendix B. 

**Implementation details: Kernel fusion.** Tiling enables us to implement our algorithm in one CUDA kernel, loading input from HBM, performing all the computation steps (matrix multiply, softmax, optionally masking and dropout, matrix multiply), then write the result back to HBM (masking and dropout in Appendix B). This avoids repeatedly reading and writing of inputs and outputs from and to HBM. 

**Algorithm 1** FlashAttention 

**Require:** Matrices **Q** _,_ **K** _,_ **V** ∈ R _[𝑁]_[×] _[𝑑]_ in HBM, on-chip SRAM of size _𝑀_ . 

1: Set block sizes _𝐵𝑐_ = � 4 _𝑀𝑑_ � _, 𝐵𝑟_ = min[��] 4 _[𝑀] 𝑑_ � _, 𝑑_[�] . 2: Initialize **O** = (0) _𝑁_ × _𝑑_ ∈ R _[𝑁]_[×] _[𝑑] , ℓ_ = (0) _𝑁_ ∈ R _[𝑁] , 𝑚_ = (−∞) _𝑁_ ∈ R _[𝑁]_ in HBM. 3: Divide **Q** into _𝑇𝑟_ = � _𝐵𝑁𝑟_ � blocks **Q** 1 _, . . . ,_ **Q** _𝑇𝑟_ of size _𝐵𝑟_ × _𝑑_ each, and divide **K** _,_ **V** in to _𝑇𝑐_ = � _𝐵𝑁𝑐_ � blocks **K** 1 _, . . . ,_ **K** _𝑇𝑐_ and **V** 1 _, . . . ,_ **V** _𝑇𝑐_ , of size _𝐵𝑐_ × _𝑑_ each. 

> 4: Divide **O** into _𝑇𝑟_ blocks **O** _𝑖, . . . ,_ **O** _𝑇𝑟_ of size _𝐵𝑟_ × _𝑑_ each, divide _ℓ_ into _𝑇𝑟_ blocks _ℓ𝑖, . . . , ℓ𝑇𝑟_ of size _𝐵𝑟_ each, divide _𝑚_ into _𝑇𝑟_ blocks _𝑚_ 1 _, . . . , 𝑚𝑇𝑟_ of size _𝐵𝑟_ each. 5: **for** 1 ≤ _𝑗_ ≤ _𝑇𝑐_ **do** 6: Load **K** _𝑗 ,_ **V** _𝑗_ from HBM to on-chip SRAM. 7: **for** 1 ≤ _𝑖_ ≤ _𝑇𝑟_ **do** 8: Load **Q** _𝑖,_ **O** _𝑖, ℓ𝑖, 𝑚𝑖_ from HBM to on-chip SRAM. 9: On chip, compute **S** _𝑖𝑗_ = **Q** _𝑖_ **K** _[𝑇] 𝑗_[∈][R] _[𝐵][𝑟]_[×] _[𝐵][𝑐]_[.] ˜ ˜ 

> 10: On chip, compute _𝑚𝑖𝑗_ = rowmax( **S** _𝑖𝑗_ ) ∈ R _[𝐵][𝑟]_ , **P**[˜] _𝑖𝑗_ = exp( **S** _𝑖𝑗_ − _𝑚𝑖𝑗_ ) ∈ R _[𝐵][𝑟]_[×] _[𝐵][𝑐]_ (pointwise), _ℓ_[˜] _𝑖𝑗_ = 11: rowsumOn chip,( **P**[˜] compute _𝑖𝑗_ ) ∈ R _[𝐵][𝑟] 𝑚_ . _𝑖_[new] = max( _𝑚𝑖, 𝑚_ ˜ _𝑖𝑗_ ) ∈ R _[𝐵][𝑟]_ , _ℓ𝑖_[new] = _𝑒[𝑚]_ ˜ _[𝑖]_[−] _[𝑚] 𝑖_[new] _ℓ𝑖_ + _𝑒[𝑚]_[˜] _[𝑖𝑗]_[−] _[𝑚] 𝑖_[new] _ℓ_ ˜ _𝑖𝑗_ ∈ R _[𝐵] 𝑟_ . 12: Write **O** _𝑖_ ← diag( _ℓ𝑖_[new] )[−][1] (diag( _ℓ𝑖_ ) _𝑒[𝑚][𝑖]_[−] _[𝑚] 𝑖_[new] **O** _𝑖_ + _𝑒[𝑚]_[˜] _[𝑖𝑗]_[−] _[𝑚] 𝑖_[new] **P** _𝑖𝑗_ **V** _𝑗_ ) to HBM. 13: Write _ℓ𝑖_ ← _ℓ𝑖_[new] , _𝑚𝑖_ ← _𝑚𝑖_[new] to HBM. 14: **end for** 15: **end for** 

- 16: Return **O** . 

We show FlashAttention’s correctness, runtime, and memory requirement (proof in Appendix C). 

**Theorem 1.** _Algorithm 1 returns_ **O** = softmax( **QK**[⊤] ) **V** _with 𝑂_ ( _𝑁_[2] _𝑑_ ) _FLOPs and requires 𝑂_ ( _𝑁_ ) _additional memory beyond inputs and output._ 

## **3.2 Analysis: IO Complexity of FlashAttention** 

We analyze the IO complexity of FlashAttention, showing significant reduction in HBM accesses compared to standard attention. We also provide a lower bound, proving that no exact attention algorithm can 

> 2This style of aggregation is called _algebraic aggregation_ [33]. 

5 

**==> picture [382 x 74] intentionally omitted <==**

**----- Start of picture text -----**<br>
Effect of Block Size Sparsity Speedup<br>6 Dense<br>Attention Standard FlashAttention 6 150 FlashAttention<br>GFLOPs 66.6 75.2 4 Runtime 100<br>HBM R/W (GB) 40.3 4.4 2<br>Runtime (ms) 41.7 7.3 2 50<br>64 128 256 512 20 60<br>Block Size % Non-Zero Blocks<br>HBM<br>Accesses<br>Block-Sparse<br>FlashAttention<br>Fwd + Bwd (ms)<br>HBM Accesses (GB) Fwd Runtime (ms)<br>**----- End of picture text -----**<br>


Figure 2: **Left** : Forward + backward runtime of standard attention and FlashAttention for GPT-2 medium (seq. length 1024, head dim. 64, 16 heads, batch size 64) on A100 GPU. HBM access is the primary factor affecting runtime. **Middle** : Forward runtime of FlashAttention (seq. length 1024, head dim. 64, 16 heads, batch size 64) on A100 GPU. Fewer HBM accesses result in faster runtime, up to a point. **Right** : The runtime (for seq. length 4K) of block-sparse FlashAttention is faster than FlashAttention by a factor proportional to the sparsity. 

asymptotically improve on HBM accesses over all SRAM sizes. Proofs are in Appendix C. 

**Theorem 2.** _Let 𝑁 be the sequence length, 𝑑 be the head dimension, and 𝑀 be size of SRAM with 𝑑_ ≤ _𝑀_ ≤ _𝑁𝑑. Standard attention (Algorithm 0) requires_ Θ( _𝑁𝑑_ + _𝑁_[2] ) _HBM accesses, while_ FlashAttention _(Algorithm 1) requires_ Θ( _𝑁_[2] _𝑑_[2] _𝑀_[−][1] ) _HBM accesses._ 

For typical values of _𝑑_ (64-128) and _𝑀_ (around 100KB), _𝑑_[2] is many times smaller than _𝑀_ , and thus FlashAttention requires many times fewer HBM accesses than standard implementation. This leads to both faster execution and lower memory footprint, which we validate in Section 4.3. 

The main idea of the proof is that given the SRAM size of _𝑀_ , we can load blocks of **K** _,_ **V** of size Θ( _𝑀_ ) each (Algorithm 1 line 6). For each block of **K** and **V** , we iterate over all blocks of **Q** (Algorithm 1 line 8) to compute the intermediate values, resulting in Θ( _𝑁𝑑𝑀_[−][1] ) passes over **Q** . Each pass loads Θ( _𝑁𝑑_ ) elements, which amounts to Θ( _𝑁_[2] _𝑑_[2] _𝑀_[−][1] ) HBM accesses. We similarly prove that the backward pass of standard attention requires Θ( _𝑁𝑑_ + _𝑁_[2] ) HBM accesses while the backward pass of FlashAttention requires Θ( _𝑁_[2] _𝑑_[2] _𝑀_[−][1] ) HBM accesses (Appendix B). 

We prove a lower-bound: one cannot asymptotically improve on the number of HBM accesses for all values of _𝑀_ (the SRAM size) when computing exact attention. 

**Proposition 3.** _Let 𝑁 be the sequence length, 𝑑 be the head dimension, and 𝑀 be size of SRAM with 𝑑_ ≤ _𝑀_ ≤ _𝑁𝑑. There does not exist an algorithm to compute exact attention with 𝑜_ ( _𝑁_[2] _𝑑_[2] _𝑀_[−][1] ) _HBM accesses for all 𝑀 in the range_ [ _𝑑, 𝑁𝑑_ ] _._ 

The proof relies on the fact that for _𝑀_ = Θ( _𝑁𝑑_ ) any algorithm must perform Ω( _𝑁_[2] _𝑑_[2] _𝑀_[−][1] ) = Ω( _𝑁𝑑_ ) HBM accesses. This type of lower bound over a subrange of _𝑀_ is common in the streaming algorithms literature [88]. We leave proving parameterized complexity [27] lower bounds in terms of _𝑀_ as exciting future work. 

We validate that the number of HBM accesses is the main determining factor of attention run-time. In Fig. 2 (left), we see that even though FlashAttention has higher FLOP count compared to standard attention (due to recomputation in the backward pass), it has much fewer HBM accesses, resulting in much faster runtime. In Fig. 2 (middle), we vary the block size _𝐵𝑐_ of FlashAttention, which results in different amounts of HBM accesses, and measure the runtime of the forward pass. As block size increases, the number of HBM accesses decreases (as we make fewer passes over the input), and runtime decreases. For large enough block size (beyond 256), the runtime is then bottlenecked by other factors (e.g., arithmetic operations). Moreover, larger block size will not fit into the small SRAM size. 

## **3.3 Extension: Block-Sparse FlashAttention** 

We extend FlashAttention to approximate attention: we propose block-sparse FlashAttention, whose IO complexity is smaller than FlashAttention by a factor proportional to the sparsity. Given inputs **Q** _,_ **K** _,_ **V** ∈ R _[𝑁]_[×] _[𝑑]_ and a mask matrix **M**[˜] ∈{0 _,_ 1} _[𝑁]_[×] _[𝑁]_ , we want to compute: 

**==> picture [303 x 13] intentionally omitted <==**

where ( **S** ⊙ 𝟙 **M** ˜ ) _𝑘𝑙_ = **S** _𝑘𝑙_ if **M**[˜] _𝑘𝑙_ = 1 and −∞ if **M** _𝑘𝑙_ = 0. We require **M**[˜] to have block form: for some block sizes _𝐵𝑟 , 𝐵𝑐_ , for all _𝑘, 𝑙_ , **M**[˜] _𝑘,𝑙_ = **M** _𝑖𝑗_ with _𝑖_ = ⌊ _𝑘_ / _𝐵𝑟_ ⌋ _, 𝑗_ = ⌊ _𝑙_ / _𝐵𝑐_ ⌋ for some **M** ∈{0 _,_ 1} _[𝑁]_[/] _[𝐵][𝑟]_[×] _[𝑁]_[/] _[𝐵][𝑐]_ . 

6 

Given a predefined block sparsity mask **M** ∈{0 _,_ 1} _[𝑁]_[/] _[𝐵][𝑟]_[×] _[𝑁]_[/] _[𝐵][𝑐]_ we can easily adapt Algorithm 1 to only compute the nonzero blocks of the attention matrix. The algorithm is identical to Algorithm 1, except we skip zero blocks. We reproduce the algorithm description in Algorithm 5 in Appendix B. 

We also analyze the IO complexity of block-sparse FlashAttention. 

**Proposition 4.** _Let 𝑁 be the sequence length, 𝑑 be the head dimension, and 𝑀 be size of SRAM with 𝑑_ ≤ _𝑀_ ≤ _𝑁𝑑. Block-sparse_ FlashAttention _(Algorithm 5) requires_ Θ( _𝑁𝑑_ + _𝑁_[2] _𝑑_[2] _𝑀_[−][1] _𝑠_ ) _HBM accesses where 𝑠 is the fraction of nonzero blocks in the block-sparsity mask._ 

We see that applying block-sparsity yields a direct improvement by the sparsity to the larger term in the IO complexity. For large sequence lengths _𝑁_ , _𝑠_ is often set to _𝑁_[−][1][/][2] [11] or _𝑁_[−][1] log _𝑁_ [3, 17, 92], resulting in Θ( _𝑁_ √ _𝑁_ ) or Θ( _𝑁_ log _𝑁_ ) IO complexity. For downstream experiments, we use the fixed butterfly sparsity pattern [17], which has been shown to be able to approximate arbitrary sparsity [16]. 

In Fig. 2 (right), we validate that as the sparsity increases, the runtime of block-sparse FlashAttention improves proportionally. On the LRA benchmark, block-sparse FlashAttention achieves 2.8× speedup, while performing on par with standard attention (Section 4). 

## **4 Experiments** 

We evaluate the impact of using FlashAttention to train Transformer models. We validate two claims about training time and model accuracy, and report attention runtime and memory benchmarks. 

- **Training Speed.** FlashAttention outperforms the MLPerf 1.1 [58] speed record for BERT by 15%, and speeds up GPT-2 up to 3× over HuggingFace [87] and 1 _._ 8× over Megatron [77] over standard Transformers. FlashAttention speeds up the long-range arena (LRA) benchmark 2.4×. 

- **Quality.** FlashAttention scales Transformers to longer sequences, yielding higher quality. FlashAttention trains GPT-2 with context length 4K faster than Megatron trains GPT-2 with context length 1K, while achieving 0.7 better perplexity. Modeling longer sequences yields 6.4 points of lift on two longdocument classification tasks. Finally, FlashAttention yields the **first Transformer** that can achieve better-than-random performance on the challenging Path-X task (sequence length 16K), and block-sparse FlashAttention yields the **first sequence model** that we know of that can achieve better-than-random performance on Path-256 (sequence length 64K). 

- **Benchmarking Attention.** We measure the runtime and memory performance of FlashAttention and block-sparse FlashAttention based on sequence length. We confirm that the memory footprint of FlashAttention scales linearly with seq. length and is up to 3× faster than standard attention for common seq. lengths (up to 2K). We confirm that runtime of block-sparse FlashAttention scales linearly in seq. length and is faster than all existing approximate attention baselines. 

- Additional experiment details are in Appendix E. 

## **4.1 Faster Models with FlashAttention** 

**BERT.** FlashAttention yields the fastest single-node BERT training speed that we know of. We train a BERT-large [22] model with FlashAttention on Wikipedia. Table 1 compares our training time to the implementation from Nvidia that set the training speed record for MLPerf 1.1 [58]. Our implementation is 15% faster. 

Table 1: Training time of BERT-large, starting from the same initialization provided by the MLPerf benchmark, to reach the target accuracy of 72.0% on masked language modeling. Averaged over 10 runs on 8×A100 GPUs. 

|BERT Implementation|Training time (minutes)|
|---|---|
|Nvidia MLPerf 1.1 [58]<br>FlashAttention (ours)|20.0 ± 1.5<br>**17.4** ± 1.4|



**GPT-2.** FlashAttention yields faster training times for GPT-2 [67] on the large OpenWebtext dataset [32] than the widely used HuggingFace [87] and Megatron-LM [77] implementations. Table 2 shows up to 3× endto-end speedup compared to Huggingface and 1.7× speedup compared to Megatron-LM. FlashAttention 

7 

achieves the same perplexity as the other two implementations, as we do not change the model definition. Appendix E includes plots of the validation perplexity throughout training, confirming that FlashAttention is as numerically stable as the baselines and produces the same training / validation curves. Table 2: GPT-2 small and medium using FlashAttention achieve up to 3× speed up compared to Huggingface implementation and up to 1.7× compared to Megatron-LM. Training time reported on 8×A100s GPUs. 

|Model implementations|OpenWebText (ppl)<br>Training time (speedup)|
|---|---|
|GPT-2 small - Huggingface [87]<br>GPT-2 small - Megatron-LM [77]<br>GPT-2 small - FlashAttention|18.2<br>9.5 days (1.0×)<br>18.2<br>4.7 days (2.0×)<br>18.2<br>**2.7 days (3.5**×**)**|
|GPT-2 medium - Huggingface [87]<br>GPT-2 medium - Megatron-LM [77]<br>GPT-2 medium - FlashAttention|14.2<br>21.0 days (1.0×)<br>14.3<br>11.5 days (1.8×)<br>14.3<br>**6.9 days (3.0**×**)**|



**Long-range Arena.** We compare vanilla Transformer (with either standard implementation or FlashAttention) on the long-range arena (LRA [80]) benchmark. We measure accuracy, throughput, and training time of all models. Each task has a different sequence length varying between 1024 and 4096. We follow the implementation and experimental setting in Tay et al. [80]and Xiong et al. [90].[3] Table 3 shows that FlashAttention achieves up 2.4× speed-up compared to standard attention. Block-sparse FlashAttention is faster than all of the approximate attention methods that we have tested. 

Table 3: The performance of standard attention, FlashAttention, block-sparse FlashAttention, and approximate attention baselines on the Long-Range-Arena benchmarks. 

|Models|ListOps<br>Text<br>Retrieval<br>Image<br>Pathfnder|Avg|Speedup|
|---|---|---|---|
|Transformer<br>FlashAttention<br>Block-sparse FlashAttention|36.0<br>63.6<br>81.6<br>42.3<br>72.7<br>37.6<br>63.9<br>81.4<br>43.5<br>72.7<br>37.0<br>63.0<br>81.3<br>43.6<br>73.3|59.3<br>59.8<br>59.6|-<br>2.4×<br>**2.8**×|
|Linformer [84]<br>Linear Attention [50]<br>Performer [12]<br>Local Attention [80]<br>Reformer [51]<br>Smyrf [19]|35.6<br>55.9<br>77.7<br>37.8<br>67.6<br>38.8<br>63.2<br>80.7<br>42.6<br>72.5<br>36.8<br>63.6<br>82.2<br>42.1<br>69.9<br>36.1<br>60.2<br>76.7<br>40.6<br>66.6<br>36.5<br>63.8<br>78.5<br>39.6<br>69.4<br>36.1<br>64.1<br>79.0<br>39.6<br>70.5|54.9<br>59.6<br>58.9<br>56.0<br>57.6<br>57.9|2.5×<br>2.3×<br>1.8×<br>1.7×<br>1.3×<br>1.7×|



## **4.2 Better Models with Longer Sequences** 

**Language Modeling with Long Context.** The runtime and memory-efficiency of FlashAttention allow us to increase the context length of GPT-2 by 4× while still running faster than the optimized implementation from Megatron-LM. Table 4 shows that that GPT-2 with FlashAttention and context length 4K is still 30% faster than GPT-2 from Megatron with context length 1K, while achieving 0.7 better perplexity. 

Table 4: GPT-2 small with FlashAttention, with 4× larger context length compared to Megatron-LM, is still 30% faster while achieving 0.7 better perplexity. Training time on 8×A100 GPUs is reported. 

|Model implementations|Context length<br>OpenWebText (ppl)<br>Training time (speedup)|
|---|---|
|GPT-2 small - Megatron-LM<br>GPT-2 small - FlashAttention<br>GPT-2 small - FlashAttention<br>GPT-2 small - FlashAttention|1k<br>18.2<br>4.7 days (1.0×)<br>1k<br>18.2<br>**2.7 days (1.7**×**)**<br>2k<br>17.6<br>3.0 days (1.6×)<br>4k<br>**17.5**<br>3.6 days (1.3×)|



**Long Document Classification.** Training Transformers with longer sequences with FlashAttention improves performance on the MIMIC-III [47] and ECtHR [6, 7] datasets. MIMIC-III contains intensive care unit patient discharge summaries, each annotated with multiple labels. ECtHR contains legal cases from the 

> 3LRA accuracy results are known to be highly dependent on the tuning procedure [90]. Our reproduced baselines perform better than as reported in the original comparison [80]. 

8 

**==> picture [394 x 119] intentionally omitted <==**

**----- Start of picture text -----**<br>
Attention Runtime (Fwd Pass + Bwd Pass) Attention Memory Usage<br>10 [2]<br>Crossover Points 20 2x<br>10 [1]<br>10<br>20x<br>10 [0]<br>128 256 512 1024 2048 4096 8192 256 8K 16K 32K 64K<br>Sequence Length Sequence Length<br>FlashAttention PyTorch Attention Linformer Attention<br>Block-Sparse FlashAttention Megatron Attention OpenAI Sparse Attention<br>Runtime (ms)<br>Memory Footprint (GB)<br>**----- End of picture text -----**<br>


Figure 3: **Left:** runtime of forward pass + backward pass. **Right:** attention memory usage. 

European Court of Human Rights, each of which is mapped to articles of the Convention of Human Rights that were allegedly violaged. Both of these datasets contain very long text documents; the average number of tokens in MIMIC is 2,395 tokens, and the longest document contains 14,562 tokens, while the average and longest numbers in ECtHR are 2,197 and 49,392, respectively. We evaluate lift from increasing the sequence length of a pretrained RoBERTa model [56] (we repeat the positional embeddings, as in Beltagy et al. [3]). Table 5 shows that sequence length 16K outperforms length 512 by 4.3 points on MIMIC, and that length 8K outperforms length 512 by 8.5 points on ECtHR. The discrepancies may be due to subtle distribution shifts: MIMIC-III contains specialized medical text and thus may be more susceptible to a distribution shift in the document length, whereas ECtHR contains general language. 

Table 5: Long Document performance (micro _𝐹_ 1) at different sequence lengths using FlashAttention. 

||512<br>1024<br>2048<br>4096<br>8192<br>16384|
|---|---|
|MIMIC-III [47]<br>ECtHR [6]|52.8<br>50.7<br>51.7<br>54.6<br>56.4<br>**57.1**<br>72.2<br>74.3<br>77.1<br>78.6<br>**80.7**<br>79.2|



Table 6: We report the first Transformer model that can achieve non-random performance on Path-X and Path-256. 

|**Model**|Path-X<br>Path-256|
|---|---|
|Transformer<br>Linformer [84]<br>Linear Attention [50]<br>Performer [12]<br>Local Attention [80]<br>Reformer [51]<br>SMYRF [19]|<br><br><br><br><br><br><br><br><br><br><br><br><br>|
|FlashAttention<br>Block-sparse FlashAttention|**61.4**<br><br>56.0<br>**63.1**|



**Path-X and Path-256.** The Path-X and Path-256 benchmarks are challenging tasks from the long-range arena benchmark designed to test long context. The task is to classify whether two points in a black and white 128×128 (or 256×256) image have a path connecting them, and the images are fed to the transformer one pixel at a time. In prior work, all transformer models have either run out of memory, or only achieved random performance [80]. There has been a search for alternative architectures that can model such long context [37]. We present here the first result of Transformer models being able to solve Path-X and Path-256 (Table 6). We pretrain a transformer on Path-64, and then transfer to Path-X by spatially interpolating the positional embeddings. FlashAttention achieves 61.4 accuracy on Path-X. Additionally, block-sparse FlashAttention enables the Transformers to scale to sequence length 64K, achieving 63.1 accuracy[4] on Path-256. 

## **4.3 Benchmarking Attention** 

We vary sequence length and measure runtime and memory usage of FlashAttention and block-sparse FlashAttention against various attention baselines on one A100 GPU with 40 GB HBM, with dropout and a padding mask. We compare against reference implementations for exact attention, approximate attention, and sparse attention. We report a subset of baselines in the main body; Appendix E contains more baselines and full details. 

> 4Path-256 requires longer sequences but has relatively shorter paths than Path-X, so it is easier to obtain a higher accuracy. 

9 

**Runtime.** Figure 3 (left) reports the runtime in milliseconds of the forward + backward pass of FlashAttention and block-sparse FlashAttention compared to the baselines in exact, approximate, and sparse attention (exact numbers in Appendix E). Runtime grows quadratically with sequence length, but FlashAttention runs significantly faster than **exact attention** baselines, up to 3× faster than the PyTorch implementation. The runtimes of many approximate/sparse attention mechanisms grow linearly with sequence length, but FlashAttention still runs faster than approximate and sparse attention for short sequences due to fewer memory accesses. The **approximate attention** runtimes begin to cross over with FlashAttention at sequences between 512 and 1024. On the other hand, block-sparse FlashAttention is faster than all implementations of exact, sparse, and approximate attention that we know of, across all sequence lengths. 

**Memory Footprint.** Figure 3 (right) shows the memory footprint of FlashAttention and block-sparse FlashAttention compared to various exact, approximate, and sparse attention baselines. FlashAttention and block-sparse FlashAttention have the same memory footprint, which grows linearly with sequence length. FlashAttention is up to 20× more memory efficient than **exact attention** baselines, and is more memory-efficient than the **approximate attention** baselines. All other algorithms except for Linformer run out of memory on an A100 GPU before 64K, and FlashAttention is still 2× more efficient than Linformer. 

## **5 Limitations and Future Directions** 

We discuss limitations of our approach and future directions. Related work is given in Appendix A. 

**Compiling to CUDA.** Our current approach to building IO-aware implementations of attention requires writing a new CUDA kernel for each new attention implementation. This requires writing the attention algorithm in a considerably lower-level language than PyTorch, and requires significant engineering effort. Implementations may also not be transferrable across GPU architectures. These limitations suggest the need for a method that supports writing attention algorithms in a high-level language (e.g., PyTorch), and compiling to IO-aware implementations in CUDA—similar to efforts such as Halide in image processing [70]. 

**IO-Aware Deep Learning.** We believe that the IO-aware approach can extend beyond attention. Attention is the most memory-intensive computation in Transformers, but every layer in a deep network touches GPU HBM. We hope our work inspires IO-aware implementations of additional modules. We discuss these potential extensions in Appendix D. 

**Multi-GPU IO-Aware Methods.** Our IO-aware implementation of attention is optimal within constants for computing attention on a single GPU. However, the attention computation may be parallelizable across multiple GPUs [72]. Using multiple GPUs adds an additional layer to IO analysis—accounting for data transfer between GPUs. We hope our work inspires future work in this direction. 

## **Acknowledgments** 

Our implementation uses Apex’s FMHA code ( `https://github.com/NVIDIA/apex/tree/master/apex/ contrib/csrc/fmha` ) as a starting point. We thank Young-Jun Ko for the in-depth explanation of his FMHA implementation and for his thoughtful answers to our questions about CUDA. We thank Sabri Eyuboglu, Megan Leszczynski, Laurel Orr, Yuhuai Wu, Beidi Chen, and Xun Huang for their constructive feedback and suggestions on early drafts of the paper. We thank Markus Rabe and Charles Staats for helpful discussion of their attention algorithm. 

We gratefully acknowledge the support of NIH under No. U54EB020405 (Mobilize), NSF under Nos. CCF1763315 (Beyond Sparsity), CCF1563078 (Volume to Velocity), and 1937301 (RTML); ARL under No. W911NF-21-2-0251 (Interactive Human-AI Teaming); ONR under No. N000141712266 (Unifying Weak Supervision); ONR N00014-20-1-2480: Understanding and Applying Non-Euclidean Geometry in Machine Learning; N000142012275 (NEPTUNE); NXP, Xilinx, LETI-CEA, Intel, IBM, Microsoft, NEC, Toshiba, TSMC, ARM, Hitachi, BASF, Accenture, Ericsson, Qualcomm, Analog Devices, Google Cloud, Salesforce, Total, the HAI-GCP & HAI-Azure Cloud Credits for Research program, the Stanford Data Science Initiative (SDSI), Department of Defense (DoD) through the National Defense Science and Engineering Graduate Fellowship (NDSEG) Program, and members of the Stanford DAWN project: Facebook, Google, and VMWare. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes 

10 

notwithstanding any copyright notation thereon. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views, policies, or endorsements, either expressed or implied, of NIH, ONR, or the U.S. Government. Atri Rudra’s research is supported by NSF grant CCF-1763481. 

## **References** 

- [1] Alok Aggarwal and S Vitter, Jeffrey. The input/output complexity of sorting and related problems. _Communications of the ACM_ , 31(9):1116–1127, 1988. 

- [2] Irwan Bello. LambdaNetworks: Modeling long-range interactions without attention. _arXiv preprint arXiv:2102.08602_ , 2021. 

- [3] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. _arXiv preprint arXiv:2004.05150_ , 2020. 

- [4] L Susan Blackford, Antoine Petitet, Roldan Pozo, Karin Remington, R Clint Whaley, James Demmel, Jack Dongarra, Iain Duff, Sven Hammarling, Greg Henry, et al. An updated set of basic linear algebra subprograms (blas). _ACM Transactions on Mathematical Software_ , 28(2):135–151, 2002. 

- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- [6] Ilias Chalkidis, Ion Androutsopoulos, and Nikolaos Aletras. Neural legal judgment prediction in English. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 4317–4323, Florence, Italy, 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1424. URL `https://www.aclweb.org/anthology/P19-1424` . 

- [7] Ilias Chalkidis, Manos Fergadiotis, Dimitrios Tsarapatsanis, Nikolaos Aletras, Ion Androutsopoulos, and Prodromos Malakasiotis. Paragraph-level rationale extraction through regularization: A case study on european court of human rights cases. In _Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics_ , Mexico City, Mexico, 2021. Association for Computational Linguistics. 

- [8] Benjamin Charlier, Jean Feydy, Joan Alexis Glaunès, François-David Collin, and Ghislain Durif. Kernel operations on the gpu, with autodiff, without memory overflows. _Journal of Machine Learning Research_ , 22(74):1–6, 2021. URL `http://jmlr.org/papers/v22/20-275.html` . 

- [9] Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Atri Rudra, and Christopher Ré. Scatterbrain: Unifying sparse and low-rank attention. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2021. 

- [10] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. _arXiv preprint arXiv:1604.06174_ , 2016. 

- [11] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. _arXiv preprint arXiv:1904.10509_ , 2019. 

- [12] Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. In _International Conference on Learning Representations (ICLR)_ , 2020. 

- [13] Xiang Dai, Ilias Chalkidis, Sune Darkner, and Desmond Elliott. Revisiting transformer-based models for long document classification. _arXiv preprint arXiv:2204.06683_ , 2022. 

- [14] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive language models beyond a fixed-length context. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 2978–2988, 2019. 

11 

- [15] Tri Dao, Albert Gu, Matthew Eichhorn, Atri Rudra, and Christopher Ré. Learning fast algorithms for linear transforms using butterfly factorizations. In _International Conference on Machine Learning (ICML)_ , 2019. 

- [16] Tri Dao, Nimit Sohoni, Albert Gu, Matthew Eichhorn, Amit Blonder, Megan Leszczynski, Atri Rudra, and Christopher Ré. Kaleidoscope: An efficient, learnable representation for all structured linear maps. In _International Conference on Learning Representations (ICLR)_ , 2020. 

- [17] Tri Dao, Beidi Chen, Kaizhao Liang, Jiaming Yang, Zhao Song, Atri Rudra, and Christopher Ré. Pixelated butterfly: Simple and efficient sparse training for neural network models. In _International Conference on Learning Representations (ICLR)_ , 2022. 

- [18] Tri Dao, Beidi Chen, Nimit Sohoni, Arjun Desai, Michael Poli, Jessica Grogan, Alexander Liu, Aniruddh Rao, Atri Rudra, and Christopher Ré. Monarch: Expressive structured matrices for efficient and accurate training. In _International Conference on Machine Learning (ICML)_ , 2022. 

- [19] Giannis Daras, Nikita Kitaev, Augustus Odena, and Alexandros G Dimakis. Smyrf-efficient attention using asymmetric clustering. _Advances in Neural Information Processing Systems_ , 33:6476–6489, 2020. 

- [20] Christopher De Sa, Albert Gu, Rohan Puttagunta, Christopher Ré, and Atri Rudra. A two-pronged progress in structured dense matrix vector multiplication. In _Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms_ , pages 1060–1079. SIAM, 2018. 

- [21] Peter J Denning. The working set model for program behavior. _Communications of the ACM_ , 11(5): 323–333, 1968. 

- [22] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. 2019. 

- [23] Xin Dong, Shangyu Chen, and Sinno Jialin Pan. Learning to prune deep neural networks via layer-wise optimal brain surgeon. _arXiv preprint arXiv:1705.07565_ , 2017. 

- [24] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_ , 2020. 

- [25] Y Eidelman and I Gohberg. On a new class of structured matrices. _Integral Equations and Operator Theory_ , 34(3):293–324, 1999. 

- [26] Jean Feydy, Joan Glaunès, Benjamin Charlier, and Michael Bronstein. Fast geometric learning with symbolic matrices. _Advances in Neural Information Processing Systems_ , 33, 2020. 

- [27] Jörg Flum and Martin Grohe. _Parameterized Complexity Theory_ . Springer, 2006. 

- [28] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In _International Conference on Learning Representations_ , 2018. 

- [29] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M Roy, and Michael Carbin. Stabilizing the lottery ticket hypothesis. _arXiv preprint arXiv:1903.01611_ , 2019. 

- [30] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In _International Conference on Machine Learning_ , pages 3259–3269. PMLR, 2020. 

- [31] Karan Goel, Albert Gu, Chris Donahue, and Christopher Ré. It’s raw! audio generation with state-space models. In _International Conference on Machine Learning (ICML)_ , 2022. 

- [32] Aaron Gokaslan, Vanya Cohen, Pavlick Ellie, and Stefanie Tellex. Openwebtext corpus, 2019. 

12 

- [33] Jim Gray, Surajit Chaudhuri, Adam Bosworth, Andrew Layman, Don Reichart, Murali Venkatrao, Frank Pellow, and Hamid Pirahesh. Data cube: A relational aggregation operator generalizing group-by, cross-tab, and sub-totals. _Data mining and knowledge discovery_ , 1(1):29–53, 1997. 

- [34] Andreas Griewank and Andrea Walther. _Evaluating derivatives: principles and techniques of algorithmic differentiation_ . SIAM, 2008. 

- [35] Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Ré. Hippo: Recurrent memory with optimal polynomial projections. In _Advances in neural information processing systems (NeurIPS)_ , 2020. 

- [36] Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher Ré. Combining recurrent, convolutional, and continuous-time models with linear state space layers. _Advances in Neural Information Processing Systems_ , 34, 2021. 

- [37] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. In _The International Conference on Learning Representations (ICLR)_ , 2022. 

- [38] Song Han, Jeff Pool, John Tran, and William J Dally. Learning both weights and connections for efficient neural networks. _arXiv preprint arXiv:1506.02626_ , 2015. 

- [39] Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In _International Conference on Learning Representations_ , 2016. 

- [40] John Hennessy and David Patterson. Memory hierarchy design. _Computer Architecture: A Quantitative Approach_ , pages 390–525, 2003. 

- [41] Sara Hooker. The hardware lottery. _arXiv preprint arXiv:2009.06489_ , 2020. 

- [42] Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc V Le. Transformer quality in linear time. _arXiv preprint arXiv:2202.10447_ , 2022. 

- [43] Andrei Ivanov, Nikoli Dryden, Tal Ben-Nun, Shigang Li, and Torsten Hoefler. Data movement is all you need: A case study on optimizing transformers. _Proceedings of Machine Learning and Systems_ , 3: 711–732, 2021. 

- [44] Zhe Jia and Peter Van Sandt. Dissecting the Ampere GPU architecture via microbenchmarking. GPU Technology Conference, 2021. 

- [45] Zhe Jia, Marco Maggioni, Benjamin Staiger, and Daniele P Scarpazza. Dissecting the nvidia Volta GPU architecture via microbenchmarking. _arXiv preprint arXiv:1804.06826_ , 2018. 

- [46] Zhe Jia, Blake Tillman, Marco Maggioni, and Daniele Paolo Scarpazza. Dissecting the graphcore IPU architecture via microbenchmarking. _arXiv preprint arXiv:1912.03413_ , 2019. 

- [47] Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iii, a freely accessible critical care database. _Scientific data_ , 3(1):1–9, 2016. 

- [48] Norman P Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, et al. In-datacenter performance analysis of a tensor processing unit. In _Proceedings of the 44th annual international symposium on computer architecture_ , pages 1–12, 2017. 

- [49] Thomas Kailath, Sun-Yuan Kung, and Martin Morf. Displacement ranks of matrices and linear equations. _Journal of Mathematical Analysis and Applications_ , 68(2):395–407, 1979. 

- [50] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In _International Conference on Machine Learning_ , pages 5156–5165. PMLR, 2020. 

13 

- [51] Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In _The International Conference on Machine Learning (ICML)_ , 2020. 

- [52] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite BEDRT for self-supervised learning of language representations. In _The International Conference on Learning Representations (ICLR)_ , 2020. 

- [53] Mingzhen Li, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hailong Yang, Zhongzhi Luan, Lin Gan, Guangwen Yang, and Depei Qian. The deep learning compiler: A comprehensive survey. _IEEE Transactions on Parallel and Distributed Systems_ , 32(3):708–727, 2020. 

- [54] Valerii Likhosherstov, Krzysztof Choromanski, Jared Davis, Xingyou Song, and Adrian Weller. Sub-linear memory: How to make performers slim. _arXiv preprint arXiv:2012.11346_ , 2020. 

- [55] Ji Lin, Yongming Rao, Jiwen Lu, and Jie Zhou. Runtime neural pruning. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, _Advances in Neural Information Processing Systems_ , volume 30. Curran Associates, Inc., 2017. 

- [56] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_ , 2019. 

- [57] Xuezhe Ma, Xiang Kong, Sinong Wang, Chunting Zhou, Jonathan May, Hao Ma, and Luke Zettlemoyer. Luna: Linear unified nested attention. _Advances in Neural Information Processing Systems_ , 34, 2021. 

- [58] Peter Mattson, Christine Cheng, Gregory Diamos, Cody Coleman, Paulius Micikevicius, David Patterson, Hanlin Tang, Gu-Yeon Wei, Peter Bailis, Victor Bittorf, et al. Mlperf training benchmark. _Proceedings of Machine Learning and Systems_ , 2:336–349, 2020. 

- [59] Frank McSherry, Michael Isard, and Derek G Murray. Scalability! but at what {COST}? In _15th Workshop on Hot Topics in Operating Systems (HotOS XV)_ , 2015. 

- [60] Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. _arXiv preprint arXiv:1805.02867_ , 2018. 

- [61] NVIDIA. Nvidia Tesla V100 GPU architecture, 2017. 

- [62] NVIDIA. Nvidia A100 tensor core GPU architecture, 2020. 

- [63] NVIDIA. Nvidia H100 tensor core GPU architecture, 2022. 

- [64] D Stott Parker. Random butterfly transformations with applications in computational linear algebra. 1995. 

- [65] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. _Advances in neural information processing systems_ , 32, 2019. 

- [66] Markus N Rabe and Charles Staats. Self-attention does not need _𝑂_ ( _𝑛_[2] ) memory. _arXiv preprint arXiv:2112.05682_ , 2021. 

- [67] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 

- [68] Jack Rae and Ali Razavi. Do transformers need deep long-range memory? In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , Online, July 2020. Association for Computational Linguistics. URL `https://www.aclweb.org/anthology/2020.acl-main.672` . 

- [69] Jack W Rae, Anna Potapenko, Siddhant M Jayakumar, and Timothy P Lillicrap. Compressive transformers for long-range sequence modelling. In _The International Conference on Learning Representations (ICLR)_ , 2020. 

14 

- [70] Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, and Saman Amarasinghe. Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines. _Acm Sigplan Notices_ , 48(6):519–530, 2013. 

- [71] Raghu Ramakrishnan, Johannes Gehrke, and Johannes Gehrke. _Database management systems_ , volume 3. McGraw-Hill New York, 2003. 

- [72] Benjamin Recht and Christopher Ré. Parallel stochastic gradient algorithms for large-scale matrix completion. _Mathematical Programming Computation_ , 5(2):201–226, 2013. 

- [73] Hongyu Ren, Hanjun Dai, Zihang Dai, Mengjiao Yang, Jure Leskovec, Dale Schuurmans, and Bo Dai. Combiner: Full attention transformer with sparse computation cost. _Advances in Neural Information Processing Systems_ , 34, 2021. 

- [74] Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. _Transactions of the Association for Computational Linguistics_ , 9: 53–68, 2021. 

- [75] Amit Sabne. XLA: Compiling machine learning for peak performance. 2020. 

- [76] Victor Sanh, Thomas Wolf, and Alexander M Rush. Movement pruning: Adaptive sparsity by fine-tuning. _arXiv preprint arXiv:2005.07683_ , 2020. 

- [77] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism. _arXiv preprint arXiv:1909.08053_ , 2019. 

- [78] Vikas Sindhwani, Tara Sainath, and Sanjiv Kumar. Structured transforms for small-footprint deep learning. In _Advances in Neural Information Processing Systems_ , pages 3088–3096, 2015. 

- [79] Sainbayar Sukhbaatar, Edouard Grave, Piotr Bojanowski, and Armand Joulin. Adaptive attention span in transformers. In _Proceedings of the Annual Meeting of the Association for Computational Linguistics_ , 2019. 

- [80] Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. Long range arena: A benchmark for efficient transformers. In _International Conference on Learning Representations_ , 2020. 

- [81] Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. Efficient transformers: A survey. _arXiv preprint arXiv:2009.06732_ , 2020. 

- [82] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- [83] Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, and Furu Wei. Deepnet: Scaling transformers to 1,000 layers. _arXiv preprint arXiv:2203.00555_ , 2022. 

- [84] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. _arXiv preprint arXiv:2006.04768_ , 2020. 

- [85] Samuel Williams, Andrew Waterman, and David Patterson. Roofline: an insightful visual performance model for multicore architectures. _Communications of the ACM_ , 52(4):65–76, 2009. 

- [86] Michael E Wolf and Monica S Lam. A data locality optimizing algorithm. In _Proceedings of the ACM SIGPLAN 1991 conference on Programming language design and implementation_ , pages 30–44, 1991. 

15 

- [87] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_ , pages 38–45, Online, October 2020. Association for Computational Linguistics. URL `https://www.aclweb.org/anthology/2020.emnlp-demos.6` . 

- [88] David P Woodruff. Optimal space lower bounds for all frequency moments. In _SODA_ , volume 4, pages 167–175. Citeseer, 2004. 

- [89] Felix Wu, Angela Fan, Alexei Baevski, Yann N Dauphin, and Michael Auli. Pay less attention with lightweight and dynamic convolutions. In _The International Conference on Learning Representations (ICLR)_ , 2019. 

- [90] Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, and Vikas Singh. Nyströmformer: A nystöm-based algorithm for approximating self-attention. In _Proceedings of the AAAI Conference on Artificial Intelligence. AAAI Conference on Artificial Intelligence_ , volume 35, page 14138, 2021. 

- [91] Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zi-Hang Jiang, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. Tokens-to-token vit: Training vision transformers from scratch on imagenet. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 558–567, 2021. 

- [92] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. _Advances in Neural Information Processing Systems_ , 33, 2020. 

- [93] Shuangfei Zhai, Walter Talbott, Nitish Srivastava, Chen Huang, Hanlin Goh, Ruixiang Zhang, and Josh Susskind. An attention free transformer. _arXiv preprint arXiv:2105.14103_ , 2021. 

- [94] Chen Zhu, Wei Ping, Chaowei Xiao, Mohammad Shoeybi, Tom Goldstein, Anima Anandkumar, and Bryan Catanzaro. Long-short transformer: Efficient transformers for language and vision. _Advances in Neural Information Processing Systems_ , 34, 2021. 

16 

## **A Related Work** 

**IO-Aware Runtime Optimization.** The broad concept of optimizing for reading and writing to fast/slow memory has a long history in computer science and has been known by many names. We draw the most direct connection to the literature of analyzing I/O complexity in this work [1], but concepts of memory hierarchies are fundamental and has appeared in many forms, from the working set model [21], to data locality [86], to the Roofline model of arithmetic intensity [85], to analyses of scalability [59], to standard textbook treatments of computer architecture [40]. We hope that this work encourages the community to adopt these ideas in more parts of the deep learning stack. 

**Efficient ML Models with Structured Matrices.** Matrix multiply is the core computational bottleneck of most machine learning models. To reduce the computational complexity, there have been numerous approaches to learn over a more efficient set of matrices. These matrices are called _structured matrices_ , which have subquadratic ( _𝑜_ ( _𝑛_[2] ) for dimension _𝑛_ × _𝑛_ ) number of parameters and runtime. Most common examples of structured matrices are sparse and low-rank matrices, along with fast transforms commonly encountered in signal processing (Fourier, Chebyshev, sine/cosine, orthogonal polynomials). There have been several more general classes of structured matrices proposed in machine learning: Toeplitz-like [78], low-displacement rank [49], quasi-separable [25]). The butterfly pattern we use for our block-sparse attention is motivated by the fact that butterfly matrices [15, 64] and their products have been shown to be able to express any structured matrices with almost optimal runtime and number of parameters [16, 20]. However, even though structured matrices are efficient in theory, they have not seen wide adoption since it is hard to translate their efficiency to wall-clock speedup since dense unconstrained matrix multiply has very optimize implementation, a phenomenon known as the hardware lottery [41]. Extensions of butterfly matrices [17, 18] aimed to make butterfly matrices more hardware-friendly. 

**Sparse Training.** Our block-sparse FlashAttention can be seen as a step towards making sparse model training more efficient. Sparse models have seen success in compressing models for inference (pruning) by sparsifying the weight matrices [23, 38, 39, 55, 76]. For model training, the lottery tickets hypothesis [28, 29, 30] suggests that there are a set of small sub-networks derived from a larger dense network that performs as well as the original dense network. Out block-sparse FlashAttention can also be seen as a fixed lottery ticket in the context of attention: we fix the sparsity pattern to be the butterfly pattern through training, and observe that it performs almost as well as the (dense) FlashAttention on the Long-range Arena tasks. 

**Efficient Transformer.** Transformer-based models have become the most widely-used architecture in natural language processing [22] and computer vision [24, 91]. However, one of their computational bottlenecks is that their time and memory scales quadratic in the sequence length. There are numerous approaches to overcome this bottleneck, including approximation with hashing (i.e., sparse) such as Reformer [51] and Smyrf [19] and with low-rank approximation such as Performer [12, 54]. One can even combine sparse and low-rank approximation for better accuracy (e.g., Longformer [3], BigBird [92], Scatterbrain [9], Long-short transformer [94], Combiner [73]). Other approaches include compressing along the sequence dimension to attend to multiple tokens at once [52, 57, 79, 89]. One can also attend over the states from previous sequences to help lengthen the context (e.g., Transformer-XL [14] and Compressive Transformer [69]). We recommend the survey [81] for more details. 

There are several lines of work on developing other modules instead of attention to model longer context. HiPPO [35] and its extensions, most notably S4 [31, 36, 37] projects the history on a polynomial basis, allowing accurate reconstruction of the history through state-space models. They combine the strengths of CNNs (efficient training), RNNs (efficient inference), and continuous models (robust to change in sampling rates). LambdaNetworks [2], AFT [93] and FLASH [42] are other attempts at replacing attention in the context of image classification and language modeling. 

## **B Algorithm Details** 

We first derive the forward and backward passes of attention and show that they can be computed in a memory-efficient manner (requiring extra memory linear instead of quadratic in the sequence length). Though they reduce the amount of extra memory required, naively they still incur quadratic HBM accesses, resulting in slower execution speed. We describe the FlashAttention algorithm to implement both the forward 

17 

and the backward passes on GPUs that reduces HBM accesses, leading to both faster runtime and smaller memory footprint. 

## **B.1 Memory-efficient forward pass** 

The main challenge in making attention memory-efficient is the softmax that couples the columns of **K** (and columns of **V** ). Our approach is to compute the softmax normalization constant separately to decouple the columns. This technique [60] has been used in the literature [51, 66] to show that attention computation does not need quadratic _extra_ memory (though the number of HBM accesses is still quadratic, resulting in slow run-time). 

For simplicity, we omit here the max-shifting step during softmax. The full algorithm in Appendix B.3 contains all the steps. 

Recall that given input sequences **Q** _,_ **K** _,_ **V** ∈ R _[𝑁]_[×] _[𝑑]_ , we want to compute the attention output **O** ∈ R _[𝑁]_[×] _[𝑑]_ : 

**==> picture [280 x 12] intentionally omitted <==**

We have that _𝑆𝑖𝑗_ = _𝑞[𝑇] 𝑖[𝑘][𝑗]_[where] _[𝑞][𝑖]_[and] _[𝑘][𝑗]_[are][the] _[𝑖]_[-th][and] _[𝑗]_[-th][columns][of] **[Q]**[and] **[K]**[respectively.][Define] the normalization constants of softmax: 

**==> picture [266 x 25] intentionally omitted <==**

Let _𝑣 𝑗_ be the _𝑗_ -th column of **V** , then the _𝑖_ -th columns of the output is 

**==> picture [311 x 30] intentionally omitted <==**

We see that once _𝐿𝑖_ is computed, we can compute _𝑜𝑖_ without extra memory by repeatedly summing _𝑒[𝑞𝑇] 𝑖[𝑘𝑗] 𝐿𝑖 𝑣 𝑗_ . Therefore the forward pass can be computed with _𝑂_ ( _𝑛_ ) extra memory: 

1. Compute _𝐿𝑖_ for all _𝑖_ according to Eq. (1), which takes _𝑂_ ( _𝑛_ ) extra memory. 

2. Compute _𝑜𝑖_ for all _𝑖_ according to Eq. (2), which takes _𝑂_ ( _𝑑_ ) extra memory. 

## **B.2 Memory-efficient backward pass** 

We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe and Staats [66] suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly and show how it can be computed in a memory-efficient manner. 

Suppose that there is a scalar loss function _𝜙_ , and let the output gradient be **dO** ∈ R _[𝑛]_[×] _[𝑑]_ (where **dO** denotes _𝜕𝜙_[We][want][to][compute][the][input][gradients] **[dQ]** _[,]_ **[ dK]** _[,]_ **[ dV]**[∈][R] _[𝑛]_[×] _[𝑑]_[(where] **[dQ]** _[,]_ **[ dK]** _[,]_ **[ dV]**[denote] _[𝜕][𝜙][𝜕][𝜙][𝜕][𝜙] 𝜕_ **O**[).] _𝜕_ **Q** _[,] 𝜕_ **K** _[,] 𝜕_ **V** respectively). 

The gradient **dV** is easy to see. Applying reverse-mode autodiff by hand (aka the chain rule), we obtain (in matrix notation) **dV** = **P** _[𝑇]_ **dO** . Thus: 

**==> picture [304 x 29] intentionally omitted <==**

Since we already computed _𝐿𝑖_ , _𝑑𝑣 𝑗_ can be computed without extra memory by repeated summing. 

The gradients **dQ** and **dK** are a little more complicated. We go through the gradients **dP** and **dS** first. From Eq. (2), we have that **dP** = **dOV** _[𝑇]_ , and so: 

**==> picture [60 x 13] intentionally omitted <==**

Recall that _𝑃𝑖_ : = softmax( _𝑆𝑖_ :). Using the fact that the Jacobian of _𝑦_ = softmax( _𝑥_ ) is diag( _𝑦_ ) − _𝑦𝑦[𝑇]_ , we have that 

**==> picture [240 x 13] intentionally omitted <==**

18 

where ◦ denotes pointwise multiplication. Define 

**==> picture [359 x 31] intentionally omitted <==**

then 

**==> picture [104 x 9] intentionally omitted <==**

Hence 

_𝑑𝑆𝑖𝑗_ = _𝑃𝑖𝑗 𝑑𝑃𝑖𝑗_ − _𝐷𝑖 𝑃𝑖𝑗_ = _𝑃𝑖𝑗_ ( _𝑑𝑃𝑖𝑗_ − _𝐷𝑖_ ) _._ 

Now we can get the gradients **dQ** and **dK** . Recall that _𝑆𝑖𝑗_ = _𝑞[𝑇] 𝑖[𝑘][𝑗]_[,][so] 

Similarly, 

**==> picture [378 x 80] intentionally omitted <==**

Therefore the backward pass can also be computed with _𝑂_ ( _𝑛_ ) extra memory: 

1. Compute _𝑑𝑣 𝑗_ for all _𝑗_ according to Eq. (3), which takes _𝑂_ ( _𝑑_ ) extra memory. 

2. Compute _𝐷𝑖_ for all _𝑖_ according to Eq. (4), which takes _𝑂_ ( _𝑛_ ) extra memory. 

3. Compute _𝑑𝑞𝑖_ for all _𝑖_ according to Eq. (5), which takes _𝑂_ ( _𝑑_ ) extra memory. 

4. Compute _𝑑𝑘 𝑗_ for all _𝑗_ according to Eq. (6), which takes _𝑂_ ( _𝑑_ ) extra memory. 

## **B.3 FlashAttention: Forward Pass** 

We describe the full details of FlashAttention forward pass. Given input sequences **Q** _,_ **K** _,_ **V** ∈ R _[𝑁]_[×] _[𝑑]_ , we want to compute the attention output **O** ∈ R _[𝑁]_[×] _[𝑑]_ : 

**==> picture [360 x 29] intentionally omitted <==**

1 where _𝜏_ ∈ R is some softmax scaling (typically ~~√~~ _𝑑_[),][mask][is][some][masking][function][that][sets][some][entries][of] the input to −∞ and keep other entries the same (e.g., key padding mask when sequences in the batch don’t have the same lengths and are padded), and dropout( _𝑥, 𝑝_ ) applies dropout to _𝑥_ elementwise (i.e., output 1− _𝑥𝑝_ with probability 1 − _𝑝_ and output 0 with probability _𝑝_ for each element _𝑥_ ). 

The full algorithm is in Algorithm 2. We save the output **O** , the softmax statistics _ℓ_ and _𝑚_ , and the pseudo-random number generator state R for the backward pass. 

19 

**Algorithm 2** FlashAttention Forward Pass 

- **Require:** Matrices **Q** _,_ **K** _,_ **V** ∈ R _[𝑁]_[×] _[𝑑]_ in HBM, on-chip SRAM of size _𝑀_ , softmax scaling constant _𝜏_ ∈ R, masking function mask, dropout probability _𝑝_ drop. 

- 1: Initialize the pseudo-random number generator state R and save to HBM. 2: Set block sizes _𝐵𝑐_ = � 4 _𝑀𝑑_ � _, 𝐵𝑟_ = min[��] 4 _[𝑀] 𝑑_ � _, 𝑑_[�] . 3: Initialize **O** = (0) _𝑁_ × _𝑑_ ∈ R _[𝑁]_[×] _[𝑑] , ℓ_ = (0) _𝑁_ ∈ R _[𝑁] , 𝑚_ = (−∞) _𝑁_ ∈ R _[𝑁]_ in HBM. 4: Divide **Q** into _𝑇𝑟_ = � _𝐵𝑁𝑟_ � blocks **Q** 1 _, . . . ,_ **Q** _𝑇𝑟_ of size _𝐵𝑟_ × _𝑑_ each, and divide **K** _,_ **V** in to _𝑇𝑐_ = � _𝐵𝑁𝑐_ � blocks **K** 1 _, . . . ,_ **K** _𝑇𝑐_ and **V** 1 _, . . . ,_ **V** _𝑇𝑐_ , of size _𝐵𝑐_ × _𝑑_ each. 

- 5: Divide **O** into _𝑇𝑟_ blocks **O** _𝑖, . . . ,_ **O** _𝑇𝑟_ of size _𝐵𝑟_ × _𝑑_ each, divide _ℓ_ into _𝑇𝑟_ blocks _ℓ𝑖, . . . , ℓ𝑇𝑟_ of size _𝐵𝑟_ each, divide _𝑚_ into _𝑇𝑟_ blocks _𝑚_ 1 _, . . . , 𝑚𝑇𝑟_ of size _𝐵𝑟_ each. 

- 6: **for** 1 ≤ _𝑗_ ≤ _𝑇𝑐_ **do** 

- 7: Load **K** _𝑗 ,_ **V** _𝑗_ from HBM to on-chip SRAM. 

- 8: **for** 1 ≤ _𝑖_ ≤ _𝑇𝑟_ **do** 9: Load **Q** _𝑖,_ **O** _𝑖, ℓ𝑖, 𝑚𝑖_ from HBM to on-chip SRAM. 

- 10: On chip, compute **S** _𝑖𝑗_ = _𝜏_ **Q** _𝑖_ **K** _[𝑇] 𝑗_[∈][R] _[𝐵][𝑟]_[×] _[𝐵][𝑐]_[.] 11: On chip, compute **S**[masked] _𝑖𝑗_ = mask( **S** _𝑖𝑗_ ). ˜ ˜ 

- 12: On˜ chip, compute˜ _𝑚𝑖𝑗_ = rowmax( **S**[masked] _𝑖𝑗_ ) ∈ R _[𝐵][𝑟]_ , **P**[˜] _𝑖𝑗_ = exp( **S**[masked] _𝑖𝑗_ − _𝑚𝑖𝑗_ ) ∈ R _[𝐵][𝑟]_[×] _[𝐵][𝑐]_ (pointwise), 13: _ℓ_ On _𝑖𝑗_ =chip, rowsumcompute( **P** _𝑖𝑗_ ) ∈ _𝑚_ R _𝑖_[new] _[𝐵] 𝑟_ .= max( _𝑚𝑖, 𝑚_ ˜ _𝑖𝑗_ ) ∈ R _[𝐵][𝑟]_ , _ℓ𝑖_[new] = _𝑒[𝑚][𝑖]_[−] _[𝑚] 𝑖_[new] _ℓ𝑖_ + _𝑒[𝑚]_[˜] _[𝑖𝑗]_[−] _[𝑚] 𝑖_[new] _ℓ_ ˜ _𝑖𝑗_ ∈ R _[𝐵] 𝑟_ . 14: On chip, compute **P**[˜][dropped] _𝑖𝑗_ = dropout( **P**[˜] _𝑖𝑗 , 𝑝_ drop). ˜ 15: Write **O** _𝑖_ ← diag( _ℓ𝑖_[new] )[−][1] (diag( _ℓ𝑖_ ) _𝑒[𝑚][𝑖]_[−] _[𝑚] 𝑖_[new] **O** _𝑖_ + _𝑒[𝑚]_[˜] _[𝑖𝑗]_[−] _[𝑚] 𝑖_[new] **P**[dropped] _𝑖𝑗_ **V** _𝑗_ ) to HBM. 16: Write _ℓ𝑖_ ← _ℓ𝑖_[new] , _𝑚𝑖_ ← _𝑚𝑖_[new] to HBM. 17: **end for** 18: **end for** 

- 19: Return **O** _, ℓ, 𝑚,_ R. 

## **B.4 FlashAttention: Backward Pass** 

We describe the full details of FlashAttention backward pass. Given input sequences **Q** _,_ **K** _,_ **V** ∈ R _[𝑁]_[×] _[𝑑]_ , the output **O** ∈ R _[𝑁]_[×] _[𝑑]_ , and the output gradient **dO** , we want to compute the input gradients **dQ** _,_ **dK** _,_ **dV** ∈ R _[𝑁]_[×] _[𝑑]_ . We first describe the standard attention backward pass in Algorithm 3 for completeness. 

**Algorithm 3** Standard Attention Backward Pass 

**Require:** Matrices **Q** _,_ **K** _,_ **V** _,_ **dO** ∈ R _[𝑁]_[×] _[𝑑]_ , **P** ∈ R _[𝑁]_[×] _[𝑁]_ in HBM. 

1: Load **P** _,_ **dO** by blocks from HBM, compute **dV** = **P**[⊤] **dO** ∈ R _[𝑁]_[×] _[𝑑]_ , write **dV** to HBM. 

- 2: Load **dO** _,_ **V** by blocks from HBM, compute **dP** = **dOV**[⊤] ∈ R _[𝑁]_[×] _[𝑁]_ , write **dP** to HBM. 

- 3: Read **P** _,_ **dP** from HBM, compute **dS** ∈ R _[𝑁]_[×] _[𝑁]_ where _𝑑𝑆𝑖𝑗_ = _𝑃𝑖𝑗_ ( _𝑑𝑃𝑖𝑗_ −[�] _𝑙[𝑃] 𝑖𝑙[𝑑𝑃] 𝑖𝑙_[)][,][write] **[dS]**[to][HBM.] 

- 4: Load **dS** and **K** by blocks from HBM, compute **dQ** = **dSK** , write **dQ** to HBM. 

- 5: Load **dS** and **Q** by blocks from HBM, compute **dK** = **dS**[⊤] **Q** , write **dK** to HBM. 

- 6: Return **dQ** _,_ **dK** _,_ **dV** . 

We now make two observations about FlashAttention backward pass: 

1. We do not need to store the dropout mask of size _𝑂_ ( _𝑁_[2] ) from the forward pass. Instead, we can save the pseudo-random number generator states from the forward pass and re-generate the dropout mask in the backward pass. This allows us to only use _𝑂_ ( _𝑁_ ) extra memory. 

2. When computing the softmax gradient, we use Eq. (4) to compute _𝐷𝑖_ = _𝑃𝑖_[⊤] : _[𝑑𝑃][𝑖]_[:][without][reducing][over] _𝑃𝑖_ : and _𝑑𝑃𝑖_ : of size _𝑁_ (they might not fit into SRAM). Instead we can rewrite _𝐷𝑖_ = _𝑑𝑜_[⊤] _𝑖[𝑜][𝑖]_[and][compute] the dot product between vectors of size _𝑑_ . 

20 

The full FlashAttention backward pass algorithm is in Algorithm 4. Conceptually it is just a block version of the derivation in Appendix B.2. 

**Algorithm 4** FlashAttention Backward Pass 

**Require:** Matrices **Q** _,_ **K** _,_ **V** _,_ **O** _,_ **dO** ∈ R _[𝑁]_[×] _[𝑑]_ in HBM, vectors _ℓ, 𝑚_ ∈ R _[𝑁]_ in HBM, on-chip SRAM of size _𝑀_ , softmax scaling constant _𝜏_ ∈ R, masking function mask, dropout probability _𝑝_ drop, pseudo-random number generator state R from the forward pass. 1: Set the pseudo-random number generator state to R. 2: Set block sizes _𝐵𝑐_ = � 4 _𝑀𝑑_ � _, 𝐵𝑟_ = min[��] 4 _[𝑀] 𝑑_ � _, 𝑑_[�] . 3: Divide **Q** into _𝑇𝑟_ = � _𝐵𝑁𝑟_ � blocks **Q** 1 _, . . . ,_ **Q** _𝑇𝑟_ of size _𝐵𝑟_ × _𝑑_ each, and divide **K** _,_ **V** in to _𝑇𝑐_ = � _𝐵𝑁𝑐_ � blocks **K** 1 _, . . . ,_ **K** _𝑇𝑐_ and **V** 1 _, . . . ,_ **V** _𝑇𝑐_ , of size _𝐵𝑐_ × _𝑑_ each. 4: Divide **O** into _𝑇𝑟_ blocks **O** _𝑖, . . . ,_ **O** _𝑇𝑟_ of size _𝐵𝑟_ × _𝑑_ each, divide **dO** into _𝑇𝑟_ blocks **dO** _𝑖, . . . ,_ **dO** _𝑇𝑟_ of size _𝐵𝑟_ × _𝑑_ each, divide _ℓ_ into _𝑇𝑟_ blocks _ℓ𝑖, . . . , ℓ𝑇𝑟_ of size _𝐵𝑟_ each, divide _𝑚_ into _𝑇𝑟_ blocks _𝑚_ 1 _, . . . , 𝑚𝑇𝑟_ of size _𝐵𝑟_ each. 5: Initialize **dQ** = (0) _𝑁_ × _𝑑_ in HBM and divide it into _𝑇𝑟_ blocks **dQ** 1 _, . . . ,_ **dQ** _𝑇𝑟_ of size _𝐵𝑟_ × _𝑑_ each. Initialize **dK** = (0) _𝑁_ × _𝑑,_ **dV** = (0) _𝑁_ × _𝑑_ in HBM and divide **dK** _,_ **dV** in to _𝑇𝑐_ blocks **dK** 1 _, . . . ,_ **dK** _𝑇𝑐_ and **dV** 1 _, . . . ,_ **dV** _𝑇𝑐_ , of size _𝐵𝑐_ × _𝑑_ each. 6: **for** 1 ≤ _𝑗_ ≤ _𝑇𝑐_ **do** 7: Load **K** _𝑗 ,_ **V** _𝑗_ from HBM to on-chip SRAM. 8: Initialize **dK**[˜] _𝑗_ = (0) _𝐵𝑐_ × _𝑑,_ **dV**[˜] _𝑗_ = (0) _𝐵𝑐_ × _𝑑_ on SRAM. 9: **for** 1 ≤ _𝑖_ ≤ _𝑇𝑟_ **do** 10: Load **Q** _𝑖,_ **O** _𝑖,_ **dO** _𝑖,_ **dQ** _𝑖, ℓ𝑖, 𝑚𝑖_ from HBM to on-chip SRAM. 11: On chip, compute **S** _𝑖𝑗_ = _𝜏_ **Q** _𝑖_ **K** _[𝑇] 𝑗_[∈][R] _[𝐵][𝑟]_[×] _[𝐵][𝑐]_[.] 12: On chip, compute **S**[masked] _𝑖𝑗_ = mask( **S** _𝑖𝑗_ ). 13: On chip, compute **P** _𝑖𝑗_ = diag( _𝑙𝑖_ )[−][1] exp( **S**[masked] _𝑖𝑗_ − _𝑚𝑖_ ) ∈ R _[𝐵][𝑟]_[×] _[𝐵][𝑐]_ . 1 14: On chip, compute dropout mask **Z** _𝑖𝑗_ ∈ R _[𝐵][𝑟]_[×] _[𝐵][𝑐]_ where each entry has value 1− _𝑝_ drop[with][probability] 1 − _𝑝_ drop and value 0 with probability _𝑝_ drop. 15: On chip, compute **P** ˜[dropped] _𝑖𝑗_ =˜ **P** _𝑖𝑗_ ◦ **Z** _𝑖𝑗_ (pointwise multiply). 16: On chip, compute **dV** _𝑗_ ← **dV** _𝑗_ + ( **P**[dropped] _𝑖𝑗_ )[⊤] **dO** _𝑖_ ∈ R _[𝐵][𝑐]_[×] _[𝑑]_ . 17: On chip, compute **dP**[dropped] _𝑖𝑗_ = **dO** _𝑖_ **V**[⊤] _𝑗_[∈][R] _[𝐵][𝑟]_[×] _[𝐵][𝑐]_[.] 18: On chip, compute **dP** _𝑖𝑗_ = **dP**[dropped] _𝑖𝑗_ ◦ **Z** _𝑖𝑗_ (pointwise multiply). 19: On chip, compute _𝐷𝑖_ = rowsum( **dO** _𝑖_ ◦ **O** _𝑖_ ) ∈ R _[𝐵][𝑟]_ . 20: On chip, compute **dS** _𝑖𝑗_ = **P** _𝑖𝑗_ ◦( **dP** _𝑖𝑗_ − _𝐷𝑖_ ) ∈ R _[𝐵][𝑟]_[×] _[𝐵][𝑐]_ . 21: Write **dQ** _𝑖_ ← **dQ** _𝑖_ + _𝜏_ **dS** _𝑖𝑗_ **K** _𝑗_ ∈ R _[𝐵][𝑟]_[×] _[𝑑]_ to HBM. 22: On chip, compute **dK**[˜] _𝑗_ ← **dK**[˜] _𝑗_ + _𝜏_ **dS**[⊤] _𝑖𝑗_ **[Q]** _[𝑖]_[∈][R] _[𝐵][𝑐]_[×] _[𝑑]_[.] 23: **end for** ˜ ˜ 24: Write **dK** _𝑗_ ← **dK** _𝑗 ,_ **dV** _𝑗_ ← **dV** _𝑗_ to HBM. 25: **end for** 

- 26: Return **dQ** _,_ **dK** _,_ **dV** . 

We see that similar to the forward pass, the backward pass performs _𝑂_ ( _𝑁_[2] ) FLOPs and only requires _𝑂_ ( _𝑁_ ) extra memory beyond inputs, output, output gradient, and input gradients. 

We analyze the IO-complexity of the backward pass, similar to the forward pass (Theorem 2). 

**Theorem 5.** _Let 𝑁 be the sequence length, 𝑑 be the head dimension, and 𝑀 be size of SRAM with 𝑑_ ≤ _𝑀_ ≤ _𝑁𝑑. Standard attention (Algorithm 0) backward pass requires_ Θ( _𝑁𝑑_ + _𝑁_[2] ) _HBM accesses, while_ FlashAttention _backward pass (Algorithm 4) requires_ Θ( _𝑁_[2] _𝑑_[2] _𝑀_[−][1] ) _HBM accesses._ 

The proof is in Appendix C. 

21 

## **B.5 Comparison with Rabe and Staats [66]** 

We describe here some similarities and differences between our FlashAttention algorithm and the algorithm of Rabe and Staats [66]. 

Conceptually, both FlashAttention and Rabe and Staats [66] operate on blocks of the attention matrix using the well-established technique of tiling (or softmax scaling) [51, 60]. To reduce the memory footprint, both methods avoid storing the large attention matrix in the forward pass and recompute it in the backward pass. 

The first major difference is that Rabe and Staats [66] focuses on the reducing the total memory footprint (maximum amount of GPU memory required) while FlashAttention focuses on reducing memory accesses (the number of memory reads/writes). As mentioned in Section 2, the amount of memory access is the primary determining factor of runtime. Reducing memory accesses also necessarily reduces the total amount of memory required (e.g., if an operation incurs _𝐴_ memory accesses, then its total memory requirement is at most _𝐴_ ). As a result, FlashAttention is faster than standard attention (2-4×) while Rabe and Staats [66] is around the same speed or slightly slower than standard attention. In terms of total memory required, both methods offer substantial memory saving. 

The second difference between the two methods is the way information is summarized from each block to pass to the next block. Rabe and Staats [66] summarizes each block with its temporary output along with the softmax normalization statistics. At the end of the forward pass, the temporary outputs of all the blocks are combined using the statistics to produce the final output. FlashAttention instead incrementally updates the output (Algorithm 1 line 12) after processing each block, so only one copy of the output is needed (instead of _𝐾_ copies for _𝐾_ blocks). This means that FlashAttention has smaller total memory requirement compared to Rabe and Staats [66]. 

The final major difference is the way the backward pass is computed. Rabe and Staats [66] uses gradient checkpointing to recompute the attention matrix and the temporary output of each block. FlashAttention instead simplifies the backward pass analytically (Appendices B.2 and B.4). It only recomputes the attention matrix and does not recompute the temporary output of each block. This reduces the memory requirement for the backward pass and yields speedup. 

## **C Proofs** 

_Proof of Theorem 1._ We first count the number of FLOPs and extra memory required. 

The dominating FLOPs are from matrix multiplication. In the inner loop, (Algorithm 1 line 9), we compute **Q** _𝑖_ **K**[⊤] _𝑗_[∈][R] _[𝐵][𝑟]_[×] _[𝐵][𝑐]_[for] **[Q]** _[𝑖]_[∈][R] _[𝐵][𝑟]_[×] _[𝑑]_[and] **[K]** _[ 𝑗]_[∈][R] _[𝐵][𝑐]_[×] _[𝑑]_[,][which][takes] _[𝑂]_[(] _[𝐵][𝑟][𝐵][𝑐][𝑑]_[)][FLOPs.][We][also][compute] (Algorithm 1 line 12) **P**[˜] _𝑖𝑗_ **V** _𝑗_ ∈ R _[𝐵][𝑟]_[×] _[𝑑]_ for **P**[˜] _𝑖𝑗_ ∈ R _[𝐵][𝑟]_[×] _[𝐵][𝑐]_ and **V** _𝑗_ ∈ R _[𝐵][𝑐]_[×] _[𝑑]_ , which takes _𝑂_ ( _𝐵𝑟 𝐵𝑐 𝑑_ ) FLOPs. We execute the inner loops _𝑇𝑐𝑇𝑟_ = � _𝐵𝑁𝑐_ �� _𝐵𝑁𝑟_ � times. Therefore the total number of FLOPs is 

**==> picture [120 x 26] intentionally omitted <==**

In terms of extra memory required, we see that we need _𝑂_ ( _𝑁_ ) memory to store the statistics ( _ℓ, 𝑚_ ). 

We now prove the algorithm’s correctness by induction on _𝑗_ for 0 ≤ _𝑗_ ≤ _𝑇𝑐_ . Let **K** : _𝑗_ ∈ R _[𝑗𝐵][𝑐]_[×] _[𝑑]_ be the first _𝑗𝐵𝑐_ rows of **K** , and similarly **V** : _𝑗_ ∈ R _[𝑗𝐵][𝑐]_[×] _[𝑑]_ the the first _𝑗𝐵𝑐_ rows of **V** . Let **S** : _,_ : _𝑗_ = **QK**[⊤] : _𝑗_[∈][R] _[𝑁]_[×] _[𝑗𝐵][𝑐]_[,][and] **P** : _,_ : _𝑗_ = softmax( **S** : _,_ : _𝑗_ ) ∈ R _[𝑁]_[×] _[𝑗𝐵][𝑐]_ (softmax applied row-wise). Let _𝑚[𝑗] , ℓ_[(] _[ 𝑗]_[)] _,_ **O**[(] _[ 𝑗]_[)] be the values of _𝑚, ℓ,_ **O** in HBM after the _𝑗_ -th iteration of the outer loop (Algorithm 1 line 5). (Note that these values of _𝑚, ℓ,_ **O** are updated after each iteration of the outer loop.) We want to show that after the _𝑗_ -th iteration of the outer loop, we have computed in HBM: 

**==> picture [406 x 14] intentionally omitted <==**

Based on our initialization (Algorithm 1 line 2), this claim is true for _𝑗_ = 0 (i.e., before the any iteration of the outer loop is executed). Suppose that the claim holds for some _𝑗_ = 0 _, . . . ,𝑇𝑐_ − 1. We want to show that the claim also holds for _𝑗_ + 1. Indeed, when we update the statistics in the inner loop (Algorithm 1 line 10) 

22 

on the ( _𝑗_ + 1)-th iteration of the outer loop, we update _𝑚_[(] _[ 𝑗]_[+][1][)] = max( _𝑚_[(] _[ 𝑗]_[)] _, 𝑚_ ˜) where _𝑚_ ˜ ∈ R _[𝑁]_ is the row-max of **S** : _, 𝑗_ : _𝑗_ +1, the slice of **S** from column _𝑗𝐵𝑐_ to column ( _𝑗_ + 1) _𝐵𝑐_ − 1. This implies that 

**==> picture [134 x 13] intentionally omitted <==**

Similarly, we update 

**==> picture [152 x 13] intentionally omitted <==**

where _ℓ_[˜] = rowsum(exp( **S** : _, 𝑗_ : _𝑗_ +1 − _𝑚_ ˜)) ∈ R _[𝑁]_ . By the same algebraic manipulation in Section 3.1, we obtain: 

**==> picture [191 x 13] intentionally omitted <==**

Let **V** _𝑗_ : _𝑗_ +1 be the slice of **V** from column _𝑗𝐵𝑐_ to column ( _𝑗_ + 1) _𝐵𝑐_ − 1, we also update: 

**==> picture [452 x 173] intentionally omitted <==**

We then see that the claim is also true for _𝑗_ + 1. By induction, the claim is true for all _𝑗_ = 0 _, . . . ,𝑇𝑐_ . When _𝑗_ = _𝑇𝑐_ , we conclude that the final value of **O** in HBM is softmax( **S** ) **V** = softmax( **QK**[⊤] ) **V** . 

_Proof of Theorem 2._ We first analyze the IO complexity of standard attention implementation. The inputs **Q** _,_ **K** _,_ **V** ∈ R _[𝑁]_[×] _[𝑑]_ reside in HBM, and the at the end of the algorithm the output **O** ∈ R _[𝑁]_[×] _[𝑑]_ is written to HBM. In the first step of computing the matrix multiply **S** = **QK**[⊤] , the inputs **Q** _,_ **K** are read from HBM and the output **S** ∈ R _[𝑁]_[×] _[𝑁]_ is written to HBM (Algorithm 0 line 1). This incurs Θ( _𝑁𝑑_ + _𝑁_[2] ) HBM accesses. 

In the second step of computing **P** = softmax( **S** ), the input **S** is read from HBM and the output **P** is written to HBM (Algorithm 0 line 2). This incurs Θ( _𝑁_[2] ) HBM accesses. 

In the last step of computing **O** = **PV** , the inputs **P** _,_ **V** are read from global memory and the output **O** is written to HBM (Algorithm 0 line 3). This incurs Θ( _𝑁𝑑_ + _𝑁_[2] ) HBM accesses. 

Overall, standard attention implementation requires Θ( _𝑁𝑑_ + _𝑁_[2] ) global memory accesses. We now analyze the IO complexity of streaming attention. 

Following Algorithm 1, we see that each element of **K** and **V** is loaded from HBM once (Algorithm 1 line 6). We make _𝑇𝑐_ passes over **Q** and **O** , each pass loading all of **Q** and all of **O** to HBM (Algorithm 1 line 8). Therefore the number of HBM accesses is Θ ( _𝑁𝑑_ + _𝑁𝑑𝑇𝑐_ ) = Θ( _𝑁𝑑𝑇𝑐_ ). 

We derive the conditions on the block sizes _𝐵𝑐_ and _𝐵𝑟_ . We need the blocks **K** _𝑗_ and **V** _𝑗_ of size _𝐵𝑐_ × _𝑑_ to fit into on-chip memory, which translates to: 

**==> picture [128 x 25] intentionally omitted <==**

Similarly, we need the blocks **Q** _𝑖,_ **O** _𝑖_ of size _𝐵𝑟_ × _𝑑_ to fit into on-chip memory, which translates to: 

**==> picture [128 x 25] intentionally omitted <==**

Finally, we need the block **S** _𝑖𝑗_ of size _𝐵𝑟_ × _𝐵𝑐_ to fit into on-chip memory, which translates to: 

**==> picture [62 x 10] intentionally omitted <==**

23 

We therefore set: 

We then have: 

**==> picture [261 x 65] intentionally omitted <==**

As a result, the number of HBM accesses is: 

**==> picture [100 x 26] intentionally omitted <==**

□ 

_Proof of Proposition 3._ For contradiction, suppose that there exists an algorithm that computes exact attention where the number for HBM access for all _𝑀_ ∈[ _𝑑, 𝑁𝑑_ ] is 

**==> picture [46 x 26] intentionally omitted <==**

In the regime of _𝑀_ = Θ( _𝑁𝑑_ ), this results in the number of HBM accesses: 

**==> picture [84 x 26] intentionally omitted <==**

However, the input to attention (matrices **Q** _,_ **K** _,_ **V** ) and the output **O** have size _𝑁𝑑_ and they start out being in HBM, so if the algorithm computes exact attention it must incur at least Ω( _𝑁𝑑_ ) HBM accesses. This is a contradiction. □ 

_Proof of Theorem 5._ The IO complexity of the attention backward is very similar to the IO complexity of the attention forward (Theorem 2). Here we provide a sketch of the proof. 

We first analyze the IO complexity of standard attention backward pass. The inputs **Q** _,_ **K** _,_ **V** _,_ **dO** ∈ R _[𝑁]_[×] _[𝑑]_ reside in HBM, and the at the end of the algorithm the outputs **dQ** _,_ **dK** _,_ **dV** ∈ R _[𝑁]_[×] _[𝑑]_ are written to HBM. 

At each step of the standard attention backward pass, one needs to load inputs of size _𝑁𝑑_ or _𝑁_[2] from HBM, and needs to write the outputs of size _𝑁_[2] or _𝑁𝑑_ to HBM. This incurs Θ( _𝑁𝑑_ + _𝑁_[2] ) HBM accesses. 

We now analyze the IO complexity of FlashAttention backward pass. 

Similar to Theorem 2, we see that each element of **K** and **V** is loaded from HBM once. Each element of **dK** and **dV** is only written to HBM once. We make _𝑇𝑐_ passes over **Q** _,_ **O** _,_ **dO** , each pass loading all of **Q** _,_ **O** _,_ **dO** to HBM. We also make _𝑇𝑐_ passes over **dQ** , each pass reading/writing all of **dQ** from/to HBM. Therefore the number of HBM accesses is Θ ( _𝑁𝑑_ + _𝑁𝑑𝑇𝑐_ ) = Θ( _𝑁𝑑𝑇𝑐_ ). 

As in the proof of Theorem 2, the constraints on the block sizes are that: 

We then have: 

**==> picture [174 x 68] intentionally omitted <==**

As a result, the number of HBM accesses is: 

**==> picture [285 x 42] intentionally omitted <==**

24 

**Algorithm 5** Block-Sparse FlashAttention Forward Pass 

- **Require:** Matrices **Q** _,_ **K** _,_ **V** ∈ R _[𝑁]_[×] _[𝑑]_ in HBM, on-chip SRAM of size _𝑀_ , softmax scaling constant _𝜏_ ∈ R, masking function mask, dropout probability _𝑝_ drop, block sizes _𝐵𝑐_ = � 4 _𝑀𝑑_ � _, 𝐵𝑟_ = min[��] 4 _[𝑀] 𝑑_ � _, 𝑑_[�] , block sparsity mask _𝑀_ ∈{0 _,_ 1} _[𝑁]_[/] _[𝐵][𝑟]_[×] _[𝑁]_[/] _[𝐵][𝑐]_ .. 

- 1: Initialize the pseudo-random number generator state R and save to HBM. 

- 2: Initialize **O** = (0) _𝑁_ × _𝑑_ ∈ R _[𝑁]_[×] _[𝑑] , ℓ_ = (0) _𝑁_ ∈ R _[𝑁] , 𝑚_ = (−∞) _𝑁_ ∈ R _[𝑁]_ in HBM. 3: Divide **Q** into _𝑇𝑟_ = � _𝐵𝑁𝑟_ � blocks **Q** 1 _, . . . ,_ **Q** _𝑇𝑟_ of size _𝐵𝑟_ × _𝑑_ each, and divide **K** _,_ **V** in to _𝑇𝑐_ = � _𝐵𝑁𝑐_ � blocks **K** 1 _, . . . ,_ **K** _𝑇𝑐_ and **V** 1 _, . . . ,_ **V** _𝑇𝑐_ , of size _𝐵𝑐_ × _𝑑_ each. 

- 4: Divide **O** into _𝑇𝑟_ blocks **O** _𝑖, . . . ,_ **O** _𝑇𝑟_ of size _𝐵𝑟_ × _𝑑_ each, divide _ℓ_ into _𝑇𝑟_ blocks _ℓ𝑖, . . . , ℓ𝑇𝑟_ of size _𝐵𝑟_ each, divide _𝑚_ into _𝑇𝑟_ blocks _𝑚_ 1 _, . . . , 𝑚𝑇𝑟_ of size _𝐵𝑟_ each. 

- 5: **for** 1 ≤ _𝑗_ ≤ _𝑇𝑐_ **do** 

- 6: Load **K** _𝑗 ,_ **V** _𝑗_ from HBM to on-chip SRAM. 

- 7: **for** 1 ≤ _𝑖_ ≤ _𝑇𝑟_ **do** 8: **if** _𝑀𝑖𝑗_ ≠ 0 **then** 

- 9: Load **Q** _𝑖,_ **O** _𝑖, ℓ𝑖, 𝑚𝑖_ from HBM to on-chip SRAM. 

- 10: On chip, compute **S** _𝑖𝑗_ = _𝜏_ **Q** _𝑖_ **K** _[𝑇] 𝑗_[∈][R] _[𝐵][𝑟]_[×] _[𝐵][𝑐]_[.] 11: On chip, compute **S**[masked] _𝑖𝑗_ = mask( **S** _𝑖𝑗_ ). ˜ ˜ 

- 12: On˜ chip, compute˜ _𝑚𝑖𝑗_ = rowmax( **S**[masked] _𝑖𝑗_ ) ∈ R _[𝐵][𝑟]_ , **P**[˜] _𝑖𝑗_ = exp( **S**[masked] _𝑖𝑗_ − _𝑚𝑖𝑗_ ) ∈ R _[𝐵][𝑟]_[×] _[𝐵][𝑐]_ (pointwise), 13: _ℓ_ On _𝑖𝑗_ =chip, rowsumcompute( **P** _𝑖𝑗_ ) ∈ _𝑚_ R _𝑖_[new] _[𝐵] 𝑟_ .= max( _𝑚𝑖, 𝑚_ ˜ _𝑖𝑗_ ) ∈ R _[𝐵][𝑟]_ , _ℓ𝑖_[new] = _𝑒[𝑚][𝑖]_[−] _[𝑚] 𝑖_[new] _ℓ𝑖_ + _𝑒[𝑚]_[˜] _[𝑖𝑗]_[−] _[𝑚] 𝑖_[new] _ℓ_ ˜ _𝑖𝑗_ ∈ R _[𝐵] 𝑟_ . 14: On chip, compute **P**[˜][dropped] _𝑖𝑗_ = dropout( **P**[˜] _𝑖𝑗 , 𝑝_ drop). ˜ 15: Write **O** _𝑖_ ← diag( _ℓ𝑖_[new] )[−][1] (diag( _ℓ𝑖_ ) _𝑒[𝑚][𝑖]_[−] _[𝑚] 𝑖_[new] **O** _𝑖_ + _𝑒[𝑚]_[˜] _[𝑖𝑗]_[−] _[𝑚] 𝑖_[new] **P**[dropped] _𝑖𝑗_ **V** _𝑗_ ) to HBM. 16: Write _ℓ𝑖_ ← _ℓ𝑖_[new] , _𝑚𝑖_ ← _𝑚𝑖_[new] to HBM. 17: **end if** 

- 18: **end for** 

- 19: **end for** 

- 20: Return **O** _, ℓ, 𝑚,_ R. 

## **D Extension Details** 

## **D.1 Block-sparse FlashAttention** 

We describe the full block-sparse FlashAttention algorithm in Algorithm 5. The algorithm is identical to Algorithm 2, except that we skip zero blocks. 

We prove the IO-complexity of block-sparse FlashAttention. 

_Proof of Proposition 4._ The proof is very similar to the proof of Theorem 2. For the block-sparse case, notice that we only need to load blocks corresponding to nonzero blocks. As a result, the number of HBM accesses are scaled by _𝑠_ , the fraction of nonzero blocks in the block-sparsity mask. However, for small values of _𝑠_ , we would still need to write the result **O** ∈ R _[𝑁]_[×] _[𝑑]_ . Therefore the number of HBM accesses is 

**==> picture [76 x 25] intentionally omitted <==**

**==> picture [8 x 6] intentionally omitted <==**

## **D.2 Potential Extensions** 

We discuss here a few potential extensions of the IO-aware approach to speed up deep learning training. **Multi-GPU Attention.** Large language models are trained on hundreds or thousands of GPUs, and one typically splits the attention computation between 4-8 GPUs on the same node [77]. This introduces another level of memory hierarchy: beside GPU SRAM and GPU HBM, we also have the HBM of other 

25 

GPUs. For very long sequences, the different GPUs on the same node can cooperate to compute attention by taking into account the asymmetry of different levels of memory hierarchy. 

**Sparse MLP layers.** Typical dense MLP layers are compute-bound and not memory-bound. To improve their efficiency, MLP layers with sparse weight matrices can be used [17]. However, many sparse MLP layers are instead memory-bound, and their speedup is often not proportional to the sparsity. We believe that an IO-aware implementation can alleviate this issue and realize the benefits of sparsity. We are excited about future work in this direction, to reduce the computational requirement of large models and improve their wall-block runtime. 

**Kernel machine learning.** Our approach in FlashAttention relies on the fact that the _𝑁_ × _𝑁_ attention matrix is a function of a low-rank matrix **QK**[⊤] (of rank _𝑑_ ≪ _𝑁_ ). As a result, we can repeatedly load the inputs **Q** _,_ **K** and recompute the block of the attention matrix that we need, significantly reducing HBM access. As similar scenario happens in kernel machine learning: each element _𝐾𝑖𝑗_ of the _𝑁_ × _𝑁_ kernel matrix **K** is a function of two vectors of size _𝑑_ ≪ _𝑁_ , as it measures the similarity between two datapoints _𝑥𝑖_ and _𝑥 𝑗_ . The KeOps library [8, 26] is a successful example of how reducing memory reads/writes can speed up kernel operations. We hope that this will motivate kernel methods that focus more on reducing IOs instead of just FLOPs. 

## **E Full Experimental Results** 

## **E.1 BERT** 

We train BERT-large following the training procedure and hyperparameters of the reference MLPerf 1.1 implementation. In particular, we use the LAMB optimizer with learning rate 3.75e-3, with batch size 448, trained for at most 7100 steps. The training is stopped once the validation accuracy (for masked language modeling) reaches the target 72.0%, and the wall-clock run-time is measured. We train with FP16 precision using Apex AMP (with O2 optimization level). 

We compare our results with the reported training speed from Nvidia that was submitted to MLPerf 1.1 (Table 1). 

We use the same train / validation data split provided by MLPerf 1.1 reference implementation. In particular, we evaluate on the same 10000 validation examples as the baseline from Nvidia. 

We train the model on 8×A100-80GB GPUs. Each training run takes between 16 and 19 minutes, and we average the results of 10 runs. 

## **E.2 GPT-2** 

We use the standard implementations of GPT-2 [67] from Huggingface `transformers` library and from Nvidia’s Megatron-LM repo. We follow the training recipe of the Megatron-LM repo. 

We use an effective batch size of 512, and use gradient accumulation to fit into available GPU memory. We use the AdamW optimizer, with learning rate 6e-4 for GPT-2 small and 1.5e-4 for GPT-2 medium, and weight decay of 0.1. All models are trained with the same hyperparameters for 400K steps. We run all implementations with mixed-precision training (PyTorch AMP). 

We use the Openwebtext dataset, with the GPT-2 BPE tokenizer. We randomly select 0.5% of the dataset as the validation set, with the rest being used as training set. This random selection of validation set is done once, and all models are evaluated on the same validation set. 

We train the model on 8×A100-40GB GPUs, and we measure the wall-clock training time. Training GPT-2 small takes between 2.7-9.5 days, and training GPT-2 medium takes between 6.9-21.0 days (Table 2). 

In Fig. 4, we plot of the validation perplexity throughout training of GPT-2 small/medium, using either HuggingFace implementation or our FlashAttention implementation. We see that FlashAttention behaves the same as the baseline implementation and the validation perplexity curves of the two implementations almost lie on top of each other. 

**Long Document Classification.** For MIMIC-III and ECtHR, we follow the hyperparameters of Dai et al. [13]. 

26 

**==> picture [328 x 257] intentionally omitted <==**

**----- Start of picture text -----**<br>
30<br>GPT-2-small HuggingFace<br>GPT-2-small FlashAttention<br>25 GPT-2-medium HuggingFace<br>GPT-2-medium FlashAttention<br>20<br>15<br>10<br>100k 200k 300k<br>Training steps<br>Val perplexity<br>**----- End of picture text -----**<br>


Figure 4: Validation perplexity of GPT-2 small/medium using two implementations. We confirm that FlashAttention yields the same validation curves as the baseline implementation from HuggingFace. 

## **E.3 LRA details** 

We follow the hyperparameters from the Long-range arena paper [80], the Long-range arena repo ( `https: //github.com/google-research/long-range-arena` ), and the Nyströmformer reproduction [90]. To be generous to the baseline methods, if we are unable to reproduce the performance of any baseline for any of the five tasks, we report the better performance from Tay et al. [80] or Xiong et al. [90] for that baseline on that task. 

After hyperparameter tuning, almost all of the attention methods achieve similar accuracy on all of the five LRA tasks. 

We run all methods with mixed-precision training, except for Performer (not stable with mixed precision) and Local Attention (implementation does not support FP16). 

To calculate the overall wallclock-time speedup, we take the geometric mean of the wallclock-time speedup of each of the five tasks. 

**Path-X** For Path-X and Path-256, we follow the hyperparameters from the PathFinder-32 experiments from the long-range arena paper[80]. For both, we first pretrain a model on Path-64. We take the checkpoint after 200 epochs, upsample its positional embedding (we duplicate the positional embeddings gridwise in space), and fine-tune it on the downstream task for 200 epochs with one epoch of linear warmup, and cosine decay of the learning rate. For Path-X, we take the best performing checkpoint (according to val accuracy), and additionally fine-tune it for 200 epochs with the same warmup and learning rate (this adds roughly 4 points of accuracy to FlashAttention for Path-X, but the model starts overfitting afterwards). 

## **E.4 Comparison with Apex FMHA** 

We compare our method/implementation with Apex FMHA ( `https://github.com/NVIDIA/apex/tree/ master/apex/contrib/csrc/fmha` ). 

When we started this project, Apex FMHA was the fastest implementation of attention (that we knew of), tailored for short sequences of length at most 512. In fact, almost all MLPerf submissions for BERT training benchmark running on Nvidia GPUs use FMHA for their model code, as of MLPerf 1.1 [58]. Since 

27 

Table 7: Runtime (ms) of FlashAttention compared to FMHA by sequence length, with masking and dropout, measured on an A100-SXM4-40GB GPU. Batch size 64, 16 heads, head dimension 64 (i.e., BERT-large size). 

|**Attention Method**|128<br>256<br>512|
|---|---|
|**Apex FMHA forward**<br>**FlashAttention forward**|0.10<br>0.29<br>1.14<br>**0.08**<br>**0.22**<br>**0.81**|
|**Apex FMHA backward**<br>**FlashAttention backward**|**0.17**<br>**0.52**<br>**1.81**<br>0.20<br>0.53<br>2.00|
|**Apex FMHA forward + backward**<br>**FlashAttention forward + backward**|**0.27**<br>0.81<br>2.95<br>0.28<br>**0.75**<br>**2.81**|



FMHA targets BERT models, it only supports head dimension 64, and only runs on A100 GPUs. FMHA fuses the attention computation dropout(softmax(mask( **QK**[⊤] ))) **V** into one CUDA kernel. In the forward pass, it stores the attention matrix softmax(mask( **QK** _[𝑇]_ )) to HBM to be used in gradient computation. As a result, it does not offer substantial memory saving (though for shorter sequences memory footprint is often not a primary concern). 

We use FMHA code as a starting point, and apply two well-established techniques (tiling and recomputation) to deal with long sequences and to save memory as mentioned in Section 3. As a result, we can support much longer sequences (e.g., up to length 64K). We also support more head dimensions (16, 32, 64, 128) and broader GPU types (all Turing and Ampere GPUs at the time of writing). 

In Table 7, we compare the performance of FlashAttention and Apex FMHA for short sequences (as FMHA only supports sequence length at most 512). Generally FlashAttention is slightly faster than FMHA in the forward pass and slightly slower than FMHA in the backward pass. This is because we do not store the attention matrix in the forward pass and recompute it in the backward pass. Compared to FMHA, the overall runtime of FlashAttention is about 4% slower for sequence length 128, 8% faster for sequence length 256, and 5% faster for sequence length 512. 

## **E.5 Speedup On Different Hardware and Configurations** 

Speedup varies between different types of GPU types and generations depending on HBM bandwidth and SRAM size. In this section, we profile FlashAttention speedup on different GPUs and configurations. 

**==> picture [397 x 184] intentionally omitted <==**

Figure 5: Speedup over standard PyTorch attention at different sequence lengths, on A100. 

**A100** Figure 5 shows speedup on an A100 GPU with batch size 8, head dimension 64, and 12 attention heads, across different sequence lengths. We generally see 2-4× speedup, and we see more speedup when using dropout and masking due to kernel fusion. 

28 

**==> picture [397 x 175] intentionally omitted <==**

Figure 6: Speedup over standard PyTorch attention at different sequence lengths, on A100, with head dimension 128. 

**A100, Head Dimension 128** Speedup also changes when we increase the head dimension. Each block requires more memory, so we need to use smaller block sizes to fit into SRAM. Figure 6 shows speedup with head dimension 128 on an A100 (batch size 16, 12 heads). We see less speedup overall—but we can still see significant speedup (up to 3×) with a causal mask, where half the blocks are masked out. 

**==> picture [397 x 184] intentionally omitted <==**

Figure 7: Speedup over standard PyTorch attention at different sequence lengths, on RTX 3090. 

**RTX 3090** Figure 7 shows speedup on an RTX 3090 GPU. Here, we use batch size 12 with 12 attention heads. We observe slightly higher speedups on the RTX 3090 (between 2.5-4.5×), since the memory bandwidth on an RTX 3090 is lower than on an A100 (roughly 900 GB/s vs. 1.5 TB/s). 

**T4** Figure 8 shows speedup on a T4 GPU. T4 SRAM is smaller than A100, so we need to make the block sizes smaller in FlashAttention. As a result, we observe less speedup on T4, which matches the IO complexity analysis in Section 3.2. T4 GPUs are commonly used for inference, so we also report speedup on the forward pass only. 

29 

**==> picture [396 x 181] intentionally omitted <==**

**==> picture [397 x 184] intentionally omitted <==**

Figure 8: Speedup over standard PyTorch attention at different sequence lengths, on T4. **Top:** Combined forward pass + backward pass. **Bottom:** Forward pass only. 

## **E.6 Full Benchmarking Results** 

We report the full benchmarking results and experimental details on A100. 

**Baselines** We compare against reference implementations for exact attention from PyTorch/HuggingFace and Megatron, approximate attention, and sparse attention. For approximate attention, we compare against reference implementations of Reformer [51], Local Attention [68], Linformer Attention [84], Smyrf [19], and LongShortFormer (LSFormer) [94]. For sparse attention, we compare against reference implementations of Block-Sparse Attention form OpenAI [11], Longformer[3], and BigBird Attention [92]. For the approximate and sparse attention, we use a compression ratio of 1/8, or a compressed sequence length of 256, whichever is smaller. 

**Setup** We measure runtime and memory usage of the attention computation with 8 heads of dimension 64, and batch size 16 on a machine with one A100 GPU with 40 GB of GPU HBM. We vary sequence length in our experiments. We compute attention on random vectors for **Q** , **K** , and **V** (we do not measure the projection from the hidden layer). For dropout, we use dropout 0.1; for masking, we use a padding mask with uniformly-random mask lengths between the total sequence length and the total sequence length minus 20. To measure runtime, we take the average of 100 measurements of the attention call. We only measure memory footprint once, since it does not vary between runs. 

30 

Table 8: Pointers to results tables. 

|**Dropout**<br>**Masking**<br>**Pass**|**Table**|
|---|---|
|Yes<br>Yes<br>Forward<br>Yes<br>Yes<br>Backward<br>Yes<br>Yes<br>Combined<br>No<br>Yes<br>Forward<br>No<br>Yes<br>Backward<br>No<br>Yes<br>Combined<br>Yes<br>No<br>Forward<br>Yes<br>No<br>Backward<br>Yes<br>No<br>Combined<br>No<br>No<br>Forward<br>No<br>No<br>Backward<br>No<br>No<br>Combined<br>No<br>No<br>Memory Usage (Combined)|Table 9<br>Table 10<br>Table 11<br>Table 12<br>Table 13<br>Table 14<br>Table 15<br>Table 16<br>Table 17<br>Table 18<br>Table 19<br>Table 20<br>Table 21|



Table 9: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, **with dropout and masking** . Best in **bold** , second best underlined. 

|**Attention Method**|128<br>256<br>512<br>1024<br>2048<br>4096<br>8192<br>16384<br>32768<br>65536|
|---|---|
|**PyTorch Attention**<br>**Megatron**|0.36<br>0.34<br>0.78<br>2.54<br>9.33<br>36.33<br>-<br>-<br>-<br>-<br>0.40<br>0.40<br>1.10<br>3.65<br>16.19<br>-<br>-<br>-<br>-<br>-|
|**Reformer**<br>**Local Attention**<br>**Linformer**<br>**Smyrf**<br>**LSformer**|2.03<br>3.15<br>5.67<br>11.02<br>22.59<br>46.14<br>97.38<br>212.13<br>-<br>-<br>0.83<br>0.86<br>1.01<br>2.20<br>7.13<br>14.32<br>28.60<br>57.79<br>117.67<br>-<br>0.67<br>0.52<br>0.69<br>0.71<br>1.65<br>3.18<br>6.15<br>12.16<br>24.17<br>52.39<br>2.27<br>2.34<br>3.91<br>7.44<br>14.71<br>29.22<br>58.27<br>116.41<br>-<br>-<br>1.18<br>1.27<br>1.34<br>3.38<br>11.40<br>22.55<br>44.95<br>89.76<br>179.66<br>-|
|**Block Sparse**<br>**Longformer**<br>**BigBird**|1.12<br>1.11<br>2.13<br>2.77<br>6.95<br>20.91<br>-<br>-<br>-<br>-<br>1.22<br>1.14<br>1.08<br>1.95<br>5.72<br>12.98<br>-<br>-<br>-<br>-<br>1.13<br>1.12<br>1.12<br>1.77<br>6.03<br>13.68<br>-<br>-<br>-<br>-|
|**FlashAttention**<br>**Block-Sparse FlashAttention**|**0.04**<br>0.06<br>0.21<br>0.82<br>2.85<br>10.41<br>41.74<br>167.19<br>670.76<br>2682.35<br>0.06<br>**0.06**<br>**0.06**<br>**0.12**<br>**0.44**<br>**0.86**<br>**1.70**<br>**3.29**<br>**6.55**<br>**13.34**|



We report timing results on the forward pass, backward pass, and combined forward + backward pass. We measure each method with and without dropout, masking, or both—except for Block Sparse, Longformer, and BigBird. These methods did not successfully run the backward pass with masking due to a bug in external libraries, so we measured them without masking to be generous. We use FP16 for all measurements, except for Local Attention, whose implementation only supports FP32. 

For each baseline, we increase sequence length until it runs out of memory on the GPU, except for the following exceptions: The Megatron implementation does not support sequence lengths longer than 2048. Block-Sparse (OpenAI) does not support sequence lengths longer than 4096. Longformer and BigBird do not support sequence lengths longer than 8092. 

We measure memory usage on the combined forward + backward pass, without dropout or masking. 

**Results** Table 8 summarizes all the experimental configurations and contains pointers to the results tables. 

31 

Table 10: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, **with dropout and masking** . Best in **bold** , second best underlined. 

|**Attention Method**|128<br>256<br>512<br>1024<br>2048<br>4096<br>8192<br>16384<br>32768|65536|
|---|---|---|
|**PyTorch Attention**<br>**Megatron**|0.37<br>0.49<br>1.66<br>5.81<br>22.32<br>87.67<br>-<br>-<br>-<br>0.35<br>0.32<br>0.77<br>2.42<br>8.43<br>-<br>-<br>-<br>-|-<br>-|
|**Reformer**<br>**Local Attention**<br>**Linformer**<br>**Smyrf**<br>**LSformer**|2.37<br>4.59<br>8.91<br>17.68<br>35.13<br>70.05<br>140.01<br>-<br>-<br>0.55<br>0.62<br>1.49<br>4.03<br>13.78<br>27.61<br>55.20<br>110.27<br>221.40<br>0.89<br>0.80<br>0.81<br>0.93<br>2.48<br>4.75<br>9.29<br>18.27<br>36.53<br>1.41<br>2.83<br>5.43<br>10.72<br>21.25<br>42.31<br>84.48<br>168.95<br>-<br>1.75<br>1.76<br>3.01<br>7.50<br>20.07<br>39.08<br>76.39<br>150.82<br>-|-<br>-<br>-<br>-<br>-|
|**Block Sparse**<br>**Longformer**<br>**BigBird**|1.29<br>1.28<br>2.18<br>3.04<br>7.27<br>21.16<br>-<br>-<br>-<br>1.27<br>1.31<br>1.29<br>2.04<br>5.24<br>10.74<br>25.95<br>-<br>-<br>1.33<br>1.28<br>1.32<br>1.81<br>5.55<br>11.44<br>27.45<br>-<br>-|-<br>-<br>-|
|**FlashAttention**<br>**Block-Sparse FlashAttention**|**0.30**<br>**0.26**<br>0.68<br>2.02<br>6.84<br>26.89<br>105.70<br>418.96<br>1666.89<br>**0.30**<br>0.27<br>**0.29**<br>**0.59**<br>**1.50**<br>**2.94**<br>**5.82**<br>**11.85**<br>**23.98**|6660.44|
|||**47.61**|



Table 11: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, **with dropout and masking** . Best in **bold** , second best underlined. 

|**Attention Method**|128<br>256<br>512<br>1024<br>2048<br>4096<br>8192<br>16384<br>32768|65536|
|---|---|---|
|**PyTorch Attention**<br>**Megatron**|0.84<br>0.86<br>2.35<br>8.29<br>31.75<br>124.19<br>-<br>-<br>-<br>0.87<br>0.89<br>1.33<br>4.21<br>16.50<br>-<br>-<br>-<br>-|-<br>-|
|**Reformer**<br>**Local Attention**<br>**Linformer**<br>**Smyrf**<br>**LSformer**|4.30<br>7.76<br>14.60<br>28.74<br>57.79<br>116.34<br>237.57<br>-<br>-<br>1.40<br>1.60<br>2.06<br>6.06<br>20.94<br>42.01<br>84.08<br>168.48<br>339.45<br>1.57<br>1.49<br>1.55<br>1.60<br>4.19<br>8.04<br>15.71<br>30.92<br>61.47<br>3.41<br>5.08<br>9.35<br>18.18<br>36.03<br>71.68<br>143.04<br>285.87<br>-<br>3.08<br>3.10<br>4.26<br>10.90<br>31.59<br>61.72<br>121.51<br>241.18<br>-|-<br>-<br>-<br>-<br>-|
|**Block Sparse**<br>**Longformer**<br>**BigBird**|2.54<br>2.52<br>3.71<br>5.44<br>13.29<br>39.19<br>-<br>-<br>-<br>2.47<br>2.49<br>2.51<br>3.10<br>10.39<br>22.49<br>60.44<br>-<br>-<br>2.51<br>2.49<br>2.52<br>3.40<br>10.97<br>23.89<br>63.28<br>-<br>-|-<br>-<br>-|
|**FlashAttention**<br>**Block-Sparse FlashAttention**|**0.43**<br>**0.41**<br>0.95<br>2.55<br>9.56<br>37.49<br>147.75<br>586.61<br>2339.11<br>0.44<br>0.44<br>**0.45**<br>**0.89**<br>**1.95**<br>**4.12**<br>**7.64**<br>**16.60**<br>**32.73**|9341.3|
|||**64.11**|



Table 12: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, **with masking** . Best in **bold** , second best underlined. 

|**Attention Method**|128|256|512|1024|2048|4096|8192|16384|32768|65536|
|---|---|---|---|---|---|---|---|---|---|---|
|**PyTorch Attention**|0.30|0.30|0.63|1.93|7.08|27.45|112.90|-|-|-|
|**Megatron**|0.45|0.41|0.43|1.52|5.80|-|-|-|-|-|
|**Reformer**|1.87|3.00|5.37|10.43|21.40|43.83|92.80|203.24|-|-|
|**Local Attention**|0.70|0.81|1.02|2.09|6.64|13.34|26.77|54.02|110.11|-|
|**Linformer**|0.63|0.50|0.67|0.65|1.36|2.60|5.04|9.92|19.69|43.47|
|**Smyrf**|2.38|2.32|3.76|7.16|14.14|28.09|55.98|111.73|-|-|
|**LSformer**|1.22|1.29|1.44|3.28|10.99|21.72|43.29|86.32|172.76|-|
|**Block Sparse**|0.96|1.04|1.66|2.16|5.41|16.15|-|-|-|-|
|**Longformer**|0.99|0.98|0.99|1.56|4.79|11.07|32.98|-|-|-|
|**BigBird**|0.96|1.02|1.02|1.48|5.05|11.59|34.16|-|-|-|
|**FlashAttention**|**0.03**|**0.04**|0.17|0.68|2.28|8.40|33.55|134.14|537.50|2150.88|
|**Block-Sparse FlashAttention**|0.05|**0.04**|**0.05**|**0.11**|**0.35**|**0.68**|**1.33**|**2.54**|**5.34**|**10.73**|



Table 13: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, **with masking** . Best in **bold** , second best underlined. 

|**Attention Method**|128<br>256<br>512<br>1024<br>2048<br>4096<br>8192<br>16384<br>32768<br>65536|
|---|---|
|**PyTorch Attention**<br>**Megatron**|0.44<br>0.46<br>1.53<br>5.33<br>20.34<br>79.87<br>-<br>-<br>-<br>-<br>0.29<br>0.31<br>0.65<br>1.95<br>6.49<br>-<br>-<br>-<br>-<br>-|
|**Reformer**<br>**Local Attention**<br>**Linformer**<br>**Smyrf**<br>**LSformer**|2.31<br>4.47<br>8.68<br>17.20<br>34.14<br>68.09<br>136.02<br>-<br>-<br>-<br>0.51<br>0.62<br>1.30<br>3.81<br>13.33<br>26.72<br>53.41<br>106.82<br>214.15<br>-<br>0.76<br>0.81<br>0.94<br>0.87<br>2.24<br>4.25<br>8.35<br>16.38<br>32.67<br>72.11<br>1.34<br>2.77<br>5.30<br>10.46<br>20.73<br>41.27<br>82.41<br>164.86<br>-<br>-<br>1.66<br>1.61<br>3.09<br>7.42<br>19.68<br>38.35<br>74.92<br>147.86<br>-<br>-|
|**Block Sparse**<br>**Longformer**<br>**BigBird**|1.24<br>1.25<br>2.04<br>2.91<br>6.78<br>19.67<br>-<br>-<br>-<br>-<br>1.27<br>1.23<br>1.24<br>1.85<br>4.99<br>10.21<br>24.89<br>-<br>-<br>-<br>1.43<br>1.50<br>1.44<br>1.69<br>5.25<br>10.86<br>26.26<br>-<br>-<br>-|
|**FlashAttention**<br>**Block-Sparse FlashAttention**|**0.21**<br>**0.22**<br>0.62<br>1.84<br>5.77<br>22.25<br>86.21<br>338.91<br>1343.91<br>5361.09<br>0.22<br>0.22<br>**0.26**<br>**0.57**<br>**1.55**<br>**3.13**<br>**5.98**<br>**12.21**<br>**23.49**<br>**47.85**|



32 

Table 14: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, **with masking** . Best in **bold** , second best underlined. 

|**Attention Method**|128<br>256<br>512<br>1024<br>2048<br>4096<br>8192<br>16384<br>32768|65536|
|---|---|---|
|**PyTorch Attention**<br>**Megatron**|0.80<br>0.81<br>2.08<br>7.23<br>27.51<br>107.58<br>-<br>-<br>-<br>0.81<br>0.83<br>1.09<br>3.36<br>12.39<br>-<br>-<br>-<br>-|-<br>-|
|**Reformer**<br>**Local Attention**<br>**Linformer**<br>**Smyrf**<br>**LSformer**|4.16<br>7.46<br>14.06<br>27.68<br>55.66<br>112.15<br>229.37<br>-<br>-<br>1.39<br>1.68<br>2.08<br>5.83<br>20.04<br>40.16<br>80.44<br>161.35<br>325.11<br>1.51<br>1.42<br>1.56<br>1.67<br>3.67<br>6.99<br>13.63<br>26.77<br>53.36<br>3.38<br>4.93<br>9.07<br>17.66<br>34.94<br>69.55<br>138.72<br>277.41<br>-<br>3.08<br>3.10<br>4.26<br>10.90<br>31.59<br>61.72<br>121.51<br>241.18<br>-|-<br>-<br>117.56<br>-<br>-|
|**Block Sparse**<br>**Longformer**<br>**BigBird**|2.39<br>2.40<br>3.31<br>5.02<br>12.25<br>35.94<br>-<br>-<br>-<br>2.36<br>2.34<br>2.38<br>2.94<br>9.83<br>21.35<br>58.12<br>-<br>-<br>2.35<br>2.35<br>2.37<br>3.25<br>10.36<br>22.57<br>60.63<br>-<br>-|-<br>-<br>-|
|**FlashAttention**<br>**Block-Sparse FlashAttention**|**0.32**<br>**0.30**<br>0.83<br>2.37<br>7.95<br>30.77<br>119.98<br>473.65<br>1883.43<br>0.34<br>0.34<br>**0.36**<br>**0.69**<br>**1.85**<br>**3.89**<br>**7.16**<br>**14.85**<br>**30.46**|7513.01<br>**60.03**|



Table 15: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, **with dropout** . Best in **bold** , second best underlined. 

|**Attention Method**|128|256|512|1024|2048|4096|8192|16384|32768|65536|
|---|---|---|---|---|---|---|---|---|---|---|
|**PyTorch Attention**|0.26|0.24|0.57|1.80|6.56|25.34|-|-|-|-|
|**Megatron**|0.27|0.27|0.56|1.88|6.56|-|-|-|-|-|
|**Reformer**|1.83|2.96|5.31|10.33|21.19|43.42|91.96|201.34|-|-|
|**Local Attention**|0.51|0.60|0.78|2.01|6.23|12.52|25.07|50.50|102.18|-|
|**Linformer**|0.47|0.37|0.49|**0.52**|1.37|2.65|5.12|10.13|20.25|44.16|
|**Smyrf**|2.12|2.01|3.15|5.97|11.83|23.36|46.48|92.72|-|-|
|**LSformer**|1.28|1.33|1.51|3.39|11.40|22.54|44.96|89.85|179.73|-|
|**Block Sparse**|1.03|1.00|1.72|2.39|5.96|17.88|-|-|-|-|
|**Longformer**|1.02|1.03|1.03|1.73|5.10|11.63|34.22|-|-|-|
|**BigBird**|0.99|1.03|1.01|1.58|5.36|12.27|35.56|-|-|-|
|**FlashAttention**|**0.10**|**0.10**|**0.22**|0.83|2.81|10.38|41.63|167.01|668.74|2678.11|
|**Block-Sparse FlashAttention**|0.54|0.51|0.68|0.61|**0.67**|**1.10**|**1.89**|**3.71**|**7.18**|**14.41**|



Table 16: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, **with dropout** . Best in **bold** , second best underlined. 

|**Attention Method**|128<br>256<br>512<br>1024<br>2048<br>4096<br>8192<br>16384<br>32768|65536|
|---|---|---|
|**PyTorch Attention**<br>**Megatron**|0.44<br>0.35<br>0.90<br>2.94<br>10.77<br>41.67<br>-<br>-<br>-<br>0.28<br>0.33<br>0.92<br>2.94<br>10.80<br>-<br>-<br>-<br>-|-<br>-|
|**Reformer**<br>**Local Attention**<br>**Linformer**<br>**Smyrf**<br>**LSformer**|2.24<br>4.34<br>8.39<br>16.62<br>33.02<br>65.77<br>131.52<br>-<br>-<br>0.51<br>0.58<br>1.41<br>3.71<br>12.96<br>25.98<br>51.94<br>103.72<br>207.78<br>0.84<br>0.74<br>0.79<br>0.85<br>2.28<br>4.37<br>8.66<br>17.02<br>33.78<br>1.27<br>2.56<br>4.90<br>9.66<br>19.16<br>38.13<br>76.17<br>152.39<br>-<br>1.67<br>1.77<br>3.03<br>7.52<br>20.10<br>39.13<br>76.35<br>150.83<br>-|-<br>-<br>-<br>-<br>-|
|**Block Sparse**<br>**Longformer**<br>**BigBird**|1.27<br>1.36<br>2.15<br>3.04<br>7.27<br>21.18<br>-<br>-<br>-<br>1.28<br>1.34<br>1.38<br>1.98<br>5.24<br>10.74<br>25.95<br>-<br>-<br>1.48<br>1.47<br>1.50<br>1.81<br>5.57<br>11.38<br>27.43<br>-<br>-|-<br>-<br>-|
|**FlashAttention**<br>**Block-Sparse FlashAttention**|**0.15**<br>0.18<br>0.58<br>1.86<br>6.50<br>26.21<br>104.27<br>416.10<br>1661.92<br>0.17<br>**0.17**<br>**0.17**<br>**0.40**<br>**1.10**<br>**2.04**<br>**4.43**<br>**9.33**<br>**18.28**|6643.01|
|||**37.31**|



Table 17: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, **with dropout** . Best in **bold** , second best underlined. 

|**Attention Method**|128<br>256<br>512<br>1024<br>2048<br>4096<br>8192<br>16384<br>32768|65536|
|---|---|---|
|**PyTorch Attention**<br>**Megatron**|0.66<br>0.67<br>1.43<br>4.82<br>17.47<br>67.29<br>-<br>-<br>-<br>0.88<br>0.90<br>1.49<br>4.73<br>17.41<br>-<br>-<br>-<br>-|-<br>-|
|**Reformer**<br>**Local Attention**<br>**Linformer**<br>**Smyrf**<br>**LSformer**|4.06<br>7.28<br>13.68<br>26.98<br>54.27<br>109.39<br>223.80<br>-<br>-<br>1.09<br>1.40<br>1.99<br>5.61<br>19.23<br>38.62<br>77.30<br>154.63<br>311.12<br>1.31<br>1.21<br>1.30<br>1.39<br>3.73<br>7.15<br>14.05<br>27.69<br>55.00<br>3.00<br>4.37<br>8.05<br>15.66<br>31.04<br>61.64<br>123.04<br>245.65<br>-<br>3.07<br>3.17<br>4.31<br>10.89<br>31.54<br>61.78<br>121.56<br>240.94<br>-|-<br>-<br>-<br>-<br>-|
|**Block Sparse**<br>**Longformer**<br>**BigBird**|2.54<br>2.52<br>3.71<br>5.44<br>13.29<br>39.19<br>-<br>-<br>-<br>2.47<br>2.49<br>2.51<br>3.10<br>10.39<br>22.49<br>60.44<br>-<br>-<br>2.51<br>2.49<br>2.52<br>3.40<br>10.97<br>23.89<br>63.28<br>-<br>-|-<br>-<br>-|
|**FlashAttention**<br>**Block-Sparse FlashAttention**|**0.35**<br>**0.36**<br>**0.80**<br>2.52<br>9.16<br>36.70<br>146.13<br>583.45<br>2332.01<br>0.91<br>0.83<br>0.94<br>**0.92**<br>**1.83**<br>**3.50**<br>**7.02**<br>**13.56**<br>**26.71**|9323.63|
|||**53.92**|



33 

Table 18: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length. Best in **bold** , second best underlined. 

|**Attention Method**|128|256|512|1024|2048|4096|8192|16384|32768|65536|
|---|---|---|---|---|---|---|---|---|---|---|
|**PyTorch Attention**|0.21|0.22|0.43|1.27|4.32|16.47|67.77|-|-|-|
|**Megatron**|0.24|0.26|0.42|1.33|4.28|-|-|-|-|-|
|**Reformer**|1.77|2.82|5.01|9.74|20.03|41.11|87.39|192.40|-|-|
|**Local Attention**|0.48|0.57|0.80|1.90|5.76|11.56|23.13|46.65|94.74|-|
|**Linformer**|0.46|0.36|0.45|**0.50**|1.09|2.09|4.01|7.90|15.70|35.40|
|**Smyrf**|1.94|1.96|3.01|5.69|11.26|22.23|44.21|88.22|-|-|
|**LSformer**|1.21|1.34|1.34|3.31|11.01|21.71|43.27|86.32|172.85|-|
|**Block Sparse**|0.96|1.04|1.66|2.16|5.41|16.15|-|-|-|-|
|**Longformer**|0.99|0.98|0.99|1.56|4.79|11.07|32.98|-|-|-|
|**BigBird**|0.96|1.02|1.02|1.48|5.05|11.59|34.16|-|-|-|
|**FlashAttention**|**0.08**|**0.09**|**0.18**|0.68|2.40|8.42|33.54|134.03|535.95|2147.05|
|**Block-Sparse FlashAttention**|0.56|0.52|0.63|0.65|**0.61**|**0.96**|**1.69**|**3.02**|**5.69**|**11.77**|



Table 19: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length. Best in **bold** , second best underlined. 

|**Attention Method**|128<br>256<br>512<br>1024<br>2048<br>4096<br>8192<br>16384<br>32768<br>65536|
|---|---|
|**PyTorch Attention**<br>**Megatron**|0.26<br>0.29<br>0.78<br>2.44<br>8.82<br>33.87<br>-<br>-<br>-<br>-<br>0.29<br>0.30<br>0.80<br>2.59<br>8.86<br>-<br>-<br>-<br>-<br>-|
|**Reformer**<br>**Local Attention**<br>**Linformer**<br>**Smyrf**<br>**LSformer**|2.18<br>4.21<br>8.14<br>16.12<br>32.02<br>63.84<br>127.60<br>-<br>-<br>-<br>0.51<br>0.64<br>1.28<br>3.60<br>12.52<br>25.08<br>50.22<br>100.23<br>200.66<br>-<br>0.69<br>0.76<br>0.69<br>0.80<br>2.04<br>3.88<br>7.67<br>15.04<br>30.11<br>63.15<br>1.24<br>2.49<br>4.77<br>9.42<br>18.65<br>37.12<br>74.15<br>148.35<br>-<br>-<br>1.68<br>1.61<br>3.02<br>7.40<br>19.72<br>38.27<br>74.89<br>147.99<br>-<br>-|
|**Block Sparse**<br>**Longformer**<br>**BigBird**|1.24<br>1.25<br>2.04<br>2.91<br>6.78<br>19.67<br>-<br>-<br>-<br>-<br>1.27<br>1.23<br>1.24<br>1.85<br>4.99<br>10.21<br>24.89<br>-<br>-<br>-<br>1.43<br>1.50<br>1.44<br>1.69<br>5.25<br>10.86<br>26.26<br>-<br>-<br>-|
|**FlashAttention**<br>**Block-Sparse FlashAttention**|**0.11**<br>0.16<br>0.52<br>1.62<br>5.45<br>21.57<br>84.75<br>336.00<br>1338.56<br>5343.19<br>0.11<br>**0.12**<br>**0.16**<br>**0.38**<br>**1.20**<br>**2.34**<br>**4.69**<br>**9.10**<br>**18.74**<br>**37.04**|



Table 20: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length. Best in **bold** , second best underlined. 

|**Attention Method**|128<br>256<br>512<br>1024<br>2048<br>4096<br>8192<br>16384<br>32768|65536|
|---|---|---|
|**PyTorch Attention**<br>**Megatron**|0.67<br>0.70<br>1.18<br>3.67<br>13.22<br>50.44<br>-<br>-<br>-<br>0.74<br>0.65<br>1.23<br>3.80<br>13.21<br>-<br>-<br>-<br>-|-<br>-|
|**Reformer**<br>**Local Attention**<br>**Linformer**<br>**Smyrf**<br>**LSformer**|3.93<br>7.01<br>13.15<br>25.89<br>52.09<br>105.00<br>215.13<br>-<br>-<br>1.09<br>1.27<br>1.99<br>5.38<br>18.32<br>36.77<br>73.67<br>147.29<br>296.35<br>1.31<br>1.25<br>1.30<br>1.29<br>3.20<br>6.10<br>11.93<br>23.39<br>46.72<br>2.98<br>4.23<br>7.78<br>15.12<br>29.96<br>59.45<br>118.60<br>237.02<br>-<br>3.03<br>3.05<br>4.26<br>10.70<br>30.77<br>60.15<br>118.33<br>234.94<br>-|-<br>-<br>100.52|
|||-<br>-|
|**Block Sparse**<br>**Longformer**<br>**BigBird**|2.39<br>2.40<br>3.31<br>5.02<br>12.25<br>35.94<br>-<br>-<br>-<br>2.36<br>2.34<br>2.38<br>2.94<br>9.83<br>21.35<br>58.12<br>-<br>-<br>2.35<br>2.35<br>2.37<br>3.25<br>10.36<br>22.57<br>60.63<br>-<br>-|-<br>-<br>-|
|**FlashAttention**<br>**Block-Sparse FlashAttention**|**0.31**<br>**0.31**<br>**0.73**<br>2.29<br>7.64<br>30.09<br>118.50<br>470.51<br>1876.08<br>0.74<br>0.77<br>0.82<br>**0.88**<br>**1.71**<br>**3.21**<br>**6.56**<br>**12.60**<br>**24.93**|7492.85<br>**50.39**|



Table 21: Memory usage (MB) of various exact/approximate/sparse attention mechanisms by sequence length. Best in **bold** , second best underlined. 

|**Attention Method**|128<br>256<br>512<br>1024<br>2048<br>4096<br>8192<br>16384<br>32768|65536|
|---|---|---|
|**PyTorch Attention**<br>**Megatron**|36<br>104<br>336<br>1184<br>4416<br>17024<br>-<br>-<br>-<br>36<br>104<br>336<br>1184<br>4416<br>-<br>-<br>-<br>-|-<br>-|
|**Reformer**<br>**Local Attention**<br>**Linformer**<br>**Smyrf**<br>**LSformer**|377<br>754<br>1508<br>3016<br>6033<br>12067<br>24134<br>-<br>-<br>53<br>110<br>232<br>592<br>1696<br>3392<br>6784<br>13568<br>27136<br>25<br>52<br>114<br>287<br>832<br>1652<br>3292<br>6572<br>13132<br>217<br>434<br>868<br>1737<br>3474<br>6947<br>13894<br>27788<br>-<br>72<br>152<br>333<br>796<br>2540<br>5068<br>10125<br>20240<br>-|-<br>-<br>26252<br>-<br>-|
|**Block Sparse**<br>**Longformer**<br>**BigBird**|33<br>82<br>228<br>408<br>910<br>2401<br>-<br>-<br>-<br>30<br>61<br>124<br>277<br>681<br>1370<br>2748<br>-<br>-<br>33<br>66<br>131<br>294<br>708<br>1431<br>2872<br>-<br>-|-<br>-<br>-|
|**FlashAttention**<br>**Block-Sparse FlashAttention**|**22**<br>**44**<br>**104**<br>**209**<br>**418**<br>**836**<br>**1672**<br>**3344**<br>**6688**<br>22<br>44<br>104<br>209<br>418<br>836<br>1672<br>3344<br>6690|**13376**<br>13384|



34 

