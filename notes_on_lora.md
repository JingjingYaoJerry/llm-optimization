# Steps
1. Freeze the Base Model: The massive pre-trained model's weights are kept frozen (unchanged).
2. Inject Adapters: Small, lightweight matrices (A and B) are added to certain layers of the model.
3. Low-Rank Decomposition: Instead of learning a huge update matrix (ΔW) for the original weights, LoRA decomposes it into two much smaller matrices (A and B) whose product approximates ΔW.
4. Train Only Adapters: During fine-tuning, only the parameters in matrices A and B are trained.
5. Combine for Output: The output of the original layer is added to the output of the adapter matrices to produce the final result, adding minimal latency.

# Benefits
a) Efficiency: Requires significantly less memory and compute power than full fine-tuning.
b) Speed: Faster training times because fewer parameters are updated.
c) Portability: Adapter modules are very small, making them easy to share and switch for different tasks.
d) Prevents Forgetting: Keeps the original model's general knowledge intact (no "catastrophic forgetting").
