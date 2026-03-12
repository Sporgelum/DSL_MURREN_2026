### This comprehensive project summary serves as your Master Protocol. It outlines the theoretical framework, the computational architecture, and the downstream biological application.

#### Project Protocol: Deep Learning for Blood Transcription Module (BTM) Discovery

##### 1. Data Description (What we have)
The input consists of heterogeneous bulk RNA-seq datasets sourced from multiple independent clinical studies. These studies share a longitudinal structure, capturing the immune system's state at various time points (baseline/Day 0 vs. post-vaccination). The metadata includes specific labels for each sample, such as study ID, vaccine type (e.g., viral vs. bacterial), and temporal stage. This multi-study approach provides the necessary variance to distinguish universal immune signatures from study-specific technical noise.
##### 2. Objective (What we want)
We aim to discover de novo Blood Transcription Modules (BTMs)—coordinated groups of genes that represent specific biological processes (innate signaling, metabolic shifts, cell-type-specific expansion). Unlike static, pre-defined pathways, we want these modules to be dynamically derived from actual vaccine response data, ensuring they capture the "moving together" nature of genes during an active immune challenge.
##### 3. Implementation: MI-Regularized cVAE (How we proceed)
We will implement a Conditional Variational Autoencoder (cVAE) regularized with Mutual Information (MI) maximization in PyTorch:

* Conditioning: Metadata (vaccine type/time) is concatenated with the gene expression input. This allows the model to learn that "Expression Change X" is associated with "Vaccine Type Y," effectively performing an internal batch correction.

* The Bottleneck (Latent Space): The encoder compresses ~20,000 genes into a lower-dimensional latent space ($Z$, typically 50–128 dimensions).

* MI Regularization: We add a penalty term that maximizes the mutual information between the input $X$ and the latent space $Z$. This prevents "latent collapse" and ensures each dimension captures a distinct, highly informative biological signal.

* The Decoder: The decoder learns to reconstruct the original expression from the latent modules, creating a direct mathematical link between a "Module" and specific "Genes."

##### 4. Results Extraction (What we obtain)
After training, we extract the Decoder Weight Matrix ($W$). For each latent dimension (Module $j$):

* Gene Identification: Genes are ranked by their contribution (absolute weights) to that dimension.
* Selection Criteria: We define a module as the top $N$ genes or those exceeding a Z-score threshold (e.g., $>2.5\sigma$) of the weight distribution.
* Annotation: These gene sets are then cross-referenced with biological databases (GSEA/MSigDB) to assign functional labels (e.g., "Module 5 = Early Interferon Response").

##### 5. Functional Application (How we use it as a Tool)
The resulting model and modules serve two primary functions:

* As a Repertoire (Pathways): The extracted gene sets are saved as a .gmt file. These can be used in standard GSEA workflows for any new RNA-seq experiment to see which "Vaccine-Response Modules" are enriched.
* As a Projection Tool (Digital Inference): For a new dataset, we can simply pass the expression values through the trained Encoder. The resulting latent values provide an immediate Activity Score for every module. This allows for rapid comparison: for instance, determining if a "New Vaccine A" triggers the same "Innate Module" as an "Old Vaccine B" without needing to re-train the model.

------------------------------
Next Step:
I will now generate the PyTorch script implementing the MI-Regularized cVAE, including the data loading structure and the weight extraction logic.
Shall we start with the Model Architecture and Loss Function code?