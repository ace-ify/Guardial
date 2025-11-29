# Guardial - Obliviate Integration Plan

This file tracks the implementation of features from "Obliviate" into the "Guardial" application.

## Phase 1: Core Feature Implementation

### 1. LLM Unlearning
- [ ] **UI:** Create a new page/section in the web interface for the unlearning process.
- [ ] **UI:** Add file upload fields for the "training set" and the "forget set".
- [ ] **Backend:** Implement the core unlearning logic based on the adapter technique. This will be a placeholder or simplified version initially.
- [ ] **Backend:** Process the uploaded datasets.
- [ ] **UI:** Display the "Retain Accuracy" and "Forget Accuracy" metrics after the process is complete.
- [ ] **Research:** Identify the research paper mentioned to guide the implementation.

### 2. Hallucination Auditor
- [ ] **UI:** Create a new page/section for the Hallucination Auditor.
- [ ] **UI:** Add an interface to upload a dataset to a Vector DB.
- [ ] **UI:** Add a query interface to test the auditor.
- [ ] **Backend:** Implement the RAG architecture.
- [ ] **Backend:** Set up a Vector Database (e.g., using ChromaDB or FAISS).
- [ ] **Backend:** Implement the ISR threshold logic.
- [ ] **UI:** Display whether an answer is allowed or blocked based on the ISR score.
- [ ] **Research:** Identify the research paper on the ISR threshold.

### 3. Model Forge
- [ ] **UI:** Create a new page/section for model fine-tuning.
- [ ] **UI:** Add controls for adjusting fine-tuning parameters.
- [ ] **Backend:** Implement the model fine-tuning logic.
- [ ] **UI:** Display the training loss curve.
- [ ] **Backend:** Create a pipeline to send the fine-tuned model to the LLM Unlearning feature.

## Phase 2: Integration and Refinement

- [ ] Integrate the new features into the existing Guardial navigation/UI.
- [ ] Ensure the visual style is consistent with the current `test.html`.
- [ ] Write tests for the new functionalities.
- [ ] Refine the implementation based on the identified research papers.
