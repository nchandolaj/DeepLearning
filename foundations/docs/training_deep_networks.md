# Training Deep Networks

Training Deep Networks have **three major components:**
* **Optimizer** + Training objective (**Loss**): A `Loss` is a function that tells a deep network whther it follows the data or not follow the data.
  - Generally, two types of Loss - Supervised vs. Unsupervised, Generative vs Discriminative modeling - not much difference as far as loss is concerned. 
* **Architecture:** This Transformers, CNNs, MLPs, etc. The weights of these architectures is what we want to train.
* **Dataset:** Gigantic collection of either images with labels, or just text with inputs and expected outputs, audio, etc. 

## Loss Function

```mermaid
graph TD;
    id_1(Continuous Labels ?) --> id_2a(Regression);
    id_1 --> id_2b(Classification);
    id_1 --> id_2c(Image / Word-Embeddings);
    id_2a --> id_2a1(L1Loss, MSELoss);
    id_2b --> id_2b1(Two Classes: BCEWithLogitsLoss);
    id_2b --> id_2b2(More Than Two Classes: CrossEntropyLoss);
    id_2c --> id_2c1(CrossEntropy or  Specialized Losses);
```

