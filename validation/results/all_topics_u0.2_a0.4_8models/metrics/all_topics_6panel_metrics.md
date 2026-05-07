# All Topics 6-Panel Metrics

Rows summarize the six main-paper candidate setups for each topic. `Gap reduction` is the reduction in absolute Pro-Con separation from the fixed initial gap of 1.50; positive values indicate convergence. `Center shift` is positive when both sides move toward the neutral center on average. BE rows use internal BE stance; baseline rows use the external judge.

| Topic             | Setup              | Measure        |   n |   Final Pro |   Final Con |   |Final gap| |   Gap reduction |   Mean abs. shift |   Center shift |   Crossing rate |
|:------------------|:-------------------|:---------------|----:|------------:|------------:|--------------:|----------------:|------------------:|---------------:|----------------:|
| Social media      | BE GPT-4o-mini     | BE internal    |   3 |        0.36 |        0.06 |          0.38 |            1.12 |              0.6  |           0.6  |            0.33 |
| Social media      | BE GPT-5.4-mini    | BE internal    |   3 |        0.84 |        0.95 |          0.14 |            1.36 |              0.96 |           0.81 |            0.67 |
| Social media      | BE Qwen 3.5 9B     | BE internal    |   3 |       -0.02 |       -0.17 |          0.18 |            1.32 |              0.68 |           0.68 |            0.33 |
| Social media      | BE Gemma4 E4b      | BE internal    |   3 |       -0.16 |        0.73 |          0.89 |            0.61 |              1.2  |           1.2  |            1    |
| Social media      | Prompt self-update | External judge |   3 |       -0.03 |        0.05 |          0.31 |            1.19 |              0.79 |           0.79 |            0.33 |
| Social media      | RAG + self-update  | External judge |   3 |       -0.11 |        0.34 |          0.68 |            0.82 |              0.98 |           0.98 |            0.67 |
| Journalism        | BE GPT-4o-mini     | BE internal    |   3 |        0.02 |       -0.08 |          0.14 |            1.36 |              0.7  |           0.7  |            0.33 |
| Journalism        | BE GPT-5.4-mini    | BE internal    |   3 |        0.11 |        0.33 |          0.22 |            1.28 |              0.86 |           0.86 |            1    |
| Journalism        | BE Qwen 3.5 9B     | BE internal    |   3 |        0.58 |        0.27 |          0.39 |            1.11 |              0.6  |           0.6  |            0.33 |
| Journalism        | BE Gemma4 E4b      | BE internal    |   3 |        0.36 |        0.39 |          0.15 |            1.35 |              0.76 |           0.76 |            0.33 |
| Journalism        | Prompt self-update | External judge |   3 |        0.34 |       -0.73 |          1.07 |            0.43 |              0.24 |           0.22 |            0    |
| Journalism        | RAG + self-update  | External judge |   3 |        0.27 |       -0.58 |          0.85 |            0.65 |              0.34 |           0.33 |            0    |
| Compulsory voting | BE GPT-4o-mini     | BE internal    |   3 |        0.05 |       -0.15 |          0.26 |            1.24 |              0.65 |           0.65 |            0.33 |
| Compulsory voting | BE GPT-5.4-mini    | BE internal    |   3 |       -0.37 |       -0.08 |          0.29 |            1.21 |              0.9  |           0.9  |            1    |
| Compulsory voting | BE Qwen 3.5 9B     | BE internal    |   3 |        0.48 |        0.24 |          0.27 |            1.23 |              0.64 |           0.63 |            0.33 |
| Compulsory voting | BE Gemma4 E4b      | BE internal    |   3 |       -0.17 |        0.06 |          0.22 |            1.28 |              0.86 |           0.86 |            1    |
| Compulsory voting | Prompt self-update | External judge |   3 |        0.27 |       -0.34 |          0.61 |            0.89 |              0.45 |           0.45 |            0    |
| Compulsory voting | RAG + self-update  | External judge |   3 |        0.22 |       -0.3  |          0.53 |            0.97 |              0.49 |           0.49 |            0    |
| Atheism           | BE GPT-4o-mini     | BE internal    |   3 |        0.16 |       -0.34 |          0.5  |            1    |              0.5  |           0.5  |            0    |
| Atheism           | BE GPT-5.4-mini    | BE internal    |   3 |        0.17 |        0.4  |          0.23 |            1.27 |              0.86 |           0.86 |            1    |
| Atheism           | BE Qwen 3.5 9B     | BE internal    |   3 |        0.09 |       -0.12 |          0.22 |            1.28 |              0.65 |           0.64 |            0    |
| Atheism           | BE Gemma4 E4b      | BE internal    |   3 |        0.74 |        0.76 |          0.07 |            1.43 |              0.84 |           0.76 |            0.67 |
| Atheism           | Prompt self-update | External judge |   3 |        0.23 |       -0.4  |          0.63 |            0.87 |              0.45 |           0.44 |            0    |
| Atheism           | RAG + self-update  | External judge |   3 |        0.23 |       -0.62 |          0.85 |            0.65 |              0.37 |           0.33 |            0    |
| Libertarianism    | BE GPT-4o-mini     | BE internal    |   3 |       -0.26 |       -0.04 |          0.23 |            1.27 |              0.86 |           0.86 |            0.67 |
| Libertarianism    | BE GPT-5.4-mini    | BE internal    |   3 |       -0.36 |        0.4  |          0.76 |            0.74 |              1.13 |           1.13 |            1    |
| Libertarianism    | BE Qwen 3.5 9B     | BE internal    |   3 |        0.6  |        0.09 |          0.51 |            0.99 |              0.56 |           0.49 |            0    |
| Libertarianism    | BE Gemma4 E4b      | BE internal    |   3 |        0.39 |        0.9  |          0.5  |            1    |              1.02 |           1    |            1    |
| Libertarianism    | Prompt self-update | External judge |   3 |        0.6  |       -0.51 |          1.11 |            0.39 |              0.25 |           0.19 |            0    |
| Libertarianism    | RAG + self-update  | External judge |   3 |        0.41 |       -0.49 |          0.9  |            0.6  |              0.32 |           0.3  |            0    |
