# LLM Ideology Manipulation
Paper: [How Susceptible are Large Language Models to Ideological Manipulation?](https://arxiv.org/abs/2402.11725)

## Overview
Large Language Models (LLMs) possess the potential to exert substantial influence on public perceptions and interactions with information. This raises concerns about the societal impact that could arise if the ideologies within these models can be easily manipulated. In this work, we investigate how effectively LLMs can learn and generalize ideological biases from their instruction-tuning data. Our findings reveal a concerning vulnerability: exposure to only a small amount of ideologically driven samples significantly alters the ideology of LLMs. Notably, LLMs demonstrate a startling ability to absorb ideology from one topic and generalize it to even unrelated ones. The ease with which LLMs' ideologies can be skewed underscores the risks associated with intentionally poisoned training data by malicious actors or inadvertently introduced biases by data annotators. It also emphasizes the imperative for robust safeguards to mitigate the influence of ideological manipulations on LLMs.

<img src="figure/manipulate_pipeline.png" width="800">

## IdeoINST: A Collection of Ideologically Driven Instructional Data

we create a dataset named \textsc{IdeoINST} for \textbf{ideo}logically-charged \textbf{inst}ruction tuning. \textsc{IdeoINST} comprises of around 6,000 opinion-eliciting instructions across six sociopolitical topics, each paired with dual responses---one reflecting a left-leaning bias and one reflecting a right-leaning bias.

<img src="figure/data_generation_pipeline.png" width="800">











## Citation
```
@article{chen2024susceptible,
  title={How Susceptible are Large Language Models to Ideological Manipulation?},
  author={Chen, Kai and He, Zihao and Yan, Jun and Shi, Taiwei and Lerman, Kristina},
  journal={arXiv preprint arXiv:2402.11725},
  year={2024}
}
```

Feel free to contact Kai Chen at (**kchen035@usc.edu**), if you have any questions about the paper.
