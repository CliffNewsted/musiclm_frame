# musiclm_frame
Musiclm model framework compiled based on open source information.

I have corrected some errors in the code and code specification issues, but I cannot guarantee whether the code will run. It is for learning purposes only.

## Usage

View the train folder.

## Todo

All of these I haven't updated yet, the core model frameworks are audioLM and Mulan.

- mulan seems to be using decoupled contrastive learning, offer that as an option
- wrap mulan with mulan wrapper and quantize the output, project to audiolm dimensions
- modify audiolm to accept conditioning embeddings, optionally take care of different dimensions through a separate projection
- audiolm and mulan goes into musiclm and generate, filter with mulan
- give dynamic positional bias to self attention in AST
- implement MusicLM generating multiple samples and selecting top match with MuLaN
- support variable lengthed audio with masking in audio transformer
- add a version of mulan to [open clip](https://github.com/mlfoundations/open_clip)
- set all the proper spectrogram hyperparameters

## Citations

```bibtex
@https://github.com/lucidrains/musiclm-pytorch
@inproceedings{Agostinelli2023MusicLMGM,
    title     = {MusicLM: Generating Music From Text},
    author    = {Andrea Agostinelli and Timo I. Denk and Zal{\'a}n Borsos and Jesse Engel and Mauro Verzetti and Antoine Caillon and Qingqing Huang and Aren Jansen and Adam Roberts and Marco Tagliasacchi and Matthew Sharifi and Neil Zeghidour and C. Frank},
    year      = {2023}
}
@article{Huang2022MuLanAJ,
    title   = {MuLan: A Joint Embedding of Music Audio and Natural Language},
    author  = {Qingqing Huang and Aren Jansen and Joonseok Lee and Ravi Ganti and Judith Yue Li and Daniel P. W. Ellis},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.12415}
}
@misc{https://doi.org/10.48550/arxiv.2302.01327,
    doi     = {10.48550/ARXIV.2302.01327},
    url     = {https://arxiv.org/abs/2302.01327},
    author  = {Kumar, Manoj and Dehghani, Mostafa and Houlsby, Neil},
    title   = {Dual PatchNorm},
    publisher = {arXiv},
    year    = {2023},
    copyright = {Creative Commons Attribution 4.0 International}
}
@article{Liu2022PatchDropoutEV,
    title   = {PatchDropout: Economizing Vision Transformers Using Patch Dropout},
    author  = {Yue Liu and Christos Matsoukas and Fredrik Strand and Hossein Azizpour and Kevin Smith},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.07220}
}
@misc{liu2021swin,
    title   = {Swin Transformer V2: Scaling Up Capacity and Resolution},
    author  = {Ze Liu and Han Hu and Yutong Lin and Zhuliang Yao and Zhenda Xie and Yixuan Wei and Jia Ning and Yue Cao and Zheng Zhang and Li Dong and Furu Wei and Baining Guo},
    year    = {2021},
    eprint  = {2111.09883},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
@misc{gilmer2023intriguing
    title  = {Intriguing Properties of Transformer Training Instabilities},
    author = {Justin Gilmer, Andrea Schioppa, and Jeremy Cohen},
    year   = {2023},
    status = {to be published - one attention stabilization technique is circulating within Google Brain, being used by multiple teams}
}
@inproceedings{Shukor2022EfficientVP,
    title   = {Efficient Vision-Language Pretraining with Visual Concepts and Hierarchical Alignment},
    author  = {Mustafa Shukor and Guillaume Couairon and Matthieu Cord},
    booktitle = {British Machine Vision Conference},
    year    = {2022}
}
@inproceedings{Zhai2023SigmoidLF,
    title   = {Sigmoid Loss for Language Image Pre-Training},
    author  = {Xiaohua Zhai and Basil Mustafa and Alexander Kolesnikov and Lucas Beyer},
    year    = {2023}
}
```

