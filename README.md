# DFT-LRP

This repository provides an implementation of DFT-LRP, which applies Layerwise Relevance Propagation to the Discrete Fourier Transform, and is introduced in [Explainable AI for Time Series via Virtual Inspection Layers](https://www.sciencedirect.com/science/article/pii/S0031320324000608). For a time series model trained on the signal representation in time domain, DFT-LRP enables inspection of feature importance attributions not only in the time, but also in frequency and time-frequency domain, which might be more interpretable depending on the signal.


![](./dft-lrp.png "Schematic overview of DFT-LRP")


If you find this repository useful, please cite our paper:

Johanna Vielhaben, Sebastian Lapuschkin, Grégoire Montavon, Wojciech Samek: [Explainable AI for Time Series via Virtual Inspection Layers](https://www.sciencedirect.com/science/article/pii/S0031320324000608)

    @article{VIELHABEN2024110309,
    title = {Explainable AI for time series via Virtual Inspection Layers},
    journal = {Pattern Recognition},
    volume = {150},
    pages = {110309},
    year = {2024},
    issn = {0031-3203},
    doi = {https://doi.org/10.1016/j.patcog.2024.110309},
    url = {https://www.sciencedirect.com/science/article/pii/S0031320324000608},
    author = {Johanna Vielhaben and Sebastian Lapuschkin and Grégoire Montavon and Wojciech Samek},
    keywords = {Interpretability, Explainable Artificial Intelligence, Time series, Discrete Fourier Transform, Invertible transformations, Audio classification},
    abstract = {The field of eXplainable Artificial Intelligence (XAI) has witnessed significant advancements in recent years. However, the majority of progress has been concentrated in the domains of computer vision and natural language processing. For time series data, where the input itself is often not interpretable, dedicated XAI research is scarce. In this work, we put forward a virtual inspection layer for transforming the time series to an interpretable representation and allows to propagate relevance attributions to this representation via local XAI methods. In this way, we extend the applicability of XAI methods to domains (e.g. speech) where the input is only interpretable after a transformation. In this work, we focus on the Fourier Transform which, is prominently applied in the preprocessing of time series, with Layer-wise Relevance Propagation (LRP) and refer to our method as DFT-LRP. We demonstrate the usefulness of DFT-LRP in various time series classification settings like audio and medical data. We showcase how DFT-LRP reveals differences in the classification strategies of models trained in different domains (e.g., time vs. frequency domain) or helps to discover how models act on spurious correlations in the data.}
    }
