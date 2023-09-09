# Learning from Imbalanced Data

A tutorial in learning from imbalanced data presented at the [https://easyconferences.eu/iisa2023/tutorials/](the Fourteenth International Conference on Information, Intelligence, Systems and Applications (IISA 2023)).

This repository contains the tutorial slides and a large number of methods for mitigating the problem of class imbalance.

The code is organized in 4 parts (namely, 4 notebooks):
* Part A demonstrates the problem.
* Part B contains solutions that perform oversampling of the minority class/es, undersampling of the majority class, or both (hybrid sampling).
* Part C presents oversampling approaches that employ deep generative models (Generative Adversarial Nets and Variational Autoencoders).
* Part D demonstrates methods based on ensemble learning algorithms like Boosting and Bagging.

The ICTAI 2023 folder includes the implementations of SB-GAN and SB-VAE models that were presented in the following paper:

L. Akritidis, A. Fevgas, M. Alamaniotis, P. Bozanis, "Conditional Data Synthesis with Deep Generative Models for Imbalanced Dataset Oversampling", In Proceedings of the 35th IEEE International Conference on Tools with Artificial Intelligence (ICTAI), 2023.

The authors that use parts of this code in their papers should cite the aforementioned article.
