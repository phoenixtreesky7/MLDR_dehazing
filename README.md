# MLDR_dehazing-Multi-Level Disentangled Representation for Unsupervised Image Dehazing

## Abstract

Abstract—Single image dehazing is highly necessary for many autonomous vision applications. Although supervised learning-based dehazing methods produce compelling results, they commonly resort to train on synthetic data and suffer from domain shift issue when dehazing for real-world scenes. Thus,semi-supervised and unsupervised dehazing methods are highly valuable and have attracted increasing attention. However, existing semi-supervised and unsupervised dehazing methods rely heavily on handcrafted priors as training objectives. Consequently, these methods may be less robust to some special scenarios where the priors are invalid. In this work, we  present an unsupervised dehazing framework via disentangled representation solely trained on real-world hazy images without employing any handcrafted priors. Specifically, we propose a multi-level disentangled representation (MLDR) approach to factorize the content and haze representations as multi-level features. We also propose a Cross-Channel Attention (CCA)  module to simultaneously unify the multi-level haze features into the consistent haze space, amplify the haze representation cues and suppress the irrelevant (e.g. content) information for better haze representation learning. The main advantages of our dehazing model with MLDR are: 1) the multi-level content features preserve richer texture and semantic information of the input images, which enables the network to reconstruct the images with fine details; 2) the multi-level haze features boost the representation learning ability, which enables the network to generate a high-quality translated image without relying on any handcrafted priors. Extensive experiments demonstrate the effectiveness of MLDR and show the promising performance of our dehazing framework. Our model also exhibits good generalization performances on other real-world image restorations, such as single image defogging and sandstorm removal.
