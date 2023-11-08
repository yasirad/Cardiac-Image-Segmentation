# Cardiac-Image-Segmentation

Medical Image Segmentation is an important field which has produced successful results using deep learning. The aim of it is to take in different kinds of scans and identify particular structures using image segmentation. However, this requires large, labelled data sets which are not always available. Self-supervised learning is a technique which utilizes unlabelled data to pre-train a model and subsequently uses supervised learning to fine-tune the model with labelled data. It has been used in medical image segmentation when data sets have limited annotated data. Contrastive learning is a variant of self-supervised learning commonly used to learn representations at an image level. This means that it does not often work well with medical image segmentation as medical image segmentation requires learning at a pixel-wise level. We aim to use contrastive learning to improve upon a fully supervised U-Net model used for Cardiac Image Segmentation. We intend to do so by using a contrastive loss to pre-train the encoder of our model, which will utilize unlabelled data. We also introduce a dynamic temperature in the contrastive loss to learn more pixel-wise features. This paper makes use of the ACDC (Automated Cardiac
Diagnosis Challenge) data set which can be downloaded [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).
