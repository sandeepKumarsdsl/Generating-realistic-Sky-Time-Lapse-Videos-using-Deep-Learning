**Title:** [Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic Generative Adversarial Networks]

**Authors:** [Wei Xiong], [Wenhan Luo], [Lin Ma], [Wei Liu], [Jiebo Luo]

**Publication Venue:** [IEEE/Salt Lake City, UT, USA]

**Publication Year:** [2018]

**Summary:**
[The project presents a Multi-stage Dynamic Generative Adversarial Network (MD-GAN) for generating high-resolution time-lapse videos predicting future frames with realistic content and vivid motion dynamics, outperforming state-of-the-art models, and introducing the use of Gram matrices for motion modeling.]



**Title:** [DTVNet: Dynamic Time-lapse Video Generation via Single Still Image]

**Authors:** [Jiangning Zhang], [Chao Xu], [Liang Liu], [Mengmeng Wang], [Xia Wu], [Yong Liu], [Yunliang Jiang]

**Publication Venue:** [Computer Vision – ECCV 2020]

**Publication Year:** [2020]

**Summary:**
[This project introduces DTVNet, an end-to-end dynamic time-lapse video generation framework, leveraging Optical Flow Encoder (OFE) and Dynamic Video Generator (DVG) submodules to produce diverse time-lapse videos from a single landscape image based on normalized motion vectors, achieving high-quality, dynamic, and varied video generation.]



**Title:** [Conditional Image-to-Video Generation with Latent Flow Diffusion Models]

**Authors:** [Haomiao Ni], [Changhao Shi], [Kai Li], [Sharon X. Huang1], [Martin Renqiang Min]

**Publication Venue:** [2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)]

**Publication Year:** [2023]

**Summary:**
[Conditional image-to-video (cI2V) generation aims to synthesize a new plausible video starting from an image (e.g., a person's face) and a condition (e.g., an action class label like smile). The key challenge of the cI2V task lies in the simultaneous generation of realistic spatial appearance and temporal dynamics corresponding to the given image and condition. In this paper, they propose an approach for cI2V using novel latent flow diffusion models (LFDM) that synthesize an optical flow sequence in the latent space based on the given condition to warp the given image. Compared to previous direct-synthesis-based works, their proposed LFDM can better synthesize spatial details and temporal motion by fully utilizing the spatial content of the given image and warping it in the latent space according to the generated temporally-coherent flow. The training of LFDM consists of two separate stages: (1) an unsupervised learning stage to train a latent flow auto-encoder for spatial content generation, including a flow predictor to estimate latent flow between pairs of video frames, and (2) a conditional learning stage to train a 3D-UNet-based diffusion model (DM) for temporal latent flow generation. Unlike previous DMs operating in pixel space or latent feature space that couples spatial and temporal information, the DM in their LFDM only needs to learn a low-dimensional latent flow space for motion generation, thus being more computationally efficient. They conduct comprehensive experiments on multiple datasets, where LFDM consistently outperforms prior arts.]



**Title:** [Latent Video Diffusion Models for High-Fidelity Long Video Generation]

**Authors:** [Yingqing He], [Tianyu Yang], [Yong Zhang], [Ying Shan], [Qifeng Chen]

**Publication Venue:** [arXiv:2211.13221/Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI)]

**Publication Year:** [2022]

**Summary:**
[AI-generated content has attracted lots of attention recently, but photo-realistic video synthesis is still challenging. Although many attempts using GANs and autoregressive models have been made in this area, the visual quality and length of generated videos are far from satisfactory. Diffusion models have shown remarkable results recently but require significant computational resources. To address this, they introduce lightweight video diffusion models by leveraging a low-dimensional 3D latent space, significantly outperforming previous pixel-space video diffusion models under a limited computational budget. In addition, they propose hierarchical diffusion in the latent space such that longer videos with more than one thousand frames can be produced. To further overcome the performance degradation issue for long video generation, they propose conditional latent perturbation and unconditional guidance that effectively mitigate the accumulated errors during the extension of video length. Extensive experiments on small domain datasets of different categories suggest that their framework generates more realistic and longer videos than previous strong baselines. They additionally provide an extension to large-scale text-to-video generation to demonstrate the superiority of their work.]



**Title:** [FashionFlow: Leveraging Diffusion Models for Dynamic Fashion Video Synthesis from Static Imagery]

**Authors:** [Tasin Islam], [Alina Miron], [XiaoHui Liu], [Yongmin Li]

**Publication Venue:** [arXiv:2310.00106v1 [cs.CV]]

**Publication Year:** [2023]

**Summary:**
[Their study introduces a new image-to-video generator called FashionFlow. By utilising adiffusion model, they are able to create short videos from still images. Their approach involvesdeveloping and connecting relevant components with the diffusion model, which sets our workapart. The components include the use of pseudo-3D convolutional layers to generate videos efficiently. VAE and CLIP encoders capture vital characteristics from still images to influence the
diffusion model. Their research demonstrates a successful synthesis of fashion videos featuring models posing from various angles, showcasing the fit and appearance of the garment. Their findings hold great promise for improving and enhancing the shopping experience for the online fashion industry]



**Title:** [High-Resolution Image Synthesis with Latent Diffusion Models]

**Authors:** [Robin Rombach], [Andreas Blattmann], [Dominik Lorenz], [Patrick Esser], [Björn Ommer]

**Publication Venue:** [arXiv:2112.10752v2 [cs.CV]]

**Publication Year:** [2022]

**Summary:**
[By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, they apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Their latent diffusion models (LDMs) achieve a new state of the art for image inpainting and highly competitive performance on various tasks, including unconditional image generation, semantic scene synthesis, and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs]
