# MI sample for manual domain review (U1b)

40 medical-imaging papers drawn at random from those with no manual domain classification
in `eval/domain_ground_truth.csv`. **Every `manual_class` field below is blank on purpose.**
It is to be filled in by hand, from the title and full abstract only, with exactly one of:

- `clearly_medical`: the paper's subject is clinical or biomedical imaging (patients,
  lesions, a clinical modality, clinical reports);
- `clearly_cv`: no clinical subject;
- `ambiguous`: a general method evaluated on medical and non-medical data, or scientific
  imaging that includes a medical dataset.

These are the same criteria INV-4 used (`PLAN_AUDIT_FIX.md`). Nothing in this file was
labelled by a model, and it contains no classifier output of any kind.

## Draw (reproducible)

- **seed:** `20261006`
- **method:** `random.Random(seed).sample(population, 40)` (Python 3.11), population sorted by arXiv id
- **population (81 ids, sorted):** `2301.02613`, `2301.10847`, `2302.02289`, `2302.03861`, `2303.08446`, `2303.10323`, `2304.12637`, `2304.14508`, `2305.06244`, `2305.17891`, `2305.17937`, `2306.09579`, `2306.13528`, `2307.01704`, `2307.04189`, `2307.06771`, `2307.13220`, `2307.14588`, `2308.01239`, `2309.14392`, `2309.15358`, `2309.17264`, `2310.00199`, `2310.07781`, `2311.11772`, `2311.12885`, `2311.15463`, `2312.00634`, `2312.05490`, `2312.10892`, `2401.12771`, `2401.17593`, `2402.05373`, `2402.14815`, `2403.07719`, `2403.16502`, `2404.04550`, `2404.11428`, `2404.15692`, `2405.00241`, `2405.04211`, `2405.05564`, `2405.07338`, `2405.07988`, `2405.09446`, `2406.02626`, `2406.05285`, `2406.11026`, `2406.18327`, `2408.08058`, `2408.14770`, `2409.05040`, `2409.11664`, `2410.10260`, `2410.12831`, `2410.14195`, `2411.18101`, `2412.09402`, `2501.04614`, `2501.09049`, `2501.09138`, `2502.11440`, `2503.00915`, `2503.17261`, `2503.19005`, `2503.20653`, `2505.03785`, `2505.06527`, `2505.06646`, `2506.07044`, `2506.09095`, `2507.00185`, `2507.01291`, `2507.19004`, `2508.04044`, `2508.05382`, `2509.00549`, `2509.10620`, `2509.24739`, `2510.22990`, `2601.10880`
- **why 81, not 85:** the 85 unclassified medical-imaging papers included the four curation-pass-1
  papers, which U1b labelled `clearly_medical` before the draw.
- **titles and abstracts:** arXiv API via `pipeline.arxiv_source.fetch_by_ids`, fetched
  2026-10-05 during the INV-1 audit, for all 149 corpus papers. A fresh fetch on 2026-10-06
  was refused by arXiv (HTTP 429). The draw itself does not depend on this text.

## Papers

### 1. `2503.17261`

- **assigned_domain:** medical_imaging
- **title:**

> Cross-Modal Interactive Perception Network with Mamba for Lung Tumor Segmentation in PET-CT Images

- **abstract:**

> Lung cancer is a leading cause of cancer-related deaths globally. PET-CT is crucial for imaging lung tumors, providing essential metabolic and anatomical information, while it faces challenges such as poor image quality, motion artifacts, and complex tumor morphology. Deep learning-based models are expected to address these problems, however, existing small-scale and private datasets limit significant performance improvements for these methods. Hence, we introduce a large-scale PET-CT lung tumor segmentation dataset, termed PCLT20K, which comprises 21,930 pairs of PET-CT images from 605 patients. Furthermore, we propose a cross-modal interactive perception network with Mamba (CIPA) for lung tumor segmentation in PET-CT images. Specifically, we design a channel-wise rectification module (CRM) that implements a channel state space block across multi-modal features to learn correlated representations and helps filter out modality-specific noise. A dynamic cross-modality interaction module (DCIM) is designed to effectively integrate position and context information, which employs PET images to learn regional position information and serves as a bridge to assist in modeling the relationships between local features of CT images. Extensive experiments on a comprehensive benchmark demonstrate the effectiveness of our CIPA compared to the current state-of-the-art segmentation methods. We hope our research can provide more exploration opportunities for medical image segmentation. The dataset and code are available at https://github.com/mj129/CIPA.

- **manual_class:** 

### 2. `2305.17937`

- **assigned_domain:** medical_imaging
- **title:**

> Attention Mechanisms in Medical Image Segmentation: A Survey

- **abstract:**

> Medical image segmentation plays an important role in computer-aided diagnosis. Attention mechanisms that distinguish important parts from irrelevant parts have been widely used in medical image segmentation tasks. This paper systematically reviews the basic principles of attention mechanisms and their applications in medical image segmentation. First, we review the basic concepts of attention mechanism and formulation. Second, we surveyed over 300 articles related to medical image segmentation, and divided them into two groups based on their attention mechanisms, non-Transformer attention and Transformer attention. In each group, we deeply analyze the attention mechanisms from three aspects based on the current literature work, i.e., the principle of the mechanism (what to use), implementation methods (how to use), and application tasks (where to use). We also thoroughly analyzed the advantages and limitations of their applications to different tasks. Finally, we summarize the current state of research and shortcomings in the field, and discuss the potential challenges in the future, including task specificity, robustness, standard evaluation, etc. We hope that this review can showcase the overall research context of traditional and Transformer attention methods, provide a clear reference for subsequent research, and inspire more advanced attention research, not only in medical image segmentation, but also in other image analysis scenarios.

- **manual_class:** 

### 3. `2404.04550`

- **assigned_domain:** medical_imaging
- **title:**

> NPB-REC: A Non-parametric Bayesian Deep-learning Approach for Undersampled MRI Reconstruction with Uncertainty Estimation

- **abstract:**

> The ability to reconstruct high-quality images from undersampled MRI data is vital in improving MRI temporal resolution and reducing acquisition times. Deep learning methods have been proposed for this task, but the lack of verified methods to quantify the uncertainty in the reconstructed images hampered clinical applicability. We introduce "NPB-REC", a non-parametric fully Bayesian framework, for MRI reconstruction from undersampled data with uncertainty estimation. We use Stochastic Gradient Langevin Dynamics during training to characterize the posterior distribution of the network parameters. This enables us to both improve the quality of the reconstructed images and quantify the uncertainty in the reconstructed images. We demonstrate the efficacy of our approach on a multi-coil MRI dataset from the fastMRI challenge and compare it to the baseline End-to-End Variational Network (E2E-VarNet). Our approach outperforms the baseline in terms of reconstruction accuracy by means of PSNR and SSIM ($34.55$, $0.908$ vs. $33.08$, $0.897$, $p<0.01$, acceleration rate $R=8$) and provides uncertainty measures that correlate better with the reconstruction error (Pearson correlation, $R=0.94$ vs. $R=0.91$). Additionally, our approach exhibits better generalization capabilities against anatomical distribution shifts (PSNR and SSIM of $32.38$, $0.849$ vs. $31.63$, $0.836$, $p<0.01$, training on brain data, inference on knee data, acceleration rate $R=8$). NPB-REC has the potential to facilitate the safe utilization of deep learning-based methods for MRI reconstruction from undersampled data. Code and trained models are available at \url{https://github.com/samahkh/NPB-REC}.

- **manual_class:** 

### 4. `2509.24739`

- **assigned_domain:** medical_imaging
- **title:**

> Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation

- **abstract:**

> Vision-Language Foundation Models (VLMs), trained on large-scale multimodal datasets, have driven significant advances in Artificial Intelligence (AI) by enabling rich cross-modal reasoning. Despite their success in general domains, applying these models to medical imaging remains challenging due to the limited availability of diverse imaging modalities and multilingual clinical data. Most existing medical VLMs are trained on a subset of imaging modalities and focus primarily on high-resource languages, thus limiting their generalizability and clinical utility. To address these limitations, we introduce a novel Vietnamese-language multimodal medical dataset consisting of 2,757 whole-body PET/CT volumes from independent patients and their corresponding full-length clinical reports. This dataset is designed to fill two pressing gaps in medical AI development: (1) the lack of PET/CT imaging data in existing VLMs training corpora, which hinders the development of models capable of handling functional imaging tasks; and (2) the underrepresentation of low-resource languages, particularly the Vietnamese language, in medical vision-language research. To the best of our knowledge, this is the first dataset to provide comprehensive PET/CT-report pairs in Vietnamese. We further introduce a training framework to enhance VLMs' learning, including data augmentation and expert-validated test sets. We conduct comprehensive experiments benchmarking state-of-the-art VLMs on downstream tasks. The experimental results show that incorporating our dataset significantly improves the performance of existing VLMs. We believe this dataset and benchmark will serve as a pivotal step in advancing the development of more robust VLMs for medical imaging, especially for low-resource languages and clinical use in Vietnamese healthcare. The source code is available at https://github.com/AIoT-Lab-BKAI/ViPET-ReportGen.

- **manual_class:** 

### 5. `2503.20653`

- **assigned_domain:** medical_imaging
- **title:**

> UWarp: A Whole Slide Image Registration Pipeline to Characterize Scanner-Induced Local Domain Shift

- **abstract:**

> Histopathology slide digitization introduces scanner-induced domain shift that can significantly impact computational pathology models based on deep learning methods. In the state-of-the-art, this shift is often characterized at a broad scale (slide-level or dataset-level) but not patch-level, which limits our comprehension of the impact of localized tissue characteristics on the accuracy of the deep learning models. To address this challenge, we present a domain shift analysis framework based on UWarp, a novel registration tool designed to accurately align histological slides scanned under varying conditions. UWarp employs a hierarchical registration approach, combining global affine transformations with fine-grained local corrections to achieve robust tissue patch alignment. We evaluate UWarp using two private datasets, CypathLung and BosomShieldBreast, containing whole slide images scanned by multiple devices. Our experiments demonstrate that UWarp outperforms existing open-source registration methods, achieving a median target registration error (TRE) of less than 4 pixels (<1 micrometer at 40x magnification) while significantly reducing computational time. Additionally, we apply UWarp to characterize scanner-induced local domain shift in the predictions of Breast-NEOprAIdict, a deep learning model for breast cancer pathological response prediction. We find that prediction variability is strongly correlated with tissue density on a given patch. Our findings highlight the importance of localized domain shift analysis and suggest that UWarp can serve as a valuable tool for improving model robustness and domain adaptation strategies in computational pathology.

- **manual_class:** 

### 6. `2311.12885`

- **assigned_domain:** medical_imaging
- **title:**

> Long-MIL: Scaling Long Contextual Multiple Instance Learning for Histopathology Whole Slide Image Analysis

- **abstract:**

> Histopathology image analysis is the golden standard of clinical diagnosis for Cancers. In doctors daily routine and computer-aided diagnosis, the Whole Slide Image (WSI) of histopathology tissue is used for analysis. Because of the extremely large scale of resolution, previous methods generally divide the WSI into a large number of patches, then aggregate all patches within a WSI by Multi-Instance Learning (MIL) to make the slide-level prediction when developing computer-aided diagnosis tools. However, most previous WSI-MIL models using global-attention without pairwise interaction and any positional information, or self-attention with absolute position embedding can not well handle shape varying large WSIs, e.g. testing WSIs after model deployment may be larger than training WSIs, since the model development set is always limited due to the difficulty of histopathology WSIs collection. To deal with the problem, in this paper, we propose to amend position embedding for shape varying long-contextual WSI by introducing Linear Bias into Attention, and adapt it from 1-d long sequence into 2-d long-contextual WSI which helps model extrapolate position embedding to unseen or under-fitted positions. We further utilize Flash-Attention module to tackle the computational complexity of Transformer, which also keep full self-attention performance compared to previous attention approximation work. Our method, Long-contextual MIL (Long-MIL) are evaluated on extensive experiments including 4 dataset including WSI classification and survival prediction tasks to validate the superiority on shape varying WSIs. The source code will be open-accessed soon.

- **manual_class:** 

### 7. `2507.00185`

- **assigned_domain:** medical_imaging
- **title:**

> Multimodal, Multi-Disease Medical Imaging Foundation Model (MerMED-FM)

- **abstract:**

> Current artificial intelligence models for medical imaging are predominantly single modality and single disease. Attempts to create multimodal and multi-disease models have resulted in inconsistent clinical accuracy. Furthermore, training these models typically requires large, labour-intensive, well-labelled datasets. We developed MerMED-FM, a state-of-the-art multimodal, multi-specialty foundation model trained using self-supervised learning and a memory module. MerMED-FM was trained on 3.3 million medical images from over ten specialties and seven modalities, including computed tomography (CT), chest X-rays (CXR), ultrasound (US), pathology patches, color fundus photography (CFP), optical coherence tomography (OCT) and dermatology images. MerMED-FM was evaluated across multiple diseases and compared against existing foundational models. Strong performance was achieved across all modalities, with AUROCs of 0.988 (OCT); 0.982 (pathology); 0.951 (US); 0.943 (CT); 0.931 (skin); 0.894 (CFP); 0.858 (CXR). MerMED-FM has the potential to be a highly adaptable, versatile, cross-specialty foundation model that enables robust medical imaging interpretation across diverse medical disciplines.

- **manual_class:** 

### 8. `2312.10892`

- **assigned_domain:** medical_imaging
- **title:**

> Deep Learning-based MRI Reconstruction with Artificial Fourier Transform Network (AFTNet)

- **abstract:**

> Deep complex-valued neural networks (CVNNs) provide a powerful way to leverage complex number operations and representations and have succeeded in several phase-based applications. However, previous networks have not fully explored the impact of complex-valued networks in the frequency domain. Here, we introduce a unified complex-valued deep learning framework-Artificial Fourier Transform Network (AFTNet)-which combines domain-manifold learning and CVNNs. AFTNet can be readily used to solve image inverse problems in domain transformation, especially for accelerated magnetic resonance imaging (MRI) reconstruction and other applications. While conventional methods typically utilize magnitude images or treat the real and imaginary components of k-space data as separate channels, our approach directly processes raw k-space data in the frequency domain, utilizing complex-valued operations. This allows for a mapping between the frequency (k-space) and image domain to be determined through cross-domain learning. We show that AFTNet achieves superior accelerated MRI reconstruction compared to existing approaches. Furthermore, our approach can be applied to various tasks, such as denoised magnetic resonance spectroscopy (MRS) reconstruction and datasets with various contrasts. The AFTNet presented here is a valuable preprocessing component for different preclinical studies and provides an innovative alternative for solving inverse problems in imaging and spectroscopy. The code is available at: https://github.com/yanting-yang/AFT-Net.

- **manual_class:** 

### 9. `2505.06527`

- **assigned_domain:** medical_imaging
- **title:**

> Improving Generalization of Medical Image Registration Foundation Model

- **abstract:**

> Deformable registration is a fundamental task in medical image processing, aiming to achieve precise alignment by establishing nonlinear correspondences between images. Traditional methods offer good adaptability and interpretability but are limited by computational efficiency. Although deep learning approaches have significantly improved registration speed and accuracy, they often lack flexibility and generalizability across different datasets and tasks. In recent years, foundation models have emerged as a promising direction, leveraging large and diverse datasets to learn universal features and transformation patterns for image registration, thus demonstrating strong cross-task transferability. However, these models still face challenges in generalization and robustness when encountering novel anatomical structures, varying imaging conditions, or unseen modalities. To address these limitations, this paper incorporates Sharpness-Aware Minimization (SAM) into foundation models to enhance their generalization and robustness in medical image registration. By optimizing the flatness of the loss landscape, SAM improves model stability across diverse data distributions and strengthens its ability to handle complex clinical scenarios. Experimental results show that foundation models integrated with SAM achieve significant improvements in cross-dataset registration performance, offering new insights for the advancement of medical image registration technology. Our code is available at https://github.com/Promise13/fm_sam}{https://github.com/Promise13/fm\_sam.

- **manual_class:** 

### 10. `2507.01291`

- **assigned_domain:** medical_imaging
- **title:**

> PanTS: The Pancreatic Tumor Segmentation Dataset

- **abstract:**

> PanTS is a large-scale, multi-institutional dataset curated to advance research in pancreatic CT analysis. It contains 36,390 CT scans from 145 medical centers, with expert-validated, voxel-wise annotations of over 993,000 anatomical structures, covering pancreatic tumors, pancreas head, body, and tail, and 24 surrounding anatomical structures such as vascular/skeletal structures and abdominal/thoracic organs. Each scan includes metadata such as patient age, sex, diagnosis, contrast phase, in-plane spacing, slice thickness, etc. AI models trained on PanTS achieve significantly better performance in pancreatic tumor detection, localization, and segmentation compared to those trained on existing public datasets. Our analysis indicates that these gains are directly attributable to the 16x larger-scale tumor annotations and indirectly supported by the 24 additional surrounding anatomical structures. As the largest and most comprehensive resource of its kind, PanTS offers a new benchmark for developing and evaluating AI models in pancreatic CT analysis.

- **manual_class:** 

### 11. `2506.07044`

- **assigned_domain:** medical_imaging
- **title:**

> Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning

- **abstract:**

> Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in understanding common visual elements, largely due to their large-scale datasets and advanced training strategies. However, their effectiveness in medical applications remains limited due to the inherent discrepancies between data and tasks in medical scenarios and those in the general domain. Concretely, existing medical MLLMs face the following critical limitations: (1) limited coverage of medical knowledge beyond imaging, (2) heightened susceptibility to hallucinations due to suboptimal data curation processes, (3) lack of reasoning capabilities tailored for complex medical scenarios. To address these challenges, we first propose a comprehensive data curation procedure that (1) efficiently acquires rich medical knowledge data not only from medical imaging but also from extensive medical texts and general-domain data; and (2) synthesizes accurate medical captions, visual question answering (VQA), and reasoning samples. As a result, we build a multimodal dataset enriched with extensive medical knowledge. Building on the curated data, we introduce our medical-specialized MLLM: Lingshu. Lingshu undergoes multi-stage training to embed medical expertise and enhance its task-solving capabilities progressively. Besides, we preliminarily explore the potential of applying reinforcement learning with verifiable rewards paradigm to enhance Lingshu's medical reasoning ability. Additionally, we develop MedEvalKit, a unified evaluation framework that consolidates leading multimodal and textual medical benchmarks for standardized, fair, and efficient model assessment. We evaluate the performance of Lingshu on three fundamental medical tasks, multimodal QA, text-based QA, and medical report generation. The results show that Lingshu consistently outperforms the existing open-source multimodal models on most tasks ...

- **manual_class:** 

### 12. `2303.08446`

- **assigned_domain:** medical_imaging
- **title:**

> Task-specific Fine-tuning via Variational Information Bottleneck for Weakly-supervised Pathology Whole Slide Image Classification

- **abstract:**

> While Multiple Instance Learning (MIL) has shown promising results in digital Pathology Whole Slide Image (WSI) classification, such a paradigm still faces performance and generalization problems due to challenges in high computational costs on Gigapixel WSIs and limited sample size for model training. To deal with the computation problem, most MIL methods utilize a frozen pretrained model from ImageNet to obtain representations first. This process may lose essential information owing to the large domain gap and hinder the generalization of model due to the lack of image-level training-time augmentations. Though Self-supervised Learning (SSL) proposes viable representation learning schemes, the improvement of the downstream task still needs to be further explored in the conversion from the task-agnostic features of SSL to the task-specifics under the partial label supervised learning. To alleviate the dilemma of computation cost and performance, we propose an efficient WSI fine-tuning framework motivated by the Information Bottleneck theory. The theory enables the framework to find the minimal sufficient statistics of WSI, thus supporting us to fine-tune the backbone into a task-specific representation only depending on WSI-level weak labels. The WSI-MIL problem is further analyzed to theoretically deduce our fine-tuning method. Our framework is evaluated on five pathology WSI datasets on various WSI heads. The experimental results of our fine-tuned representations show significant improvements in both accuracy and generalization compared with previous works. Source code will be available at https://github.com/invoker-LL/WSI-finetuning.

- **manual_class:** 

### 13. `2307.01704`

- **assigned_domain:** medical_imaging
- **title:**

> Graph-Ensemble Learning Model for Multi-label Skin Lesion Classification using Dermoscopy and Clinical Images

- **abstract:**

> Many skin lesion analysis (SLA) methods recently focused on developing a multi-modal-based multi-label classification method due to two factors. The first is multi-modal data, i.e., clinical and dermoscopy images, which can provide complementary information to obtain more accurate results than single-modal data. The second one is that multi-label classification, i.e., seven-point checklist (SPC) criteria as an auxiliary classification task can not only boost the diagnostic accuracy of melanoma in the deep learning (DL) pipeline but also provide more useful functions to the clinical doctor as it is commonly used in clinical dermatologist's diagnosis. However, most methods only focus on designing a better module for multi-modal data fusion; few methods explore utilizing the label correlation between SPC and skin disease for performance improvement. This study fills the gap that introduces a Graph Convolution Network (GCN) to exploit prior co-occurrence between each category as a correlation matrix into the DL model for the multi-label classification. However, directly applying GCN degraded the performances in our experiments; we attribute this to the weak generalization ability of GCN in the scenario of insufficient statistical samples of medical data. We tackle this issue by proposing a Graph-Ensemble Learning Model (GELN) that views the prediction from GCN as complementary information of the predictions from the fusion model and adaptively fuses them by a weighted averaging scheme, which can utilize the valuable information from GCN while avoiding its negative influences as much as possible. To evaluate our method, we conduct experiments on public datasets. The results illustrate that our GELN can consistently improve the classification performance on different datasets and that the proposed method can achieve state-of-the-art performance in SPC and diagnosis classification.

- **manual_class:** 

### 14. `2301.10847`

- **assigned_domain:** medical_imaging
- **title:**

> Enhancing Medical Image Segmentation with TransCeption: A Multi-Scale Feature Fusion Approach

- **abstract:**

> While CNN-based methods have been the cornerstone of medical image segmentation due to their promising performance and robustness, they suffer from limitations in capturing long-range dependencies. Transformer-based approaches are currently prevailing since they enlarge the reception field to model global contextual correlation. To further extract rich representations, some extensions of the U-Net employ multi-scale feature extraction and fusion modules and obtain improved performance. Inspired by this idea, we propose TransCeption for medical image segmentation, a pure transformer-based U-shape network featured by incorporating the inception-like module into the encoder and adopting a contextual bridge for better feature fusion. The design proposed in this work is based on three core principles: (1) The patch merging module in the encoder is redesigned with ResInception Patch Merging (RIPM). Multi-branch transformer (MB transformer) adopts the same number of branches as the outputs of RIPM. Combining the two modules enables the model to capture a multi-scale representation within a single stage. (2) We construct an Intra-stage Feature Fusion (IFF) module following the MB transformer to enhance the aggregation of feature maps from all the branches and particularly focus on the interaction between the different channels of all the scales. (3) In contrast to a bridge that only contains token-wise self-attention, we propose a Dual Transformer Bridge that also includes channel-wise self-attention to exploit correlations between scales at different stages from a dual perspective. Extensive experiments on multi-organ and skin lesion segmentation tasks present the superior performance of TransCeption compared to previous work. The code is publicly available at \url{https://github.com/mindflow-institue/TransCeption}.

- **manual_class:** 

### 15. `2401.12771`

- **assigned_domain:** medical_imaging
- **title:**

> Deep Learning-based Intraoperative MRI Reconstruction

- **abstract:**

> Purpose: To evaluate the quality of deep learning reconstruction for prospectively accelerated intraoperative magnetic resonance imaging (iMRI) during resective brain tumor surgery. Materials and Methods: Accelerated iMRI was performed during brain surgery using dual surface coils positioned around the area of resection. A deep learning (DL) model was trained on the fastMRI neuro dataset to mimic the data from the iMRI protocol. Evaluation was performed on imaging material from 40 patients imaged between 01.11.2021 - 01.06.2023 that underwent iMRI during tumor resection surgery. A comparative analysis was conducted between the conventional compressed sense (CS) method and the trained DL reconstruction method. Blinded evaluation of multiple image quality metrics was performed by two working neuro-radiologists and a working neurosurgeon on a 1 to 5 Likert scale (1=non diagnostic, 2=poor, 3=acceptable, 4=good, 5=excellent), and the favored reconstruction variant. Results: The DL reconstruction was strongly favored or favored over the CS reconstruction for 33/40, 39/40, and 8/40 of cases for reader 1, 2, and 3, respectively. Two of three readers consistently assigned higher ratings for the DL reconstructions, and the DL reconstructions had a higher score than their respective CS counterparts for 72%, 72%, and 14% of the cases for reader 1, 2, and 3, respectively. Still, the DL reconstructions exhibited shortcomings such as a striping artifact and reduced signal. Conclusion: DL shows promise to allow for high-quality reconstructions of intraoperative MRI with equal to or improved perceived spatial resolution, signal-to-noise ratio, diagnostic confidence, diagnostic conspicuity, and spatial resolution compared to compressed sense.

- **manual_class:** 

### 16. `2307.04189`

- **assigned_domain:** medical_imaging
- **title:**

> Histopathology Whole Slide Image Analysis with Heterogeneous Graph Representation Learning

- **abstract:**

> Graph-based methods have been extensively applied to whole-slide histopathology image (WSI) analysis due to the advantage of modeling the spatial relationships among different entities. However, most of the existing methods focus on modeling WSIs with homogeneous graphs (e.g., with homogeneous node type). Despite their successes, these works are incapable of mining the complex structural relations between biological entities (e.g., the diverse interaction among different cell types) in the WSI. We propose a novel heterogeneous graph-based framework to leverage the inter-relationships among different types of nuclei for WSI analysis. Specifically, we formulate the WSI as a heterogeneous graph with "nucleus-type" attribute to each node and a semantic similarity attribute to each edge. We then present a new heterogeneous-graph edge attribute transformer (HEAT) to take advantage of the edge and node heterogeneity during massage aggregating. Further, we design a new pseudo-label-based semantic-consistent pooling mechanism to obtain graph-level features, which can mitigate the over-parameterization issue of conventional cluster-based pooling. Additionally, observing the limitations of existing association-based localization methods, we propose a causal-driven approach attributing the contribution of each node to improve the interpretability of our framework. Extensive experiments on three public TCGA benchmark datasets demonstrate that our framework outperforms the state-of-the-art methods with considerable margins on various tasks. Our codes are available at https://github.com/HKU-MedAI/WSI-HGNN.

- **manual_class:** 

### 17. `2510.22990`

- **assigned_domain:** medical_imaging
- **title:**

> USF-MAE: Ultrasound Self-Supervised Foundation Model with Masked Autoencoding

- **abstract:**

> Ultrasound imaging is one of the most widely used diagnostic modalities, offering real-time, radiation-free assessment across diverse clinical domains. However, interpretation of ultrasound images remains challenging due to high noise levels, operator dependence, and limited field of view, resulting in substantial inter-observer variability. Current Deep Learning approaches are hindered by the scarcity of large labeled datasets and the domain gap between general and sonographic images, which limits the transferability of models pretrained on non-medical data. To address these challenges, we introduce the Ultrasound Self-Supervised Foundation Model with Masked Autoencoding (USF-MAE), the first large-scale self-supervised MAE framework pretrained exclusively on ultrasound data. The model was pre-trained on 370,000 2D and 3D ultrasound images curated from 46 open-source datasets, collectively termed OpenUS-46, spanning over twenty anatomical regions. This curated dataset has been made publicly available to facilitate further research and reproducibility. Using a Vision Transformer encoder-decoder architecture, USF-MAE reconstructs masked image patches, enabling it to learn rich, modality-specific representations directly from unlabeled data. The pretrained encoder was fine-tuned on three public downstream classification benchmarks: BUS-BRA (breast cancer), MMOTU-2D (ovarian tumors), and GIST514-DB (gastrointestinal stromal tumors). Across all tasks, USF-MAE consistently outperformed conventional CNN and ViT baselines, achieving F1-scores of 81.6%, 79.6%, and 82.4%, respectively. Despite not using labels during pretraining, USF-MAE approached the performance of the supervised foundation model UltraSam on breast cancer classification and surpassed it on the other tasks, demonstrating strong cross-anatomical generalization.

- **manual_class:** 

### 18. `2402.05373`

- **assigned_domain:** medical_imaging
- **title:**

> Unleashing the Infinity Power of Geometry: A Novel Geometry-Aware Transformer (GOAT) for Whole Slide Histopathology Image Analysis

- **abstract:**

> The histopathology analysis is of great significance for the diagnosis and prognosis of cancers, however, it has great challenges due to the enormous heterogeneity of gigapixel whole slide images (WSIs) and the intricate representation of pathological features. However, recent methods have not adequately exploited geometrical representation in WSIs which is significant in disease diagnosis. Therefore, we proposed a novel weakly-supervised framework, Geometry-Aware Transformer (GOAT), in which we urge the model to pay attention to the geometric characteristics within the tumor microenvironment which often serve as potent indicators. In addition, a context-aware attention mechanism is designed to extract and enhance the morphological features within WSIs.

- **manual_class:** 

### 19. `2507.19004`

- **assigned_domain:** medical_imaging
- **title:**

> MedIQA: A Scalable Foundation Model for Prompt-Driven Medical Image Quality Assessment

- **abstract:**

> Rapid advances in medical imaging technology underscore the critical need for precise and automated image quality assessment (IQA) to ensure diagnostic accuracy. Existing medical IQA methods, however, struggle to generalize across diverse modalities and clinical scenarios. In response, we introduce MedIQA, the first comprehensive foundation model for medical IQA, designed to handle variability in image dimensions, modalities, anatomical regions, and types. We developed a large-scale multi-modality dataset with plentiful manually annotated quality scores to support this. Our model integrates a salient slice assessment module to focus on diagnostically relevant regions feature retrieval and employs an automatic prompt strategy that aligns upstream physical parameter pre-training with downstream expert annotation fine-tuning. Extensive experiments demonstrate that MedIQA significantly outperforms baselines in multiple downstream tasks, establishing a scalable framework for medical IQA and advancing diagnostic workflows and clinical decision-making.

- **manual_class:** 

### 20. `2509.10620`

- **assigned_domain:** medical_imaging
- **title:**

> Building a General SimCLR Self-Supervised Foundation Model Across Neurological Diseases to Advance 3D Brain MRI Diagnoses

- **abstract:**

> 3D structural Magnetic Resonance Imaging (MRI) brain scans are commonly acquired in clinical settings to monitor a wide range of neurological conditions, including neurodegenerative disorders and stroke. While deep learning models have shown promising results analyzing 3D MRI across a number of brain imaging tasks, most are highly tailored for specific tasks with limited labeled data, and are not able to generalize across tasks and/or populations. The development of self-supervised learning (SSL) has enabled the creation of large medical foundation models that leverage diverse, unlabeled datasets ranging from healthy to diseased data, showing significant success in 2D medical imaging applications. However, even the very few foundation models for 3D brain MRI that have been developed remain limited in resolution, scope, or accessibility. In this work, we present a general, high-resolution SimCLR-based SSL foundation model for 3D brain structural MRI, pre-trained on 18,759 patients (44,958 scans) from 11 publicly available datasets spanning diverse neurological diseases. We compare our model to Masked Autoencoders (MAE), as well as two supervised baselines, on four diverse downstream prediction tasks in both in-distribution and out-of-distribution settings. Our fine-tuned SimCLR model outperforms all other models across all tasks. Notably, our model still achieves superior performance when fine-tuned using only 20% of labeled training samples for predicting Alzheimer's disease. We use publicly available code and data, and release our trained model at https://github.com/emilykaczmarek/3D-Neuro-SimCLR, contributing a broadly applicable and accessible foundation model for clinical brain MRI analysis.

- **manual_class:** 

### 21. `2312.00634`

- **assigned_domain:** medical_imaging
- **title:**

> A Recent Survey of Vision Transformers for Medical Image Segmentation

- **abstract:**

> Medical image segmentation plays a crucial role in various healthcare applications, enabling accurate diagnosis, treatment planning, and disease monitoring. Traditionally, convolutional neural networks (CNNs) dominated this domain, excelling at local feature extraction. However, their limitations in capturing long-range dependencies across image regions pose challenges for segmenting complex, interconnected structures often encountered in medical data. In recent years, Vision Transformers (ViTs) have emerged as a promising technique for addressing the challenges in medical image segmentation. Their multi-scale attention mechanism enables effective modeling of long-range dependencies between distant structures, crucial for segmenting organs or lesions spanning the image. Additionally, ViTs' ability to discern subtle pattern heterogeneity allows for the precise delineation of intricate boundaries and edges, a critical aspect of accurate medical image segmentation. However, they do lack image-related inductive bias and translational invariance, potentially impacting their performance. Recently, researchers have come up with various ViT-based approaches that incorporate CNNs in their architectures, known as Hybrid Vision Transformers (HVTs) to capture local correlation in addition to the global information in the images. This survey paper provides a detailed review of the recent advancements in ViTs and HVTs for medical image segmentation. Along with the categorization of ViT and HVT-based medical image segmentation approaches, we also present a detailed overview of their real-time applications in several medical image modalities. This survey may serve as a valuable resource for researchers, healthcare practitioners, and students in understanding the state-of-the-art approaches for ViT-based medical image segmentation.

- **manual_class:** 

### 22. `2310.07781`

- **assigned_domain:** medical_imaging
- **title:**

> 3D TransUNet: Advancing Medical Image Segmentation through Vision Transformers

- **abstract:**

> Medical image segmentation plays a crucial role in advancing healthcare systems for disease diagnosis and treatment planning. The u-shaped architecture, popularly known as U-Net, has proven highly successful for various medical image segmentation tasks. However, U-Net's convolution-based operations inherently limit its ability to model long-range dependencies effectively. To address these limitations, researchers have turned to Transformers, renowned for their global self-attention mechanisms, as alternative architectures. One popular network is our previous TransUNet, which leverages Transformers' self-attention to complement U-Net's localized information with the global context. In this paper, we extend the 2D TransUNet architecture to a 3D network by building upon the state-of-the-art nnU-Net architecture, and fully exploring Transformers' potential in both the encoder and decoder design. We introduce two key components: 1) A Transformer encoder that tokenizes image patches from a convolution neural network (CNN) feature map, enabling the extraction of global contexts, and 2) A Transformer decoder that adaptively refines candidate regions by utilizing cross-attention between candidate proposals and U-Net features. Our investigations reveal that different medical tasks benefit from distinct architectural designs. The Transformer encoder excels in multi-organ segmentation, where the relationship among organs is crucial. On the other hand, the Transformer decoder proves more beneficial for dealing with small and challenging segmented targets such as tumor segmentation. Extensive experiments showcase the significant potential of integrating a Transformer-based encoder and decoder into the u-shaped medical image segmentation architecture. TransUNet outperforms competitors in various medical applications.

- **manual_class:** 

### 23. `2411.18101`

- **assigned_domain:** medical_imaging
- **title:**

> Aligning Knowledge Concepts to Whole Slide Images for Precise Histopathology Image Analysis

- **abstract:**

> Due to the large size and lack of fine-grained annotation, Whole Slide Images (WSIs) analysis is commonly approached as a Multiple Instance Learning (MIL) problem. However, previous studies only learn from training data, posing a stark contrast to how human clinicians teach each other and reason about histopathologic entities and factors. Here we present a novel knowledge concept-based MIL framework, named ConcepPath to fill this gap. Specifically, ConcepPath utilizes GPT-4 to induce reliable diseasespecific human expert concepts from medical literature, and incorporate them with a group of purely learnable concepts to extract complementary knowledge from training data. In ConcepPath, WSIs are aligned to these linguistic knowledge concepts by utilizing pathology vision-language model as the basic building component. In the application of lung cancer subtyping, breast cancer HER2 scoring, and gastric cancer immunotherapy-sensitive subtyping task, ConcepPath significantly outperformed previous SOTA methods which lack the guidance of human expert knowledge.

- **manual_class:** 

### 24. `2505.06646`

- **assigned_domain:** medical_imaging
- **title:**

> Reproducing and Improving CheXNet: Deep Learning for Chest X-ray Disease Classification

- **abstract:**

> Deep learning for radiologic image analysis is a rapidly growing field in biomedical research and is likely to become a standard practice in modern medicine. On the publicly available NIH ChestX-ray14 dataset, containing X-ray images that are classified by the presence or absence of 14 different diseases, we reproduced an algorithm known as CheXNet, as well as explored other algorithms that outperform CheXNet's baseline metrics. Model performance was primarily evaluated using the F1 score and AUC-ROC, both of which are critical metrics for imbalanced, multi-label classification tasks in medical imaging. The best model achieved an average AUC-ROC score of 0.85 and an average F1 score of 0.39 across all 14 disease classifications present in the dataset.

- **manual_class:** 

### 25. `2301.02613`

- **assigned_domain:** medical_imaging
- **title:**

> Learning Deep MRI Reconstruction Models from Scratch in Low-Data Regimes

- **abstract:**

> Magnetic resonance imaging (MRI) is an essential diagnostic tool that suffers from prolonged scan times. Reconstruction methods can alleviate this limitation by recovering clinically usable images from accelerated acquisitions. In particular, learning-based methods promise performance leaps by employing deep neural networks as data-driven priors. A powerful approach uses scan-specific (SS) priors that leverage information regarding the underlying physical signal model for reconstruction. SS priors are learned on each individual test scan without the need for a training dataset, albeit they suffer from computationally burdening inference with nonlinear networks. An alternative approach uses scan-general (SG) priors that instead leverage information regarding the latent features of MRI images for reconstruction. SG priors are frozen at test time for efficiency, albeit they require learning from a large training dataset. Here, we introduce a novel parallel-stream fusion model (PSFNet) that synergistically fuses SS and SG priors for performant MRI reconstruction in low-data regimes, while maintaining competitive inference times to SG methods. PSFNet implements its SG prior based on a nonlinear network, yet it forms its SS prior based on a linear network to maintain efficiency. A pervasive framework for combining multiple priors in MRI reconstruction is algorithmic unrolling that uses serially alternated projections, causing error propagation under low-data regimes. To alleviate error propagation, PSFNet combines its SS and SG priors via a novel parallel-stream architecture with learnable fusion parameters. Demonstrations are performed on multi-coil brain MRI for varying amounts of training data. PSFNet outperforms SG methods in low-data regimes, and surpasses SS methods with few tens of training samples.

- **manual_class:** 

### 26. `2303.10323`

- **assigned_domain:** medical_imaging
- **title:**

> Dynamic Graph Enhanced Contrastive Learning for Chest X-ray Report Generation

- **abstract:**

> Automatic radiology reporting has great clinical potential to relieve radiologists from heavy workloads and improve diagnosis interpretation. Recently, researchers have enhanced data-driven neural networks with medical knowledge graphs to eliminate the severe visual and textual bias in this task. The structures of such graphs are exploited by using the clinical dependencies formed by the disease topic tags via general knowledge and usually do not update during the training process. Consequently, the fixed graphs can not guarantee the most appropriate scope of knowledge and limit the effectiveness. To address the limitation, we propose a knowledge graph with Dynamic structure and nodes to facilitate medical report generation with Contrastive Learning, named DCL. In detail, the fundamental structure of our graph is pre-constructed from general knowledge. Then we explore specific knowledge extracted from the retrieved reports to add additional nodes or redefine their relations in a bottom-up manner. Each image feature is integrated with its very own updated graph before being fed into the decoder module for report generation. Finally, this paper introduces Image-Report Contrastive and Image-Report Matching losses to better represent visual features and textual information. Evaluated on IU-Xray and MIMIC-CXR datasets, our DCL outperforms previous state-of-the-art models on these two benchmarks.

- **manual_class:** 

### 27. `2503.19005`

- **assigned_domain:** medical_imaging
- **title:**

> Foundation Model for Whole-Heart Segmentation: Leveraging Student-Teacher Learning in Multi-Modal Medical Imaging

- **abstract:**

> Whole-heart segmentation from CT and MRI scans is crucial for cardiovascular disease analysis, yet existing methods struggle with modality-specific biases and the need for extensive labeled datasets. To address these challenges, we propose a foundation model for whole-heart segmentation using a self-supervised learning (SSL) framework based on a student-teacher architecture. Our model is pretrained on a large, unlabeled dataset of CT and MRI scans, leveraging the xLSTM backbone to capture long-range spatial dependencies and complex anatomical structures in 3D medical images. By incorporating multi-modal pretraining, our approach ensures strong generalization across both CT and MRI modalities, mitigating modality-specific variations and improving segmentation accuracy in diverse clinical settings. The use of large-scale unlabeled data significantly reduces the dependency on manual annotations, enabling robust performance even with limited labeled data. We further introduce an xLSTM-UNet-based architecture for downstream whole-heart segmentation tasks, demonstrating its effectiveness on few-label CT and MRI datasets. Our results validate the robustness and adaptability of the proposed model, highlighting its potential for advancing automated whole-heart segmentation in medical imaging.

- **manual_class:** 

### 28. `2401.17593`

- **assigned_domain:** medical_imaging
- **title:**

> Head and Neck Tumor Segmentation from [18F]F-FDG PET/CT Images Based on 3D Diffusion Model

- **abstract:**

> Head and neck (H&N) cancers are among the most prevalent types of cancer worldwide, and [18F]F-FDG PET/CT is widely used for H&N cancer management. Recently, the diffusion model has demonstrated remarkable performance in various image-generation tasks. In this work, we proposed a 3D diffusion model to accurately perform H&N tumor segmentation from 3D PET and CT volumes. The 3D diffusion model was developed considering the 3D nature of PET and CT images acquired. During the reverse process, the model utilized a 3D UNet structure and took the concatenation of PET, CT, and Gaussian noise volumes as the network input to generate the tumor mask. Experiments based on the HECKTOR challenge dataset were conducted to evaluate the effectiveness of the proposed diffusion model. Several state-of-the-art techniques based on U-Net and Transformer structures were adopted as the reference methods. Benefits of employing both PET and CT as the network input as well as further extending the diffusion model from 2D to 3D were investigated based on various quantitative metrics and the uncertainty maps generated. Results showed that the proposed 3D diffusion model could generate more accurate segmentation results compared with other methods. Compared to the diffusion model in 2D format, the proposed 3D model yielded superior results. Our experiments also highlighted the advantage of utilizing dual-modality PET and CT data over only single-modality data for H&N tumor segmentation.

- **manual_class:** 

### 29. `2305.06244`

- **assigned_domain:** medical_imaging
- **title:**

> Explainable Knowledge Distillation for On-device Chest X-Ray Classification

- **abstract:**

> Automated multi-label chest X-rays (CXR) image classification has achieved substantial progress in clinical diagnosis via utilizing sophisticated deep learning approaches. However, most deep models have high computational demands, which makes them less feasible for compact devices with low computational requirements. To overcome this problem, we propose a knowledge distillation (KD) strategy to create the compact deep learning model for the real-time multi-label CXR image classification. We study different alternatives of CNNs and Transforms as the teacher to distill the knowledge to a smaller student. Then, we employed explainable artificial intelligence (XAI) to provide the visual explanation for the model decision improved by the KD. Our results on three benchmark CXR datasets show that our KD strategy provides the improved performance on the compact student model, thus being the feasible choice for many limited hardware platforms. For instance, when using DenseNet161 as the teacher network, EEEA-Net-C2 achieved an AUC of 83.7%, 87.1%, and 88.7% on the ChestX-ray14, CheXpert, and PadChest datasets, respectively, with fewer parameters of 4.7 million and computational cost of 0.3 billion FLOPS.

- **manual_class:** 

### 30. `2412.09402`

- **assigned_domain:** medical_imaging
- **title:**

> MultiEYE: Dataset and Benchmark for OCT-Enhanced Retinal Disease Recognition from Fundus Images

- **abstract:**

> Existing multi-modal learning methods on fundus and OCT images mostly require both modalities to be available and strictly paired for training and testing, which appears less practical in clinical scenarios. To expand the scope of clinical applications, we formulate a novel setting, "OCT-enhanced disease recognition from fundus images", that allows for the use of unpaired multi-modal data during the training phase and relies on the widespread fundus photographs for testing. To benchmark this setting, we present the first large multi-modal multi-class dataset for eye disease diagnosis, MultiEYE, and propose an OCT-assisted Conceptual Distillation Approach (OCT-CoDA), which employs semantically rich concepts to extract disease-related knowledge from OCT images and leverage them into the fundus model. Specifically, we regard the image-concept relation as a link to distill useful knowledge from the OCT teacher model to the fundus student model, which considerably improves the diagnostic performance based on fundus images and formulates the cross-modal knowledge transfer into an explainable process. Through extensive experiments on the multi-disease classification task, our proposed OCT-CoDA demonstrates remarkable results and interpretability, showing great potential for clinical application. Our dataset and code are available at https://github.com/xmed-lab/MultiEYE.

- **manual_class:** 

### 31. `2409.11664`

- **assigned_domain:** medical_imaging
- **title:**

> Agent Aggregator with Mask Denoise Mechanism for Histopathology Whole Slide Image Analysis

- **abstract:**

> Histopathology analysis is the gold standard for medical diagnosis. Accurate classification of whole slide images (WSIs) and region-of-interests (ROIs) localization can assist pathologists in diagnosis. The gigapixel resolution of WSI and the absence of fine-grained annotations make direct classification and analysis challenging. In weakly supervised learning, multiple instance learning (MIL) presents a promising approach for WSI classification. The prevailing strategy is to use attention mechanisms to measure instance importance for classification. However, attention mechanisms fail to capture inter-instance information, and self-attention causes quadratic computational complexity. To address these challenges, we propose AMD-MIL, an agent aggregator with a mask denoise mechanism. The agent token acts as an intermediate variable between the query and key for computing instance importance. Mask and denoising matrices, mapped from agents-aggregated value, dynamically mask low-contribution representations and eliminate noise. AMD-MIL achieves better attention allocation by adjusting feature representations, capturing micro-metastases in cancer, and improving interpretability. Extensive experiments on CAMELYON-16, CAMELYON-17, TCGA-KIDNEY, and TCGA-LUNG show AMD-MIL's superiority over state-of-the-art methods.

- **manual_class:** 

### 32. `2410.12831`

- **assigned_domain:** medical_imaging
- **title:**

> Segment as You Wish -- Free-Form Language-Based Segmentation for Medical Images

- **abstract:**

> Medical imaging is crucial for diagnosing a patient's health condition, and accurate segmentation of these images is essential for isolating regions of interest to ensure precise diagnosis and treatment planning. Existing methods primarily rely on bounding boxes or point-based prompts, while few have explored text-related prompts, despite clinicians often describing their observations and instructions in natural language. To address this gap, we first propose a RAG-based free-form text prompt generator, that leverages the domain corpus to generate diverse and realistic descriptions. Then, we introduce FLanS, a novel medical image segmentation model that handles various free-form text prompts, including professional anatomy-informed queries, anatomy-agnostic position-driven queries, and anatomy-agnostic size-driven queries. Additionally, our model also incorporates a symmetry-aware canonicalization module to ensure consistent, accurate segmentations across varying scan orientations and reduce confusion between the anatomical position of an organ and its appearance in the scan. FLanS is trained on a large-scale dataset of over 100k medical images from 7 public datasets. Comprehensive experiments demonstrate the model's superior language understanding and segmentation precision, along with a deep comprehension of the relationship between them, outperforming SOTA baselines on both in-domain and out-of-domain datasets.

- **manual_class:** 

### 33. `2406.05285`

- **assigned_domain:** medical_imaging
- **title:**

> VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging

- **abstract:**

> Foundation models for interactive segmentation in 2D natural images and videos have sparked significant interest in building 3D foundation models for medical imaging. However, the domain gaps and clinical use cases for 3D medical imaging require a dedicated model that diverges from existing 2D solutions. Specifically, such foundation models should support a full workflow that can actually reduce human effort. Treating 3D medical images as sequences of 2D slices and reusing interactive 2D foundation models seems straightforward, but 2D annotation is too time-consuming for 3D tasks. Moreover, for large cohort analysis, it's the highly accurate automatic segmentation models that reduce the most human effort. However, these models lack support for interactive corrections and lack zero-shot ability for novel structures, which is a key feature of "foundation". While reusing pre-trained 2D backbones in 3D enhances zero-shot potential, their performance on complex 3D structures still lags behind leading 3D models. To address these issues, we present VISTA3D, Versatile Imaging SegmenTation and Annotation model, that targets to solve all these challenges and requirements with one unified foundation model. VISTA3D is built on top of the well-established 3D segmentation pipeline, and it is the first model to achieve state-of-the-art performance in both 3D automatic (supporting 127 classes) and 3D interactive segmentation, even when compared with top 3D expert models on large and diverse benchmarks. Additionally, VISTA3D's 3D interactive design allows efficient human correction, and a novel 3D supervoxel method that distills 2D pretrained backbones grants VISTA3D top 3D zero-shot performance. We believe the model, recipe, and insights represent a promising step towards a clinically useful 3D foundation model. Code and weights are publicly available at https://github.com/Project-MONAI/VISTA.

- **manual_class:** 

### 34. `2405.05564`

- **assigned_domain:** medical_imaging
- **title:**

> Joint Edge Optimization Deep Unfolding Network for Accelerated MRI Reconstruction

- **abstract:**

> Magnetic Resonance Imaging (MRI) is a widely used imaging technique, however it has the limitation of long scanning time. Though previous model-based and learning-based MRI reconstruction methods have shown promising performance, most of them have not fully utilized the edge prior of MR images, and there is still much room for improvement. In this paper, we build a joint edge optimization model that not only incorporates individual regularizers specific to both the MR image and the edges, but also enforces a co-regularizer to effectively establish a stronger correlation between them. Specifically, the edge information is defined through a non-edge probability map to guide the image reconstruction during the optimization process. Meanwhile, the regularizers pertaining to images and edges are incorporated into a deep unfolding network to automatically learn their respective inherent a-priori information.Numerical experiments, consisting of multi-coil and single-coil MRI data with different sampling schemes at a variety of sampling factors, demonstrate that the proposed method outperforms other compared methods.

- **manual_class:** 

### 35. `2503.00915`

- **assigned_domain:** medical_imaging
- **title:**

> Multimodal Distillation-Driven Ensemble Learning for Long-Tailed Histopathology Whole Slide Images Analysis

- **abstract:**

> Multiple Instance Learning (MIL) plays a significant role in computational pathology, enabling weakly supervised analysis of Whole Slide Image (WSI) datasets. The field of WSI analysis is confronted with a severe long-tailed distribution problem, which significantly impacts the performance of classifiers. Long-tailed distributions lead to class imbalance, where some classes have sparse samples while others are abundant, making it difficult for classifiers to accurately identify minority class samples. To address this issue, we propose an ensemble learning method based on MIL, which employs expert decoders with shared aggregators and consistency constraints to learn diverse distributions and reduce the impact of class imbalance on classifier performance. Moreover, we introduce a multimodal distillation framework that leverages text encoders pre-trained on pathology-text pairs to distill knowledge and guide the MIL aggregator in capturing stronger semantic features relevant to class information. To ensure flexibility, we use learnable prompts to guide the distillation process of the pre-trained text encoder, avoiding limitations imposed by specific prompts. Our method, MDE-MIL, integrates multiple expert branches focusing on specific data distributions to address long-tailed issues. Consistency control ensures generalization across classes. Multimodal distillation enhances feature extraction. Experiments on Camelyon+-LT and PANDA-LT datasets show it outperforms state-of-the-art methods.

- **manual_class:** 

### 36. `2302.03861`

- **assigned_domain:** medical_imaging
- **title:**

> SwinCross: Cross-modal Swin Transformer for Head-and-Neck Tumor Segmentation in PET/CT Images

- **abstract:**

> Radiotherapy (RT) combined with cetuximab is the standard treatment for patients with inoperable head and neck cancers. Segmentation of head and neck (H&N) tumors is a prerequisite for radiotherapy planning but a time-consuming process. In recent years, deep convolutional neural networks have become the de facto standard for automated image segmentation. However, due to the expensive computational cost associated with enlarging the field of view in DCNNs, their ability to model long-range dependency is still limited, and this can result in sub-optimal segmentation performance for objects with background context spanning over long distances. On the other hand, Transformer models have demonstrated excellent capabilities in capturing such long-range information in several semantic segmentation tasks performed on medical images. Inspired by the recent success of Vision Transformers and advances in multi-modal image analysis, we propose a novel segmentation model, debuted, Cross-Modal Swin Transformer (SwinCross), with cross-modal attention (CMA) module to incorporate cross-modal feature extraction at multiple resolutions.To validate the effectiveness of the proposed method, we performed experiments on the HECKTOR 2021 challenge dataset and compared it with the nnU-Net (the backbone of the top-5 methods in HECKTOR 2021) and other state-of-the-art transformer-based methods such as UNETR, and Swin UNETR. The proposed method is experimentally shown to outperform these comparing methods thanks to the ability of the CMA module to capture better inter-modality complimentary feature representations between PET and CT, for the task of head-and-neck tumor segmentation.

- **manual_class:** 

### 37. `2508.04044`

- **assigned_domain:** medical_imaging
- **title:**

> Iterative pseudo-labeling based adaptive copy-paste supervision for semi-supervised tumor segmentation

- **abstract:**

> Semi-supervised learning (SSL) has attracted considerable attention in medical image processing. The latest SSL methods use a combination of consistency regularization and pseudo-labeling to achieve remarkable success. However, most existing SSL studies focus on segmenting large organs, neglecting the challenging scenarios where there are numerous tumors or tumors of small volume. Furthermore, the extensive capabilities of data augmentation strategies, particularly in the context of both labeled and unlabeled data, have yet to be thoroughly investigated. To tackle these challenges, we introduce a straightforward yet effective approach, termed iterative pseudo-labeling based adaptive copy-paste supervision (IPA-CP), for tumor segmentation in CT scans. IPA-CP incorporates a two-way uncertainty based adaptive augmentation mechanism, aiming to inject tumor uncertainties present in the mean teacher architecture into adaptive augmentation. Additionally, IPA-CP employs an iterative pseudo-label transition strategy to generate more robust and informative pseudo labels for the unlabeled samples. Extensive experiments on both in-house and public datasets show that our framework outperforms state-of-the-art SSL methods in medical image segmentation. Ablation study results demonstrate the effectiveness of our technical contributions.

- **manual_class:** 

### 38. `2405.07338`

- **assigned_domain:** medical_imaging
- **title:**

> Explainable Convolutional Neural Networks for Retinal Fundus Classification and Cutting-Edge Segmentation Models for Retinal Blood Vessels from Fundus Images

- **abstract:**

> Early detection of vision-threatening conditions such as diabetic retinopathy, glaucoma, and age-related macular degeneration depends on retinal fundus image analysis, but manual assessment is slow and expert-dependent. Automated convolutional neural networks classify fundus images accurately yet act as black boxes, and existing retinal vessel segmentation methods lose discriminative power under pathology and seldom exploit attention or transformer backbones. Using the FIVES and DRIVE fundus datasets, we develop a two-pipeline framework that pairs four-class disease classification with attention- and transformer-based vessel segmentation, organised in three stages: (1) FIVES images are augmented by rotation and horizontal and vertical flips and used to fine-tune eight ImageNet-pretrained CNNs: ResNet101, DenseNet169, Xception, InceptionV3, DenseNet121, InceptionResNetV2, ResNet50, and EfficientNetB0. (2) Five gradient-based explanation methods, Grad-CAM, Grad-CAM++, Score-CAM, Faster Score-CAM, and Layer-CAM, are computed on the final convolutional block of each classifier and compared qualitatively across architectures. (3) Ten U-Net variants are benchmarked for vessel segmentation: TransUNet (hybrid CNN--Transformer encoder) and Attention U-Net (gated skip connections), evaluated with ResNet50V2, ResNet101V2, and ResNet152V2 backbones, along with additional Attention U-Net configurations using DenseNet backbones, and the fully transformer-based Swin-UNet. ResNet101 gives the highest classification accuracy: 94.17% (F1 0.942) $>$ 88.33% for EfficientNetB0. For segmentation, the architecture ranking is consistent on both datasets: Attention U-Net $>$ TransUNet $>$ Swin-UNet. The strongest configuration is Attention U-Net with a ResNet101V2 backbone: FIVES IoU 0.722, Dice 0.838; DRIVE IoU 0.648, Dice 0.787, lifting DRIVE IoU 60.80 $\rightarrow$ 64.83 over a prior custom U-Net.

- **manual_class:** 

### 39. `2304.12637`

- **assigned_domain:** medical_imaging
- **title:**

> Generalist Vision Foundation Models for Medical Imaging: A Case Study of Segment Anything Model on Zero-Shot Medical Segmentation

- **abstract:**

> In this paper, we examine the recent Segment Anything Model (SAM) on medical images, and report both quantitative and qualitative zero-shot segmentation results on nine medical image segmentation benchmarks, covering various imaging modalities, such as optical coherence tomography (OCT), magnetic resonance imaging (MRI), and computed tomography (CT), as well as different applications including dermatology, ophthalmology, and radiology. Those benchmarks are representative and commonly used in model development. Our experimental results indicate that while SAM presents remarkable segmentation performance on images from the general domain, its zero-shot segmentation ability remains restricted for out-of-distribution images, e.g., medical images. In addition, SAM exhibits inconsistent zero-shot segmentation performance across different unseen medical domains. For certain structured targets, e.g., blood vessels, the zero-shot segmentation of SAM completely failed. In contrast, a simple fine-tuning of it with a small amount of data could lead to remarkable improvement of the segmentation quality, showing the great potential and feasibility of using fine-tuned SAM to achieve accurate medical image segmentation for a precision diagnostics. Our study indicates the versatility of generalist vision foundation models on medical imaging, and their great potential to achieve desired performance through fine-turning and eventually address the challenges associated with accessing large and diverse medical datasets in support of clinical diagnostics.

- **manual_class:** 

### 40. `2501.09138`

- **assigned_domain:** medical_imaging
- **title:**

> Few-Shot Adaptation of Training-Free Foundation Model for 3D Medical Image Segmentation

- **abstract:**

> Vision foundation models have achieved remarkable progress across various image analysis tasks. In the image segmentation task, foundation models like the Segment Anything Model (SAM) enable generalizable zero-shot segmentation through user-provided prompts. However, SAM primarily trained on natural images, lacks the domain-specific expertise of medical imaging. This limitation poses challenges when applying SAM to medical image segmentation, including the need for extensive fine-tuning on specialized medical datasets and a dependency on manual prompts, which are both labor-intensive and require intervention from medical experts. This work introduces the Few-shot Adaptation of Training-frEe SAM (FATE-SAM), a novel method designed to adapt the advanced Segment Anything Model 2 (SAM2) for 3D medical image segmentation. FATE-SAM reassembles pre-trained modules of SAM2 to enable few-shot adaptation, leveraging a small number of support examples to capture anatomical knowledge and perform prompt-free segmentation, without requiring model fine-tuning. To handle the volumetric nature of medical images, we incorporate a Volumetric Consistency mechanism that enhances spatial coherence across 3D slices. We evaluate FATE-SAM on multiple medical imaging datasets and compare it with supervised learning methods, zero-shot SAM approaches, and fine-tuned medical SAM methods. Results show that FATE-SAM delivers robust and accurate segmentation while eliminating the need for large annotated datasets and expert intervention. FATE-SAM provides a practical, efficient solution for medical image segmentation, making it more accessible for clinical applications.

- **manual_class:** 
