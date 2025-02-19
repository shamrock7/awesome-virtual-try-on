# 精选虚拟试穿（VTON）研究汇总！ <img src='awesome.png' width='37'>

> [minar09/awesome-virtual-try-on: Commit 6b991e3](https://github.com/minar09/awesome-virtual-try-on/commit/6b991e3919136b3252a850f5d4358db2a96ae3a8)

精心整理的与虚拟试穿（VTON）相关的出色研究论文、项目、代码、数据集、研讨会等的清单。

- [基于提示词的虚拟试穿](#Prompt-based-Virtual-Try-on)
- [基于 2D 图像的虚拟试穿](#Image-based-2D-Virtual-Try-on)
- [3D 虚拟试穿](#3D-virtual-try-on)
- [混搭虚拟试穿](#Mix-and-match-Virtual-Try-on)
- [自然场景虚拟试穿](#In-the-wild-Virtual-Try-on)
- [多姿态引导虚拟试穿](#Multi-Pose-Guided-Virtual-Try-on)
- [视频虚拟试穿](#Video-Virtual-Try-on)
- [视频转图像虚拟试穿](#video-to-image-virtual-try-on)
- [非服装类虚拟试穿](#non-clothing-virtual-try-on)
- [姿态引导的人体合成](#pose-guided-human-synthesis)
- [虚拟试穿数据集](#Datasets-for-Virtual-Try-on)
- [相关会议研讨会](#Related-Conference-Workshops)
- [示例](#Demos)
- [相关仓库](#Related-Repositories)

## 基于提示词的虚拟试穿<a name='Prompt-based-Virtual-Try-on'></a>

#### 控制网络

- [ControlNet](https://github.com/lllyasviel/ControlNet) - 提示：将服装图片作为图像输入，并在文本提示中提供人物描述，反之亦可。
- [EditAnything](https://github.com/sail-sg/EditAnything) - 提示：使用一张参考时尚图片作为输入，并在文本提示中说明你想要做出的改动。

#### 稳定扩散

- [Stable-DreamFusion](https://github.com/ashawkey/stable-dreamfusion) - 提示：使用 “图像条件 3D 生成” 选项来编辑你的时尚图片。
- [ThreeStudio](https://github.com/threestudio-project/threestudio) - 提示：在进行时尚图像编辑时，针对基于图像条件的文本提示，使用不同模型来生成输出内容。

## 基于 2D 图像的虚拟试穿<a name='Image-based-2D-Virtual-Try-on></a>

#### AAAI 2025

- MV-VTON：基于扩散模型的多视角虚拟试穿 - [论文](https://arxiv.org/abs/2404.17364), [代码/数据](https://github.com/hywang2002/MV-VTON), [项目](https://hywang2002.github.io/MV-VTON/)

#### CVPR 2024
 
- StableVITON：借助潜在扩散模型学习语义对应关系以实现虚拟试穿 - [项目](https://rlawjdghek.github.io/StableVITON/), [代码/数据/模型](https://github.com/rlawjdghek/StableVITON)
- CAT-DM：基于扩散模型的可控加速虚拟试穿 - [论文](https://arxiv.org/pdf/2311.18405.pdf), [项目](https://github.com/zengjianhao/CAT-DM)
- 用于高保真虚拟试穿的纹理保留扩散模型 - [项目/代码](https://github.com/Gal4way/TPD)
- PICTURE：基于无约束设计的逼真虚拟试穿 - [项目/代码](https://github.com/GAP-LAB-CUHK-SZ/PICTURE)
- M&M VTO：多服装虚拟试穿与编辑 - [项目](https://mmvto.github.io/)

#### AAAI 2024

- 通过顺序变形实现抗挤压虚拟试穿 - [论文](https://arxiv.org/abs/2312.15861), [代码](https://github.com/SHShim0513/SD-VITON)
  
#### WACV 2024

-以不同方式穿着同一套服装 —— 一种可控的虚拟试穿方法 - [论文](https://arxiv.org/abs/2211.16989)
- GC - VTON：为虚拟试穿预测全局一致且考虑遮挡的局部流并保留邻域完整性 - [论文](https://openaccess.thecvf.com/content/WACV2024/papers/Rawal_GC-VTON_Predicting_Globally_Consistent_and_Occlusion_Aware_Local_Flows_With_WACV_2024_paper.pdf)

#### ICCV 2023

- 多模态服装设计师：以人类为中心的潜在扩散模型用于时尚图像编辑 - [论文](https://arxiv.org/abs/2304.02051), [代码](https://github.com/aimagelab/multimodal-garment-designer)
- 尺码很重要：通过面向服装的变换试穿网络实现感知尺码的虚拟试穿 - [论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Size_Does_Matter_Size-aware_Virtual_Try-on_via_Clothing-oriented_Transformation_Try-on_ICCV_2023_paper.pdf), [代码](https://github.com/cotton6/COTTON-size-does-matter)
- 基于姿态 - 服装关键点引导修复的虚拟试穿 - [论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Virtual_Try-On_with_Pose-Garment_Keypoints_Guided_Inpainting_ICCV_2023_paper.pdf), [代码](https://github.com/lizhi-ntu/KGI)

#### CVPR 2023

- GP - VTON：通过协作式局部流全局解析学习实现通用虚拟试穿 - [论文](https://arxiv.org/pdf/2303.13756.pdf), [代码](https://github.com/xiezhy6/GP-VTON), [项目](https://github.com/xiezhy6/GP-VTON)
- 通过语义关联地标将服装与人关联以实现虚拟试穿 - [论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_Linking_Garment_With_Person_via_Semantically_Associated_Landmarks_for_Virtual_CVPR_2023_paper.pdf)
- TryOnDiffusion：两个 U 型网络的故事 - [论文](https://arxiv.org/pdf/2306.08276.pdf), [项目](https://tryondiffusion.github.io/)
  
#### NeurIPS 2022

- 通过 3D 感知的全局对应学习实现困难姿态下的虚拟试穿 - [论文](https://arxiv.org/pdf/2211.14052.pdf), [项目](https://neurips.cc/virtual/2022/poster/54642)
  
#### WACV 2022

- C-VTON：上下文驱动的基于图像的虚拟试穿网络 - [论文](https://openaccess.thecvf.com/content/WACV2022/papers/Fele_C-VTON_Context-Driven_Image-Based_Virtual_Try-On_Network_WACV_2022_paper.pdf), [代码/模型/数据](https://github.com/benquick123/C-VTON)
  
#### ECCV 2022

- 着装规范：高分辨率多品类虚拟试穿 - [论文](https://arxiv.org/pdf/2204.08532.pdf), [代码/数据](https://github.com/aimagelab/dress-code)
- 具有错位与遮挡处理条件的高分辨率虚拟试穿 - [论文](https://arxiv.org/abs/2206.14180), [代码/模型](https://github.com/sangyun884/HR-VITON)
- 基于可变形注意力流的单阶段虚拟试穿 - [论文](https://arxiv.org/abs/2207.09161), [代码/模型](https://github.com/OFA-Sys/DAFlow)
  
#### CVPR 2022

- 基于循环三层变换的全范围虚拟试穿 - [论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Full-Range_Virtual_Try-On_With_Recurrent_Tri-Level_Transform_CVPR_2022_paper.pdf), [项目](https://lzqhardworker.github.io/RT-VTON/)
- 基于风格的全局外观流用于虚拟试穿 - [论文](https://arxiv.org/abs/2204.01046), [代码/模型](https://github.com/SenHe/Flow-Style-VTON)
- 弱监督高保真服装模型生成 - [论文](https://arxiv.org/pdf/2112.07200.pdf), [代码/模型](https://github.com/RuiLiFeng/Deep-Generative-Projection)
- 通过观看舞蹈视频实现自然场景着装 - [论文](https://openaccess.thecvf.com/content/CVPR2022/html/Dong_Dressing_in_the_Wild_by_Watching_Dance_Videos_CVPR_2022_paper.html), [项目](https://awesome-wflow.github.io/)
  
#### CVPRW 2022

- 用于虚拟试穿的双分支协作 Transformer - [论文](https://openaccess.thecvf.com/content/CVPR2022W/CVFAD/papers/Fenocchi_Dual-Branch_Collaborative_Transformer_for_Virtual_Try-On_CVPRW_2022_paper.pdf)
- 着装规范：高分辨率多品类虚拟试穿 - [论文](https://openaccess.thecvf.com/content/CVPR2022W/CVFAD/papers/Morelli_Dress_Code_High-Resolution_Multi-Category_Virtual_Try-On_CVPRW_2022_paper.pdf), [代码/数据](https://github.com/aimagelab/dress-code)
  
#### ICCV 2021

- 有序着装：用于姿态转移、虚拟试穿和服装编辑的循环人物图像生成 - [论文](https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_Dressing_in_Order_Recurrent_Person_Image_Generation_for_Pose_Transfer_ICCV_2021_paper.pdf), [代码](https://github.com/cuiaiyu/dressing-in-order), [Colab](https://colab.research.google.com/drive/1WfeKTPtt3qtlcTlrX47J03mxUzbVvyrL?usp=sharing)
- ZFlow：基于门控外观流与 3D 先验的虚拟试穿 - [论文](https://openaccess.thecvf.com/content/ICCV2021/papers/Chopra_ZFlow_Gated_Appearance_Flow-Based_Virtual_Try-On_With_3D_Priors_ICCV_2021_paper.pdf)
- FashionMirror：基于协同注意力特征重映射及顺序模板姿态的虚拟试穿 - [论文](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_FashionMirror_Co-Attention_Feature-Remapping_Virtual_Try-On_With_Sequential_Template_Poses_ICCV_2021_paper.pdf)
  
#### CVPR 2021

- 通过提炼外观流实现无解析器的虚拟试穿 - [论文](https://arxiv.org/pdf/2103.04559.pdf), [代码/数据/模型](https://github.com/geyuying/PF-AFN)
- VITON-HD：通过错位感知归一化实现高分辨率虚拟试穿 - [论文](https://arxiv.org/pdf/2103.16874.pdf), [代码/模型](https://github.com/shadow2496/VITON-HD)
- 用于高逼真虚拟试穿的解缠循环一致性 - [论文](https://arxiv.org/pdf/2103.09479.pdf), [代码/数据/模型](https://github.com/ChongjianGE/DCTON)
- 注重细节，实现准确逼真的服装可视化 - [论文](https://arxiv.org/abs/2106.06593), [示例](https://revery.ai/demo.html)

#### ACCV 2020

- CloTH-VTON：基于混合图像的虚拟试穿的服装三维重建 - [论文](https://openaccess.thecvf.com/content/ACCV2020/html/Minar_CloTH-VTON_Clothing_Three-dimensional_reconstruction_for_Hybrid_image-based_Virtual_Try-ON_ACCV_2020_paper.html), [项目](https://minar09.github.io/clothvton/)

#### ECCV 2020

- 无需掩码时请勿掩码：一种无解析器的虚拟试穿 - [论文](https://arxiv.org/pdf/2007.02721.pdf)

#### CVPR 2020

- 通过自适应生成与保留图像内容迈向逼真的虚拟试穿 - [论文/代码/数据](https://github.com/switchablenorms/DeepFashion_Try_On)
- 基于非配对数据的图像虚拟试穿网络 - [论文](http://openaccess.thecvf.com/content_CVPR_2020/html/Neuberger_Image_Based_Virtual_Try-On_Network_From_Unpaired_Data_CVPR_2020_paper.html)
- 语义多模态图像合成 - [论文/代码/模型](https://seanseattle.github.io/SMIS/)

#### CVPRW 2020

- CP-VTON +：基于图像的服装形状与纹理保留的虚拟试穿 - [论文/代码/数据/模型](https://minar09.github.io/cpvtonplus/)
- 基于人体模型的服装 3D 重建及其在基于图像的虚拟试穿中的应用 - [论文/项目](https://minar09.github.io/c3dvton/)

#### ICCV 2019

- VTNFP：一种基于图像且保留人体与服装特征的虚拟试穿网络 - [论文](http://openaccess.thecvf.com/content_ICCV_2019/html/Yu_VTNFP_An_Image-Based_Virtual_Try-On_Network_With_Body_and_Clothing_ICCV_2019_paper.html)
- ClothFlow：一种基于流的着装人物生成模型 - [论文](http://openaccess.thecvf.com/content_ICCV_2019/html/Han_ClothFlow_A_Flow-Based_Model_for_Clothed_Person_Generation_ICCV_2019_paper.html)

#### ICCVW 2019

- UVTON：在基于图像的虚拟试穿网络中，基于 UV 映射考量人体三维结构, [论文](http://openaccess.thecvf.com/content_ICCVW_2019/html/CVFAD/Kubo_UVTON_UV_Mapping_to_Consider_the_3D_Structure_of_a_ICCVW_2019_paper.html)
- LA-VITON：一种实现吸引人的虚拟试穿网络 - [论文](http://openaccess.thecvf.com/content_ICCVW_2019/html/CVFAD/Lee_LA-VITON_A_Network_for_Looking-Attractive_Virtual_Try-On_ICCVW_2019_paper.html)
- 通过多尺度面片对抗损失实现稳健的服装变形用于虚拟试穿框架 - [论文](http://openaccess.thecvf.com/content_ICCVW_2019/html/HBU/Ayush_Robust_Cloth_Warping_via_Multi-Scale_Patch_Adversarial_Loss_for_Virtual_ICCVW_2019_paper.html)
- 通过辅助人体分割学习助力虚拟试穿 - [论文](http://openaccess.thecvf.com/content_ICCVW_2019/html/CVFAD/Ayush_Powering_Virtual_Try-On_via_Auxiliary_Human_Segmentation_Learning_ICCVW_2019_paper.html)
- 生成穿着定制服装的高分辨率时尚模特图像 - [论文](http://openaccess.thecvf.com/content_ICCVW_2019/html/CVFAD/Yildirim_Generating_High-Resolution_Fashion_Model_Images_Wearing_Custom_Outfits_ICCVW_2019_paper.html)

#### ECCV 2018

- 迈向基于图像且保留特征的虚拟试穿网络 - [论文](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bochao_Wang_Toward_Characteristic-Preserving_Image-based_ECCV_2018_paper.pdf), [代码](https://github.com/sergeywong/cp-vton)
- SwapNet：单视图图像中的服装转移 - [论文](http://openaccess.thecvf.com/content_ECCV_2018/papers/Amit_Raj_SwapNet_Garment_Transfer_ECCV_2018_paper.pdf), [代码（社区贡献）](https://github.com/andrewjong/SwapNet)

#### CVPR 2018

- VITON：一种基于图像的虚拟试穿网络 - [论文](https://arxiv.org/abs/1711.08447), [代码/模型](https://github.com/xthan/VITON)

#### 其他

- BootComp：通过个性化多服装实现可控的人物图像生成 - [论文](https://arxiv.org/abs/2411.16801), [项目](https://yisol.github.io/BootComp/)
- CatVTON：使用扩散模型进行虚拟试穿，拼接即所需 - [论文](https://arxiv.org/pdf/2407.15886), [代码](https://github.com/Zheng-Chong/CatVTON)
- IMAGDressing-v1：可定制的虚拟着装 - [示例](https://sf.dictdoc.site/), [代码](https://github.com/muzishen/IMAGDressing), [项目](https://imagdressing.github.io/)
- 神奇服装：可控的服装驱动图像合成 - [论文](https://arxiv.org/abs/2404.09512), [代码](https://github.com/ShineChen1024/MagicClothing)
- IDM-VTON：改进扩散模型以实现自然场景下真实的虚拟试穿 - [示例](https://huggingface.co/spaces/yisol/IDM-VTON), [论文](https://arxiv.org/abs/2403.05139), [项目](https://idm-vton.github.io/)
- OOTDiffusion：基于着装融合的潜在扩散用于可控虚拟试穿 - [代码](https://github.com/levihsu/OOTDiffusion)
- Outfit Anyone：面向任意服装与任意人物的超高质量虚拟试穿 - [项目](https://humanaigc.github.io/outfit-anyone/)
- Street TryOn：从未配对人物图像中学习自然场景下的虚拟试穿 -[论文](https://arxiv.org/pdf/2311.16094.pdf), [数据](https://github.com/cuiaiyu/street-tryon-benchmark)
- FICE：基于引导式生成对抗网络反演的文本条件时尚图像编辑 - [论文](https://arxiv.org/abs/2301.02110), [代码](https://github.com/martinpernus/fice)
- OccluMix：通过语义引导混合实现去遮挡虚拟试穿, TMM 2023 - [论文](https://arxiv.org/abs/2301.00965), [代码](https://github.com/jychen9811/doc-vton)
- 借助外观流驾驭扩散模型的力量实现高质量虚拟试穿, ACM Multimedia 2023 - [论文](https://arxiv.org/abs/2308.06101), [代码](https://github.com/bcmi/DCI-VTON-Virtual-Try-On)
- LaDI-VTON：潜在扩散文本反演增强的虚拟试穿, ACM Multimedia 2023 - [论文](https://arxiv.org/abs/2305.13501), [代码](https://github.com/miccunifi/ladi-vton)
- 基于渐进式服装变形的肢体感知虚拟试穿网络, IEEE Multimedia 2023 - [论文](https://ieeexplore.ieee.org/abstract/document/10152500)
- FashionTex：通过文本与纹理实现可控虚拟试穿, SIGGRAPH 2023 - [论文](https://arxiv.org/pdf/2305.04451.pdf)
- DreamPaint：无需 3D 建模的电商商品少样本修复用于虚拟试穿 - [论文](https://arxiv.org/pdf/2305.01257.pdf)
- PG-VTON：一种通过渐进推理范式实现的新型基于图像的虚拟试穿方法 - [论文](https://arxiv.org/pdf/2304.08956.pdf), [代码](https://github.com/NerdFNY/PGVTON)
- 填充面料：基于图像的虚拟试穿中基于人体感知的自监督修复, BMVC 2022 - [论文](https://arxiv.org/pdf/2210.00918.pdf), [代码/模型/数据](https://github.com/hasibzunair/fifa-tryon)
- 探索 TryOnGAN, CODS-COMAD 2022 - [论文](https://arxiv.org/pdf/2201.01703.pdf)
- FitGAN：用于时尚领域的贴合与形状逼真的生成对抗网络 - [论文](https://arxiv.org/pdf/2206.11768.pdf)
- 通过对应学习实现图像合成的空间变换 - [论文](https://arxiv.org/pdf/2207.02398.pdf)
- PASTA-GAN++：一种用于高分辨率非配对虚拟试穿的通用框架 - [论文](https://arxiv.org/pdf/2207.13475.pdf)
- 基于骨架特征在虚拟试穿中的重要性 - [论文](https://arxiv.org/pdf/2208.08076.pdf)
- 无需亲临即可试穿：虚拟试穿服装的综合调查, 《多媒体工具与应用》 - [论文](https://link.springer.com/article/10.1007/s11042-022-12802-6)
- 通过语义适配与分布式组件化实现高保真虚拟试穿网络, 《计算视觉媒体》2022 年 - [论文](https://link.springer.com/content/pdf/10.1007/s41095-021-0264-2.pdf)
- St-Vton：用于基于图像的虚拟试穿的自监督视觉 Transformer - [论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4140115)
- 具有增强变形模块的逼真虚拟试穿，《智能系统与计算进展》2022 年 - [论文](https://link.springer.com/chapter/10.1007/978-981-16-5157-1_66)
- 使用增强现实的虚拟试穿 - [论文](https://link.springer.com/chapter/10.1007/978-981-19-1122-4_54)
- WG-VITON：用于上下装的穿着引导虚拟试穿 - [论文](https://arxiv.org/pdf/2205.04759.pdf)
- VTNCT：一种结合特征与像素变换的基于图像的虚拟试穿网络, 《视觉计算机》2022 年 - [论文](https://link.springer.com/article/10.1007/s00371-022-02480-8)
- RMGN：一种用于无解析器虚拟试穿的区域掩码引导网络, IJCAI-ECAI 2022 - [论文](https://arxiv.org/pdf/2204.11258.pdf), [代码/模型/数据](https://github.com/jokerlc/RMGN-VITON)
- 一种基于流的生成网络用于逼真虚拟试穿, IEEE Access 2022 - [论文](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9757136), [代码/模型/数据](https://github.com/gxl-groups/FVNT)
- 一种面向服装商业行业的高效风格虚拟试穿网络 - [论文](https://arxiv.org/pdf/2105.13183.pdf)
- 基于 PF-AFN 的跨品类虚拟试穿技术研究, ICVIP 2021 - [论文](https://dl.acm.org/doi/pdf/10.1145/3511176.3511201?casa_token=RY8suIYfg-8AAAAA:ce3F43Et11IaFISkObmvOedFz5FrH0B6QnSHPu3Ro8_GEOLKGdf8qoww1RY5V9zdGm7T9sxpND15Wcs)
- 具有属性变换与局部渲染的虚拟试穿网络, 《IEEE 多媒体汇刊》2021 年 - [论文](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9397349&casa_token=q6ADdrsxPToAAAAA:5_PjUtYaX9d37HGetPpsvpQApUMhfCitOIOAlDpCSqW57AbRLc3Z3M0CR4vdbHsvdaEQ4qbYTkI)
- 变换、变形与着装：一种新型变换引导的虚拟试穿模型，《ACM 多媒体计算通信与应用汇刊》2022 年 - [论文](https://dl.acm.org/doi/pdf/10.1145/3491226?casa_token=7bTieMExNlkAAAAA%3AjGDdaeAR0HQ-M0ZO2u3olmaVjf3W_XtVgK5wPdgKiPhLP4F1yRUA28c7teapiVedba5FlPrfBTSvnC4)
- 通过面片路由空间自适应生成对抗网络迈向可扩展的非配对虚拟试穿，2021 年神经信息处理系统大会. - [论文](https://arxiv.org/abs/2111.10544)
- 任意虚拟试穿网络：人体与服装之间的特征保留与权衡 - [论文](https://arxiv.org/abs/2111.12346)
- PT-VTON：一种基于渐进式姿态注意力转移的基于图像的虚拟试穿网络 - [论文](https://arxiv.org/abs/2111.12167)
- 使用随机图像裁剪进行数据增强用于高分辨率虚拟试穿 (VITON-CROP) - [论文](https://arxiv.org/abs/2111.08270)
- 用于实时虚拟试穿的单件服装捕捉与合成, UIST 2021 - [论文](https://arxiv.org/abs/2109.04654), [项目](https://sites.google.com/view/deepmannequin/home)
- WAS-VTON：虚拟试穿网络的变形架构搜索 - [论文](https://arxiv.org/abs/2108.00386), [代码/模型](https://github.com/xiezhy6/WAS-VTON)
- 内衣模型的形状可控虚拟试穿 - [论文](https://arxiv.org/abs/2107.13156)
- 用于虚拟试穿的服装交互式 Transformer - [论文](https://arxiv.org/abs/2104.05519), [代码/模型](https://github.com/Amazingren/CIT)
- CloTH-VTON +：基于混合图像的虚拟试穿的服装三维重建, IEEE Access 2021 - [论文](https://ieeexplore.ieee.org/document/9354778), [项目](https://minar09.github.io/clothvtonplus/)
- VITON-GT：一种基于几何变换的基于图像的虚拟试穿模型, ICPR 2020 - [论文](https://iris.unimore.it/retrieve/312578/2020_ICPR_Virtual_Try_On.pdf)
- TryOnGAN：基于分层插值的人体感知试穿, SIGGRAPH 2021 - [论文/项目](https://tryongan.github.io/tryongan/)
- VOGUE：通过 StyleGAN 插值优化实现试穿 - [论文/项目](https://vogue-try-on.github.io/)
- 基于服装变换的深度虚拟试穿, ICS 2018 - [论文](https://link.springer.com/chapter/10.1007/978-981-13-9190-3_22), [代码](https://github.com/b01902041/Deep-Virtual-Try-on-with-Clothes-Transform)
- NDNet：服装自然变形以提升虚拟试穿体验, ACM SAC 2021
- 基于关键点的 2D 虚拟试穿网络系统, JAKO 2020 - [论文](https://www.koreascience.or.kr/article/JAKO202010163508810.pdf)
- 基于生成对抗网络的虚拟试穿：分类学综述 - [书籍章节](https://www.igi-global.com/chapter/virtual-try-on-with-generative-adversarial-networks/260791)
- LGVTON：一种基于地标引导的虚拟试穿方法 - [论文](https://arxiv.org/abs/2004.00562v1), [代码](https://github.com/dp-isi/LGVTON)
- SieveNet：一种用于稳健的基于图像的虚拟试穿的统一框架, WACV 2020 - [论文](http://openaccess.thecvf.com/content_WACV_2020/html/Jandial_SieveNet_A_Unified_Framework_for_Robust_Image-Based_Virtual_Try-On_WACV_2020_paper.html), [代码/模型（社区贡献）](https://github.com/levindabhi/SieveNet)
- GarmentGAN：逼真的对抗式时尚转移 - [论文](https://arxiv.org/abs/2003.01894)
- 通过形状匹配与多次变形迈向准确逼真的虚拟试穿 - [论文](https://arxiv.org/abs/2003.10817)
- 用于定制虚拟试穿的 3D 姿态映射与神经人体贴合的时尚合身分析, IEEE Access 2020 - [论文](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9091008)
- 基于修复的虚拟试穿网络用于选择性服装转移, IEEE Access 2019 - [论文](https://ieeexplore.ieee.org/document/8836494)
- SP-VITON：基于图像的形状保留虚拟试穿网络 2019 - [论文](https://link.springer.com/article/10.1007/s11042-019-08363-w)
- VITON-GAN：基于对抗损失训练的虚拟试穿图像生成器, 2019 年欧洲图形学会议海报 - [论文](https://arxiv.org/abs/1911.07926v1), [代码/模型](https://github.com/shionhonda/viton-gan)
- 具有结构一致性的基于图像的虚拟试穿网络, ICIP 2019 - [论文](https://ieeexplore.ieee.org/document/8803811)
- End-to-End Learning of Geometric Deformations of Feature Maps for Virtual Try-On - [论文](https://arxiv.org/abs/1906.01347v2)
- M2E-Try On Net：从模特到大众的时尚 - [论文](https://arxiv.org/abs/1811.08599v1)

## 3D 虚拟试穿<a name='3D-virtual-try-on'></a>

#### CVPR 2024

- DiffAvatar：基于可微模拟的适用于仿真的服装优化 - [项目](https://people.csail.mit.edu/liyifei/publication/diffavatar/), [代码](https://github.com/facebookresearch/DiffAvatar)
- 基于形状与变形先验的服装恢复 - [项目/代码](https://github.com/liren2515/GarmentRecovery)
- SIFU：基于侧视图条件的隐式函数用于真实可用的着装人体重建 - [项目](https://river-zhang.github.io/SIFU-projectpage/), [代码](https://github.com/River-Zhang/SIFU)

#### CVPR 2023

- DrapeNet：服装生成与自监督褶皱模拟 - [代码](https://github.com/liren2515/DrapeNet)

#### ECCV 2022

- 自然场景下的 3D 着装人体重建 - [代码](https://github.com/hygenie1228/ClothWild_RELEASE)

#### NeurIPS 2022

- ULNeF：用于混合搭配虚拟试穿的解缠分层神经场 - [论文](https://neurips.cc/virtual/2022/poster/53336), [项目](https://mslab.es/projects/ULNeF/)

#### WACV 2022

- 用于 3D 虚拟试穿系统的从单目 2D 图像稳健的 3D 服装数字化 - [论文](https://openaccess.thecvf.com/content/WACV2022/papers/Majithia_Robust_3D_Garment_Digitization_From_Monocular_2D_Images_for_3D_WACV_2022_paper.pdf)

#### ICCV 2021

- M3D-VTON：一种单目到 3D 的虚拟试穿网络 - [论文](https://arxiv.org/abs/2108.05126), [代码](https://github.com/fyviezhao/M3D-VTON)

#### CVPR 2021

- 通过生成式 3D 服装模型实现自监督碰撞处理用于虚拟试穿 - [论文](https://arxiv.org/abs/2105.06462), [项目](http://mslab.es/projects/SelfSupervisedGarmentCollisions/), [代码](https://github.com/isantesteban/vto-garment-collisions)

#### ECCV 2020

- BCNet：从单张图像学习人体与服装形状 - [论文](https://arxiv.org/pdf/2004.00214.pdf), [代码/数据](https://github.com/jby1993/BCNet)
- 基于缝纫图案图像的生成对抗网络服装生成 - [论文/代码/模型/数据](https://gamma.umd.edu/researchdirections/virtualtryon/garmentgeneration/)
- Deep Fashion3D：一个用于从单张图像进行 3D 服装重建的数据集与基准 - [论文/数据](https://kv2000.github.io/2020/03/25/deepFashion3DRevisited/)
- SIZER：一个用于解析 3D 服装并学习尺寸敏感 3D 服装的数据集与模型 - [论文/代码/数据](https://virtualhumans.mpi-inf.mpg.de/sizer/)
- CLOTH3D：着装的 3D 人体 - [论文](https://arxiv.org/pdf/1912.02792.pdf)

#### CVPR 2020

- 学习将服装图像纹理转移到 3D 人体上 - [论文/代码](http://virtualhumans.mpi-inf.mpg.de/pix2surf/)
- TailorNet：根据人体姿态、形状和服装风格预测 3D 服装 - [论文/代码/数据](http://virtualhumans.mpi-inf.mpg.de/tailornet/)
- 学习为生成的服装穿着 3D 人物 - [论文/代码/数据](https://cape.is.tue.mpg.de/)

#### ICCV 2019

- Multi - Garment Net：从图像学习为 3D 人物着装 - [论文/代码/数据](http://virtualhumans.mpi-inf.mpg.de/mgn/)
- 3DPeople：着装人体的几何建模 - [论文](https://arxiv.org/abs/1904.04571)
- GarNet：一种用于快速准确 3D 服装褶皱模拟的双流网络 - [论文/数据](https://www.epfl.ch/labs/cvlab/research/garment-simulation/garnet/)

#### ECCV 2018

- DeepWrinkles：准确逼真的服装建模 - [论文](https://arxiv.org/abs/1808.03417)

#### CVPR 2018

- 基于视频的 3D 人物模型重建 - [论文/代码/数据](http://gvv.mpi-inf.mpg.de/projects/wxu/VideoAvatar/)

#### 其他

- DM-VTON：提炼的移动端实时虚拟试穿, ISMAR 2023 - [论文](https://arxiv.org/abs/2308.13798), [代码](https://github.com/KiseKloset/DM-VTON)
- 基于外观流与形状场的三阶段 3D 虚拟试穿网络，《视觉计算机》2023 年 - [论文](https://link.springer.com/article/10.1007/s00371-023-02946-3)
- 用于 3D 虚拟试穿系统的从单目 2D 图像稳健的 3D 服装数字化, 2021 - [论文](https://arxiv.org/pdf/2111.15140.pdf)
- 通过多尺度特征捕捉实现逼真的单目到 3D 虚拟试穿, ICASSP 2022 - [论文](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9747277&casa_token=EALQVtpxonwAAAAA:GKEdQBbsji39XrEgGiSk7pJWgRB1KAH2me79_hsuaPSYTUfVn76h3Jyavxuk57CCWI6HYPRhRUk)
- 基于点的人体服装建模 - [论文](https://arxiv.org/pdf/2104.08230.pdf)
- CloTH-VTON +：基于混合图像的虚拟试穿的服装三维重建, IEEE Access 2021 - [论文](https://ieeexplore.ieee.org/document/9354778), [项目](https://minar09.github.io/clothvtonplus/)
- 基于物理的神经模拟器用于服装动画 - [论文](https://arxiv.org/pdf/2012.11310.pdf)
- GarNet++：通过曲率损失改进快速准确的静态 3D 服装褶皱模拟, IEEE T-PAMI, 2020 - [论文](https://arxiv.org/abs/2007.10867)
- DeepCloth：用于形状与风格编辑的神经服装表示 - [论文](https://arxiv.org/abs/2011.14619)
- CloTH-VTON：基于混合图像的虚拟试穿的服装三维重建, ACCV 2020 - [论文](https://openaccess.thecvf.com/content/ACCV2020/html/Minar_CloTH-VTON_Clothing_Three-dimensional_reconstruction_for_Hybrid_image-based_Virtual_Try-ON_ACCV_2020_paper.html), [项目](https://minar09.github.io/clothvton/)
- 用于参数化虚拟试穿的全卷积图神经网络, ACM SCA 2020 - [论文/项目](http://mslab.es/projects/FullyConvolutionalGraphVirtualTryOn)
- DeePSD：用于 3D 服装动画的自动深度蒙皮与姿态空间变形 - [论文](https://arxiv.org/pdf/2009.02715.pdf)
- 基于人体模型的服装 3D 重建及其在基于图像的虚拟试穿中的应用, CVPRW 2020 - [论文/项目](https://minar09.github.io/c3dvton/)
- 基于学习的服装动画用于虚拟试穿，2019 年欧洲图形学会议, Eurographics 2019 - [论文/项目](http://dancasas.github.io/projects/LearningBasedVirtualTryOn/index.html), [代码](https://github.com/isantesteban/vto-learning-based-animation)
- 学习用于服装动画交互式创作的内在服装空间, SIGGRAPH Asia 2019 - [论文/代码](http://geometry.cs.ucl.ac.uk/projects/2019/garment_authoring/)
- 从 RGB 图像进行 3D 虚拟服装建模, ISMAR 2019 - [论文](https://arxiv.org/abs/1908.00114)
- 用于虚拟试穿系统的深度服装图像抠图, ICCVW 2019 - [论文](http://openaccess.thecvf.com/content_ICCVW_2019/html/CVFAD/Shin_Deep_Garment_Image_Matting_for_a_Virtual_Try-on_System_ICCVW_2019_paper.html)
- 学习用于多模态服装设计的共享形状空间, SIGGRAPH Asia 2018 - [论文/代码/数据](http://geometry.cs.ucl.ac.uk/projects/2018/garment_design/)
- 从单视图图像进行详细的服装恢复, ACM TOG 2018 - [论文](https://arxiv.org/abs/1608.01250)
- ClothCap：无缝 4D 服装捕捉与重定向, SIGGRAPH 2017 - [论文](http://clothcap.is.tue.mpg.de/)
- 通过基于图像的渲染实现虚拟试穿, IEEE T-VCG 2013 - [论文](https://ieeexplore.ieee.org/document/6487501)
- 无标记服装捕捉, ACM TOG 2008 - [论文/数据](http://www.cs.ubc.ca/labs/imager/tr/2008/MarkerlessGarmentCapture/)

## 混搭虚拟试穿<a name='Mix-and-match-Virtual-Try-on'></a>

- 以不同方式穿着同一套服装 —— 一种可控的虚拟试穿方法, WACV 2024 -[论文](https://arxiv.org/abs/2211.16989)
- UMFuse：用于人体编辑应用的统一多视图融合, ICCV 2023 - [论文](https://openaccess.thecvf.com/content/ICCV2023/html/Jain_UMFuse_Unified_Multi_View_Fusion_for_Human_Editing_Applications_ICCV_2023_paper.html), [项目](https://mdsrlab.github.io/2023/08/13/UMFuse-ICCV.html)
- 有序着装：用于姿态转移、虚拟试穿和服装编辑的循环人物图像生成, ICCV 2021 -[论文](https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_Dressing_in_Order_Recurrent_Person_Image_Generation_for_Pose_Transfer_ICCV_2021_paper.pdf), [代码](https://github.com/cuiaiyu/dressing-in-order), [Colab](https://colab.research.google.com/drive/1WfeKTPtt3qtlcTlrX47J03mxUzbVvyrL?usp=sharing)
- 注重细节，实现准确逼真的服装可视化, CVPR 2021 -[论文](https://arxiv.org/pdf/2106.06593.pdf)
- 基于非配对数据的图像虚拟试穿网络, CVPR 2020 - [论文](https://assets.amazon.science/1a/2b/7a4dd8264ce19a959559da799aff/scipub-1281.pdf), [代码](https://github.com/trinanjan12/Image-Based-Virtual-Try-on-Network-from-Unpaired-Data)

## 自然场景虚拟试穿<a name='In-the-wild-Virtual-Try-on'></a>

- Street TryOn：从未配对人物图像中学习自然场景下的虚拟试穿 -[论文](https://arxiv.org/pdf/2311.16094.pdf), [数据](https://github.com/cuiaiyu/street-tryon-benchmark)
- 通过观看舞蹈视频实现自然场景着装, CVPR 2022 - [论文](https://arxiv.org/abs/2203.15320), [项目](https://awesome-wflow.github.io/)

## 多姿态引导虚拟试穿<a name='Multi-Pose-Guided-Virtual-Try-on'></a>

- CF-VTON：基于跨域融合的多姿态虚拟试穿, ICASSP 2023 - [论文](https://ieeexplore.ieee.org/abstract/document/10095176)
- 基于 3D 服装重建的多姿态虚拟试穿, IEEE Access 2021 - [论文](https://ieeexplore.ieee.org/document/9512041)
- SPG-VTON：多姿态虚拟试穿的语义预测引导 - [论文](https://arxiv.org/abs/2108.01578)
- 关注每一个细节：具有细粒度细节的虚拟试穿, ACM MM 2020 - [论文](https://dl.acm.org/doi/pdf/10.1145/3394171.3413514), [代码/模型](https://github.com/JDAI-CV/Down-to-the-Last-Detail-Virtual-Try-on-with-Detail-Carving), [ArXiv](https://arxiv.org/abs/1912.06324v2)
- 迈向多姿态引导的虚拟试穿网络, ICCV 2019 - [论文](https://arxiv.org/abs/1902.11026), [代码](https://github.com/thaithanhtuan/MyMGVTON)
- FIT-ME：基于图像的任意姿态虚拟试穿, ICIP 2019 - [论文](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8803681&casa_token=2CL5K9pwy1IAAAAA:OTa5P-h6RWj9BdQVvkxQURR8tDy4Eg1BZynYOizMyQACnE-zL_EHu2xRzyXBOWijP_cItaO4)
- 以任意姿态虚拟试穿新服装, ACM MM 2019 - [论文](https://dl.acm.org/doi/pdf/10.1145/3343031.3350946?casa_token=w7EzejnZIaEAAAAA:KvDBsi1xYswuQuzEdJO-rsTDvysnSLYlAYi1J2st5lf8lnyotm5-umPKQupGaMEPUGxyBzijUkA9)
- FashionOn：基于语义引导、结合详细人体和服装信息的图像式虚拟试穿, ACM MM 2019 - [论文](https://dl.acm.org/doi/pdf/10.1145/3343031.3351075?casa_token=7y85FCo6B-QAAAAA:diZbVYmcSK13bMQ94MzrMG_-VvVG_oNFoGpI8wCBFJ_dHEzYnLBAPn2ZwbAgj_pmOWFMD6_1hOuk)

## 视频虚拟试穿<a name='Video-Virtual-Try-on'></a>

- ClothFormer：在所有模块中实现视频虚拟试穿的有效控制, CVPR 2022 - [论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_ClothFormer_Taming_Video_Virtual_Try-On_in_All_Module_CVPR_2022_paper.pdf), [代码](https://github.com/luxiangju-PersonAI/ClothFormer), [项目](https://cloth-former.github.io/)
- MV-TON：基于记忆的视频虚拟试穿网络, ACM MM 2021 - [论文](https://arxiv.org/abs/2108.07502)
- ShineOn：为实用的基于视频的虚拟服装试穿提供照明设计选择, WACV 2021 Workshop - [项目/论文/代码](https://gauravkuppa.github.io/publication/2021-01-09-shine-on-1)
- FW-GAN：用于视频虚拟试穿的流导航变形生成对抗网络, ICCV 2019 - [论文](http://openaccess.thecvf.com/content_ICCV_2019/html/Dong_FW-GAN_Flow-Navigated_Warping_GAN_for_Video_Virtual_Try-On_ICCV_2019_paper.html)
- 无监督的图像到视频服装迁移, ICCVW 2019 - [论文](http://openaccess.thecvf.com/content_ICCVW_2019/html/CVFAD/Pumarola_Unsupervised_Image-to-Video_Clothing_Transfer_ICCVW_2019_paper.html)
- 单目视频序列中的服装替换, ACM TOG 2014 - [论文](https://dl.acm.org/doi/10.1145/2634212)

## 视频转图像虚拟试穿<a name='video-to-image-virtual-try-on'></a>

- DPV-VTON：保留细节的基于视频的虚拟试穿, ICMIP 2023 - [论文](https://dl.acm.org/doi/10.1145/3599589.3599599)

## 非服装类虚拟试穿<a name='non-clothing-virtual-try-on'></a>

- 通过深度逆图形学和可学习的可微渲染器，基于单张示例图像实现实时虚拟试穿, EUROGRAPHICS 2022 - [论文](https://arxiv.org/pdf/2205.06305.pdf)
- ARShoe：智能手机上的实时增强现实鞋子试穿系统, ACM Multimedia 2021 - [论文](https://arxiv.org/pdf/2108.10515.pdf)
- 基于示例的实时视频妆容合成深度图形编码器, CVPRW 2021 - [论文](https://arxiv.org/pdf/2105.06407.pdf)
- FITTINTM—— 在线 3D 鞋子试穿, 3DBODY.TECH 2020 - [论文](http://www.3dbodyscanning.org/cap/papers/2020/2058revkov.pdf)
- CA-GAN：用于可控妆容迁移的弱监督颜色感知生成对抗网络 - [论文](https://arxiv.org/pdf/2008.10298.pdf)
- 单样本虚拟试穿的正则化对抗训练, ICCVW 2019 - [论文](http://openaccess.thecvf.com/content_ICCVW_2019/html/CVFAD/Kikuchi_Regularized_Adversarial_Training_for_Single-Shot_Virtual_Try-On_ICCVW_2019_paper.html)
- 基于生成对抗网络的解纠缠妆容迁移 - [论文](https://arxiv.org/pdf/1907.01144v1.pdf)
- PIVTONS：基于条件图像补全的姿态不变性虚拟试鞋, ACCV 2018 - [论文](https://winstonhsu.info/wp-content/uploads/2018/09/chou18PIVTONS.pdf)
- 使用头部 3D 模型进行眼镜虚拟试戴 - [论文](https://dl.acm.org/doi/pdf/10.1145/2087756.2087838)
- 一种用于虚拟眼镜试戴的混合现实系统 - [论文](https://dl.acm.org/doi/pdf/10.1145/2087756.2087816)
- 基于 RGB - D 相机的增强现实鞋类个性化虚拟试穿系统 - [论文](https://www.sciencedirect.com/science/article/abs/pii/S0278612514000594)

## 姿态引导的人体合成<a name='pose-guided-human-synthesis'></a>

- 用于姿态引导人物图像合成的粗到细潜在扩散方法, CVPR 2024 - [代码](https://github.com/YanzuoLu/CFLD)
- IMAGPose：用于姿态引导人物生成的统一条件框架, NeurIPS 2024 - [论文](https://openreview.net/forum?id=6IyYa4gETN), [项目](https://github.com/muzishen/IMAGPose)
- 通过渐进式条件扩散模型推进姿态引导图像合成, ICLR 2024 - [论文](https://openreview.net/pdf?id=rHzapPnCgT), [项目](https://github.com/tencent-ailab/PCDMs)
- VGFlow：用于人体姿态调整的可见性引导流网络, CVPR 2023 - [论文](https://openaccess.thecvf.com/content/CVPR2023/html/Jain_VGFlow_Visibility_Guided_Flow_Network_for_Human_Reposing_CVPR_2023_paper.html), [项目](https://mdsrlab.github.io/2023/03/17/VGFlow-CVPR.html)
- 基于交叉注意力的风格分布用于可控人物图像合成, ECCV 2022 - [论文](https://arxiv.org/pdf/2208.00712.pdf)
- 用于全身图像生成的 InsetGAN, CVPR 2022. - [论文](https://arxiv.org/abs/2203.07293), [项目](http://afruehstueck.github.io/insetgan)
- 通过观看舞蹈视频实现自然场景下的着装合成, CVPR 2022 - [论文](https://arxiv.org/abs/2203.15320), [项目](https://awesome-wflow.github.io/)
- 有风格的姿态：基于条件风格生成对抗网络的保留细节的姿态引导图像合成, SIGGRAPH Asia 2021 - [论文](https://arxiv.org/abs/2109.06166), [项目](https://pose-with-style.github.io/)
- PoNA：用于人体姿态迁移的姿态引导非局部注意力, IEEE T-IP 2020 - [论文](https://ieeexplore.ieee.org/abstract/document/9222550)
- 通过渐进式训练实现姿态引导的高分辨率外观迁移 - [论文](https://arxiv.org/pdf/2008.11898.pdf)
- 随心所欲地重新捕捉 - [论文](https://arxiv.org/pdf/2006.01435.pdf)
- 使用感知外观的姿态风格化器生成人物图像, IJCAI 2020 - [论文](https://arxiv.org/pdf/2007.09077.pdf), [代码](https://github.com/siyuhuang/PoseStylizer)
- 基于属性分解生成对抗网络的可控人物图像合成, CVPR 2020 - [论文](https://arxiv.org/pdf/2003.12267.pdf), [代码](https://github.com/menyifang/ADGAN)
- 用于人物图像生成的深度图像空间变换, CVPR 2020 - [论文](https://arxiv.org/pdf/2003.00696v2.pdf), [代码](https://github.com/RenYurui/Global-Flow-Local-Attention)
- 通过空间自适应实例归一化实现神经姿态迁移, CVPR 2020 - [论文](https://arxiv.org/pdf/2003.07254v2.pdf), [代码](https://github.com/jiashunwang/Neural-Pose-Transfer)
- 基于双向特征变换的引导式图像到图像翻译, ICCV 2019 - [论文](https://arxiv.org/pdf/1910.11328v1.pdf), [代码](https://github.com/vt-vl-lab/Guided-pix2pix)
- 流体变形生成对抗网络：用于人体动作模仿、外观迁移和新视角合成的统一框架, ICCV 2019 - [论文](https://arxiv.org/pdf/1909.12224.pdf), [代码](https://github.com/svip-lab/impersonator)
- ClothFlow：基于流的着装人物生成模型, ICCV 2019 - [论文](http://openaccess.thecvf.com/content_ICCV_2019/html/Han_ClothFlow_A_Flow-Based_Model_for_Clothed_Person_Generation_ICCV_2019_paper.html)
- 用于人物图像生成的渐进式姿态注意力, CVPR 2019 - [论文](https://arxiv.org/pdf/1904.03349.pdf), [代码](https://github.com/tengteng95/Pose-Transfer)
- 用于人体姿态迁移的密集内在外观流, CVPR 2019 - [论文](https://arxiv.org/pdf/1903.11326v1.pdf), [代码](https://github.com/ly015/intrinsic_flow)
- 基于语义解析变换的无监督人物图像生成, CVPR 2019, TPAMI 2020 - [论文](https://arxiv.org/pdf/1904.03379.pdf), [代码](https://github.com/SijieSong/person_generation_spt)
- 使用深度生成模型进行姿态引导的时尚图像合成 - [论文](https://arxiv.org/pdf/1906.07251.pdf)
- 合成未见姿态下的人物图像, CVPR 2018 - [论文](https://arxiv.org/pdf/1804.07739.pdf), [代码](https://github.com/balakg/posewarp-cvpr2018)
- 用于姿态引导人物图像合成的软门控变形生成对抗网络, NeurIPS 2018 - [论文](https://arxiv.org/pdf/1810.11610.pdf)
- 用于基于姿态的人物图像生成的可变形生成对抗网络, CVPR 2018 - [论文](https://openaccess.thecvf.com/content_cvpr_2018/papers/Siarohin_Deformable_GANs_for_CVPR_2018_paper.pdf)
- 用于人物重识别的姿态归一化图像生成, ECCV 2018 - [论文](https://arxiv.org/pdf/1712.02225.pdf)
- 解纠缠的人物图像生成, CVPR 2018 - [论文/代码/数据](https://homes.esat.kuleuven.be/~liqianma/CVPR18_DPIG/index.html)
- 用于条件外观和形状生成的变分 U 型网络, CVPR 2018 - [论文](https://arxiv.org/pdf/1804.04694.pdf), [代码](https://github.com/CompVis/vunet)
- 人体外观迁移, CVPR 2018 - [论文](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zanfir_Human_Appearance_Transfer_CVPR_2018_paper.pdf)
- 姿态引导的人物图像生成, NeurIPS 2017 - [论文](https://arxiv.org/pdf/1705.09368.pdf), [代码](https://github.com/charliememory/Pose-Guided-Person-Image-Generation)

## 虚拟试穿数据集<a name='Datasets-for-Virtual-Try-on'></a>

- StreetTryOn - [下载](https://github.com/cuiaiyu/street-tryon-benchmark)
- CLOTH4D, CVPR 2023 - [下载](https://github.com/AemikaChow/CLOTH4D)
- DressCode - [下载](https://docs.google.com/forms/d/e/1FAIpQLSeWVzxWcj3JSALtthuw-2QDAbf2ymiK37sA4pRQD4tZz2vqsw/viewform), [论文](https://arxiv.org/pdf/2204.08532.pdf)
- VITON-HD - [下载](https://www.dropbox.com/s/10bfat0kg4si1bu/zalando-hd-resized.zip?dl=0), [项目](https://psh01087.github.io/VITON-HD/)
- VITON - [下载](https://drive.google.com/file/d/1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo/view), [论文](http://openaccess.thecvf.com/content_cvpr_2018/papers/Han_VITON_An_Image-Based_CVPR_2018_paper.pdf)
- MPV - [下载](https://drive.google.com/drive/folders/1e3ThRpSj8j9PaCUw8IrqzKPDVJK_grcA), [论文](https://arxiv.org/abs/1902.11026)
- Deep Fashion3D - [论文](https://arxiv.org/abs/2003.12753)
- DeepFashion MultiModal - [下载](https://github.com/yumingj/DeepFashion-MultiModal)
- Digital Wardrobe - [下载/论文/项目](http://virtualhumans.mpi-inf.mpg.de/mgn/)
- TailorNet Dataset - [下载](https://github.com/zycliao/TailorNet_dataset), [项目](http://virtualhumans.mpi-inf.mpg.de/tailornet/)
- CLOTH3D - [论文](https://arxiv.org/abs/1912.02792)
- 3DPeople - [项目](https://www.albertpumarola.com/research/3DPeople/index.html)
- THUman Dataset - [项目](http://www.liuyebin.com/deephuman/deephuman.html)
- Garment Dataset, Wang et al. 2018 - [项目](http://geometry.cs.ucl.ac.uk/projects/2018/garment_design/)

## 相关会议研讨会<a name='Related-Conference-Workshops'></a>

- 虚拟试穿研讨会: [CVPR 2024](https://vto-cvpr24.github.io/)
- 时尚、艺术与设计领域的计算机视觉研讨会: [CVPR 2024](https://sites.google.com/view/cvfad2024/home), [CVPR 2023](https://sites.google.com/view/cvfad2023/home), [CVPR 2022](https://sites.google.com/view/cvfad2022/home), [CVPR 2021](https://sites.google.com/zalando.de/cvfad2021/home), [CVPR 2020](https://sites.google.com/view/cvcreative2020), [ICCV 2019](https://sites.google.com/view/cvcreative), [ECCV 2018](https://sites.google.com/view/eccvfashion)
- 迈向以人为本的图像/视频合成研讨会: [CVPR 2020](https://vuhcs.github.io/), [CVPR 2019](https://vuhcs.github.io/vuhcs-2019/index.html)

## 示例<a name='demos'></a>

- 任意场景下的随心穿搭 - [示例](https://huggingface.co/spaces/selfit-camera/OutfitAnyone-in-the-Wild), [文档](https://github.com/selfitcamera/Outfit-Anyone-in-the-Wild)
- 随心搭配 - [示例](https://huggingface.co/spaces/HumanAIGC/OutfitAnyone), [文档](https://github.com/HumanAIGC/OutfitAnyone)
- Looklet 试衣间 [公司](https://looklet.com), [示例](https://dressing-room.looklet.com)
- TINT 平台：用于各类面部相关物品（妆容、眼镜、耳环、首饰等）的虚拟试戴 [公司](https://www.banuba.com/solutions/e-commerce/virtual-try-on), [示例](https://banuba.com/solutions/e-commerce/virtual-makeup-demo/).

## 相关仓库<a name='related-repositories'></a>

- [awesome-fashion-ai](https://github.com/ayushidalmia/awesome-fashion-ai)
- [Cool Fashion Papers](https://github.com/lzhbrian/Cool-Fashion-Papers)
- [Clothes-3D](https://github.com/lzhbrian/Clothes-3D)
- [Awesome 3D Human](https://github.com/lijiaman/awesome-3d-human)
- [Awesome 3D reconstruction list](https://github.com/openMVG/awesome_3DReconstruction_list)
- [Human Body Reconstruction](https://github.com/chenweikai/Body_Reconstruction_References)
- [Awesome 3D Body Papers](https://github.com/3DFaceBody/awesome-3dbody-papers)
- [Awesome pose transfer](https://github.com/Zhangjinso/Awesome-pose-transfer)

#### 欢迎 PR
