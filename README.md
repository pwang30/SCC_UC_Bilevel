# Short-circuit current + UC + Bi-level: Case studies for examining market power

**PLEASE NOTE, the models and methodologies that this work is based on were introduced in the following papers. Fully understanding these is the foundation of our work:**
- Short-Circuit Current (SCC) models refer to:
  1. Chu, Zhongda, and Fei Teng. ["Short circuit current constrained UC in high IBG-penetrated power systems." IEEE Transactions on Power Systems 36.4 (2021): 3776-3785.](https://ieeexplore.ieee.org/abstract/document/9329077)
  2. Chu, Zhongda, Jingyi Wu, and Fei Teng. ["Pricing of short circuit current in high IBR-penetrated system." Electric Power Systems Research 235 (2024): 110690.](https://www.sciencedirect.com/science/article/pii/S0378779624005765)
- Primal-Dual formulation for addressing UC issues refer to:
  1. Ye, Yujian, et al. ["Incorporating non-convex operating characteristics into bi-level optimization electricity market models." IEEE Transactions on Power Systems 35.1 (2019): 163-176.](https://ieeexplore.ieee.org/abstract/document/8746573)

**GUIDANCE abot how to use the code of our work**

The work is mainly made of two parts:
1. Modelling of SCC.
2. Modelling of primal-dual formulation.

We try to guide you to understand our logistics of coding, once you fully understand, then analyze any power systems you want.
- For the code of SCC modelling, please refer to the files named "_admittance_matrix_calculation.jl_", "_dataset_gene.jl_" and "_offline_trainning.jl_" in foler called "Model_SCL".

  1. "_admittance_matrix_calculation.jl_" calculates the impedance of transmission lines of the system, easy to follow.

  2. "_dataset_gene.jl_" generates the data for classification, i.e., the offline trainning process mentioned in the second chapter of our paper. The subfunction "_admittance_matrix_calculation.jl_" is called here to obtain the transmissin line admittance matrix which is combined with the generators' admittance matrix (refer to line 76 in "_dataset_gene.jl_", from line 79-86, it is the equation of actual, exact SCC representation). The remainder of code is generating all possible UC status pairs of generators. The matrix "I_SCC_all_buses_scenarios" are the SCC corresponding to "matrix_ω" (storing UC status and capacity factor of IBR, comprising all possible scenarios).

  3. "_offline_trainning.jl_" is the trainning process, with inputting parameters from above subfunctions.

- For the code of primal-dual modelling, i.e., the final bi-level model running, please refer to the files named "_bi_level_SCL_Local_Bus2.jl_", "_bi_level_SCL_distant.jl_" in foler called "Strategic_bidding".

  1. As we have numerous scenarios, here we only publish the case where two strategic generators located in bus 2 and the case where three strategic generators located in bus 2 and bus 27. Other cases are easy to be simulated by following a similar coding structure.
  2. As for the pricing of SCC service by the single-level formulation, please do it by readers yourselves, however we still give the marginal bid we used in the paper (the file named "AS_offer_price.mat"). 

- In the practical simulation, please put all files in one folder, e.g., "_bi_level_SCL_Local_Bus2.jl_", "_More_two_various_21_271_272.jl_", "_admittance_matrix_calculation.jl_", "_dataset_gene.jl_", "_offline_trainning.jl_", "Linespara" and "AS_offer_price.mat". Then run the main files "_bi_level_SCL_Local_Bus2.jl_" or "_More_two_various_21_271_272.jl_".

**NOTE the full form of SCC constraints is given in "_SCC_equation.jl_" for readers who have interests.**

----

If you find something helpful or use this code for your own work, please cite [this paper](https://arxiv.org/abs/2508.09425):
<ol>
      Peng Wang and Luis Badesa, "<b>Imperfect Competition in Markets for Short-Circuit Current Services</b>," arXiv preprint arXiv:2508.09425 (2025).
</ol>
      <br>
      
<ol> 
@article{wang2025imperfect, <br>
  title={Imperfect Competition in Markets for Short-Circuit Current Services}, <br>
  author={Wang, Peng and Badesa, Luis}, <br>
  journal={arXiv preprint arXiv:2508.09425}, <br>
  year={2025} <br>
}

----

This work was supported by MICIU/AEI/10.13039/501100011033 and ERDF/EU under grant PID2023-150401OA-C22, as well as by the Madrid Government (Comunidad de Madrid-Spain) under the Multiannual Agreement 2023-2026 with Universidad Politécnica de Madrid, ``Line A - Emerging PIs''. The work of Peng Wang was also supported by China Scholarship Council under grant 202408500065.

<figure style="display:inline-block; margin:10px; text-align:center;">
   <img src="./logos/MICIU+Cofinanciado+AEI.jpg" style="width:600px; height:140px; object-fit:contain; display:block;">
</figure>
<figure style="display:inline-block; margin:10px; text-align:center;">
   <img src="./logos/Logo_CM.png" style="width:200px; height:160px; object-fit:contain; display:block;">
</figure>
