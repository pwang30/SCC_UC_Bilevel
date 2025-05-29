# Short-circuit level+UC+Bi-level:Case studies for examining market power

**PLEASE NOTE, the main models and methodologies are in the listed papers here. Fully understanding these works is the foundation of our work**
- Short-Circuit Level (SCL) models refer to:
1. Chu, Zhongda, and Fei Teng. ["Short circuit current constrained UC in high IBG-penetrated power systems." IEEE Transactions on Power Systems 36.4 (2021): 3776-3785.](https://ieeexplore.ieee.org/abstract/document/9329077)
2. Chu, Zhongda, Jingyi Wu, and Fei Teng. ["Pricing of short circuit current in high IBR-penetrated system." Electric Power Systems Research 235 (2024): 110690.](https://www.sciencedirect.com/science/article/pii/S0378779624005765)
- Primal-Dual formulation for addressing UC issues refer to:
1. Ye, Yujian, et al. ["Incorporating non-convex operating characteristics into bi-level optimization electricity market models." IEEE Transactions on Power Systems 35.1 (2019): 163-176.](https://ieeexplore.ieee.org/abstract/document/8746573)

**GUIDANCE abot how to use the code of our work**

The work is mainly made of two parts:
1. Modelling of SCL.
2. Modelling of primal-dual formulation.

We try to guide you to understand our logistics of coding, once you fully understand, then analyze any power systems you want.
- For the code of SCL modelling, please refer to the files named "_admittance_matrix_calculation.jl_", "_dataset_gene.jl_" and "_offline_trainning.jl_" in foler called "Model_SCL".

  "_admittance_matrix_calculation.jl_" calculates the IMPEDANCE of transmission lines of the system, easy to follow.

  "_dataset_gene.jl_" generates the data for classification, i.e., the offline trainning process mentioned in the second chapter of our paper. The subfunction "_admittance_matrix_calculation.jl_" is called here to obtain the transmissin line admittance matrix which is combined with the generators' admittance matrix (refer to line 76 in the code). Code from line 79-86 is the equation of actual, exact SCL representation. The remainder of code is generating all possible UC status pairs of generators. The matrix "I_SCC_all_buses_scenarios" are the SCL corresponding to "matrix_ω" (storing UC status and capacity factor of IBR, comprising all possible scenarios).

  "_offline_trainning.jl_" is the trainning process, with inputting parameters from above subfunctions.

- For the code of primal-dual modelling, i.e., the final bi-level model running, as we have several scenarios, here we only publich lodal strategic bidding in Bus 2 and distant strategic bidding in Bus 27 and Bus 4. Other cases are easy to be simulated by following a similar coding structure. As for the pricing of SCL service by the single-level formulation, please do it by readers yourselves, however we still give the offer we used in the paper (the file named "AS_offer_price.mat"). 

- In the practical simulation, please put all files in one folder, e.g., "_bi_level_SCL_Local_Bus2.jl_", "_bi_level_SCL_distant.jl_", "_admittance_matrix_calculation.jl_", "_dataset_gene.jl_", "_offline_trainning.jl_" and "_Linespara_". Then run the main file "_bi_level_SCL_Local_Bus2.jl_" or "_bi_level_SCL_distant.jl_". 
----

If you find something helpful or use this code for your own work, please cite this paper:
<ol>
      Wang, Peng, Xi Zhang, and Luis Badesa. "Analyzing the Role of the DSO in Electricity Trading of VPPs via a Stackelberg Game Model." arXiv preprint arXiv:2501.07715 (2025).
</ol>
      <br>
      
<ol> 
@misc{wang2025analyzingroledsoelectricity, <br>
      title={Analyzing the Role of the DSO in Electricity Trading of VPPs via a Stackelberg Game Model}, <br>
      author={Peng Wang and Xi Zhang and Luis Badesa},<br>
      year={2025},<br>
      eprint={2501.07715},<br>
      archivePrefix={arXiv},<br>
      primaryClass={eess.SY},<br>
