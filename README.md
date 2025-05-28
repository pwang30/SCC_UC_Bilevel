# Short-circuit level+UC+Bi-level:Case studies for examining market power

**PLEASE NOTE, the main models and methodologies are in the listed papers here. Fully understanding these works is the foundation of our work**
- Short-Circuit Level (SCL) models refer to:
1. Chu, Zhongda, and Fei Teng. ["Short circuit current constrained UC in high IBG-penetrated power systems." IEEE Transactions on Power Systems 36.4 (2021): 3776-3785.](https://ieeexplore.ieee.org/abstract/document/9329077)
2. Chu, Zhongda, Jingyi Wu, and Fei Teng. ["Pricing of short circuit current in high IBR-penetrated system." Electric Power Systems Research 235 (2024): 110690.](https://www.sciencedirect.com/science/article/pii/S0378779624005765)
- Primal-Dual formulation for addressing UC issues refer to:
1. Ye, Yujian, et al. ["Incorporating non-convex operating characteristics into bi-level optimization electricity market models." IEEE Transactions on Power Systems 35.1 (2019): 163-176.](https://ieeexplore.ieee.org/abstract/document/8746573)

**EXPLANATION abot how to use the code of our work**

The work is mainly made of two parts:
1. Modelling of SCL.
2. Modelling of primal-dual formulation.

We try to guide you to understand our logistics of coding, once you fully understand, then analyze any power systems you like.
- For the code of SCL modelling, please refer to the file named "_admittance_matrix_calculation.jl_", "_dataset_gene.jl_" and "_offline_trainning.jl_".
  -"_admittance_matrix_calculation.jl_" calculates the IMPEDANCE of transmission lines

The following is the program running process (assuming it is in Visual Studio Code):
- Step 1: How to install [Julia in VsCode](https://code.visualstudio.com/docs/languages/julia).
- Step 2: In the comments of this code, there are necessary package installation instructions, just copy and paste them.
- Step 3: Run the program.

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
