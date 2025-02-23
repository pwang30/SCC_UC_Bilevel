# SCL_UC_Bilevel
**This is a test.**

Here is the road I chose to do this research:
Start with easy parts, it means let's first figure out simple and clear concepts and build the whole model step by step. For me, I started with a single-level model with UC and SCL constraints, considering a system only has 3 buses that meet a basic economic dispatch framework. Then, I escalated it to be a modified IEEE-30 bus system proposed in paper by Chongda (Feb,19,2025. Madrid). Finally, I will dive into how to insert bilevel in to the current model, which is my final goal.


This code solves a **SCL_UC_Bilevel**.

The main elements of this work include:
- [Stackelberg game](https://en.wikipedia.org/wiki/Stackelberg_competition)
- [Bilevel model](https://en.wikipedia.org/wiki/Bilevel_optimization)
- [Big-M method](https://en.wikipedia.org/wiki/Big_M_method)
- [Karush–Kuhn–Tucker conditions](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions)
- [Distribution system operator (DSO)](https://www.camus.energy/blog/what-is-a-distribution-system-operator)
- [Virtual power plant (VPP)](https://en.wikipedia.org/wiki/Virtual_power_plant)
- [Julia programming](https://julialang.org/)


The main work includes:
- Model building and solving: In the Bilevel model, DSO maximizes profits in the upper level, which requires the Big-M method to build its model. VPPs take the prices set by DSO and minimize their respective operating costs in the lower level, and they are converted into KKT conditions for further solution. DSO and VPPs are in a Stackelberg game relationship.
- Transaction behavior analysis: The selfish transaction behaviors of DSO and VPPs are analyzed, which highlight the role of DSO in trading. Their respective profits or costs have been improved, and the explicit benefits of non-strategic player have been compromised, but the overall (whole system or market) benefits are reserved, only the distribution of benefits has changed.


The following is the program running process (assuming it is in Visual Studio Code):
- Step 1: How to install [Julia in VsCode](https://code.visualstudio.com/docs/languages/julia).
- Step 2: In the comments of this code, there are necessary package installation instructions, just copy and paste them.
- Step 3: Run the program.

**Note that this work can be easily extended to larger-scale problems because of the structural independence of each agent, such as those involving more
aggregators and DER, and market mechanisms (demand response, time-of-use tariff, etc.).**

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
