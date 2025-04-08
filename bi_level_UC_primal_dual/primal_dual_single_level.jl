# Author: Peng Wang       from Technical University of Madrid (UPM)
# Supervisor: Luis Badesa

# Everyone in this setting is competitive and trustful. Solving by  [primal LL- dual LL] 
# 05.April.2025

import Pkg
using JuMP,Gurobi, CSV,DataFrames,LinearAlgebra, XLSX, IterTools, DelimitedFiles,Plots,CPLEX, MAT
include("dataset_gene.jl")
include("offline_trainning.jl")
include("admittance_matrix_calculation.jl") 
# SGs, buses:2,3,4,5,27,30    IBRs, buses:1,23,26



#-----------------------------------Define Parameters for Calculating SCL-----------------------------------
I_IBG=1      # pre-defined SCL contribution from IBG
Iₗᵢₘ= 5       # SCL limit
β=0.95       # percentage of nominal voltage, range from 0.95-1.1
v_n=1        # nominal voltage
v=0.1        # gap for classification 

I_SCC_all_buses_scenarios, matrix_ω =dataset_gene(I_IBG, β,v_n)                                                            # data set generation                      
K_g, K_c, K_m, N_type_1, N_type_2, err_type_1, err_type_2= offline_trainning(I_SCC_all_buses_scenarios, matrix_ω, Iₗᵢₘ, v)  # offline_trainning



#-----------------------------------Define Parameters for Optimization-----------------------------------
IBG₁=[2,1.5,1.6,1.8,1.3,0.6,2.8,3.3,3.9,4,3.3,2.9,2.7,2,0.2,3.2,5.1,3.1,1.8,2,1.3,1,2,3.8]*10^3                    # Wind_bus_1
IBG₂₃=[4.7,5.1,4.3,4.1,3.8,3.9,4,5,5,4.8,3.9,4.3,5,5.2,5.8,5.6,1.6,0.9,5.8,4.1,3.6,3.5,3.1,3.8]*10^3               # Wind_bus_23
IBG₂₆=[9.3,10.1,7.2,7.5,7.9,6.4,7.1,6.9,5.6,5.4,5.2,4,3.8,3,2.8,3.2,2.5,1.1,2.1,2.9,2.7,3,4.6,5.5]*10^3            # Wind_bus_26

Load_total=[18.42,17.95,18.29,18.51,18.13,17.88,19.46,21.97,23.17,23.87,
23.91,23.77,23.80,23.82,24.23,23.79,26.01,26.91,25.26,23.69,22.12,20.04,18.17,18.01]*10^3*1.2   # (MW)
T=length(Load_total)

Pˢᴳₘₐₓ=[6.584, 5.760, 3.781, 3.335, 3.252, 2.880]*10^3     #   Max generation of SGs                    SGs, buses:2,3,4,5,27,30
Pˢᴳₘᵢₙ=[3.292, 2.880, 1.512, 0.667, 0.650, 0.288]*10^3      #   Min generation of SGs                   SGs, buses:2,3,4,5,27,30
Rₘₐₓ=[1.317, 1.152, 1.512, 1.334, 1.951, 1.728]*10^3        #   Ramp limits of SGs                      SGs, buses:2,3,4,5,27,30
Kˢᵗ=[4000, 325, 142.5, 72, 55, 31]*10^3                     #   Startup cost of SGs                     SGs, buses:2,3,4,5,27,30
Kˢʰ=[800, 28.5, 18.5, 14.4, 12, 10]*10^3                    #   Shutdown cost of SGs                    SGs, buses:2,3,4,5,27,30
Oᵐ₁=[6.20, 32.10, 36.47, 64.28, 84.53, 97.36]               #   Marginal generation cost of SG 1  in    SGs, buses:2,3,4,5,27,30
Oᵐ₂=[7.07, 34.72, 38.49, 72.84, 93.60, 105.02]              #   Marginal generation cost of SG 2  in    SGs, buses:2,3,4,5,27,30
Oⁿˡ=[18.431, 17.005, 13.755, 9.930, 9.900, 8.570]*10^3      #   No-load cost of SGs                     SGs, buses:2,3,4,5,27,30
Oᴱ_c=[2.46 3.19 2.86]                                       #   Price offer of SGs                      IBRs, buses:1,23,26
P_g₀=[5.268 4.608 3.025 2.668 2.602 0]*10^3                 #   Initial generation (t=0) of SGs         SGs, buses:2,3,4,5,27,30
yˢᴳ₀=[1 1 1 1 1 0]                                          #   Initial on/off status (t=0) of SGs      SGs, buses:2,3,4,5,27,30



#-----------------------------------Define Primal-Dual Model-----------------------------------
model= Model()

#-------Define Primal Variales
@variable(model, Pˢᴳ²_1[1:T])        # generation of SGs , buses:2,3,4,5,27,30.  Include strategic and competitive players
@variable(model, Pˢᴳ²_2[1:T])  
@variable(model, Pˢᴳ³_1[1:T])     
@variable(model, Pˢᴳ³_2[1:T])                
@variable(model, Pˢᴳ⁴_1[1:T])   
@variable(model, Pˢᴳ⁴_2[1:T])             
@variable(model, Pˢᴳ⁵_1[1:T])    
@variable(model, Pˢᴳ⁵_2[1:T])            
@variable(model, Pˢᴳ²⁷_1[1:T]) 
@variable(model, Pˢᴳ²⁷_2[1:T])                
@variable(model, Pˢᴳ³⁰_1[1:T])                
@variable(model, Pˢᴳ³⁰_2[1:T])  

@variable(model, Pᴵᴮᴳ¹[1:T]>=0)         # generation of IBRs (WT) , buses:1, 23, 26 , dual variables: ζᵐⁱⁿₜ  
@variable(model, Pᴵᴮᴳ²³[1:T]>=0)              
@variable(model, Pᴵᴮᴳ²⁶[1:T]>=0)  

@variable(model, yˢᴳ²_1[1:T],Bin)      # status of SGs, buses:2,3,4,5,27,30.  Include strategic and competitive players
@variable(model, yˢᴳ²_2[1:T],Bin)          
@variable(model, yˢᴳ³_1[1:T],Bin)  
@variable(model, yˢᴳ³_2[1:T],Bin)          
@variable(model, yˢᴳ⁴_1[1:T],Bin)      
@variable(model, yˢᴳ⁴_2[1:T],Bin)      
@variable(model, yˢᴳ⁵_1[1:T],Bin)      
@variable(model, yˢᴳ⁵_2[1:T],Bin)      
@variable(model, yˢᴳ²⁷_1[1:T],Bin)  
@variable(model, yˢᴳ²⁷_2[1:T],Bin)          
@variable(model, yˢᴳ³⁰_1[1:T],Bin)   
@variable(model, yˢᴳ³⁰_2[1:T],Bin)                       

@variable(model, α₁[1:T]>=0)                   # percentage of IBGs' online capacity , dual variables:   φᵐⁱⁿₜ
@variable(model, α₂₃[1:T]>=0)
@variable(model, α₂₆[1:T]>=0)

@variable(model, Cᵁ²_1[1:T]>=0)                 # startup costs and shutdown costs for SGs , dual variables: ρˢᵗₜ , ρˢʰₜ
@variable(model, Cᵁ²_2[1:T]>=0)                 
@variable(model, Cᴰ²_1[1:T]>=0)                 
@variable(model, Cᴰ²_2[1:T]>=0)                 
@variable(model, Cᵁ³_1[1:T]>=0)    
@variable(model, Cᵁ³_2[1:T]>=0)             
@variable(model, Cᴰ³_1[1:T]>=0)  
@variable(model, Cᴰ³_2[1:T]>=0)               
@variable(model, Cᵁ⁴_1[1:T]>=0)     
@variable(model, Cᵁ⁴_2[1:T]>=0)            
@variable(model, Cᴰ⁴_1[1:T]>=0)   
@variable(model, Cᴰ⁴_2[1:T]>=0)              
@variable(model, Cᵁ⁵_1[1:T]>=0)  
@variable(model, Cᵁ⁵_2[1:T]>=0)               
@variable(model, Cᴰ⁵_1[1:T]>=0) 
@variable(model, Cᴰ⁵_2[1:T]>=0)                
@variable(model, Cᵁ²⁷_1[1:T]>=0)    
@variable(model, Cᵁ²⁷_2[1:T]>=0)             
@variable(model, Cᴰ²⁷_1[1:T]>=0)    
@variable(model, Cᴰ²⁷_2[1:T]>=0)             
@variable(model, Cᵁ³⁰_1[1:T]>=0)      
@variable(model, Cᵁ³⁰_2[1:T]>=0)            
@variable(model, Cᴰ³⁰_1[1:T]>=0)    
@variable(model, Cᴰ³⁰_2[1:T]>=0)            

@variable(model, I_₂₆[1:T])         # define SCL on bus 26,29,30 , dual variables: λ_₂₆, λ_₂₉, λ_₃₀, λ_lim₂₆, λ_lim₂₉, λ_lim₃₀
@variable(model, I_₂₉[1:T])
@variable(model, I_₃₀[1:T])

#-------Define Dual Variales
@variable(model, λᴱ[1:T]>=0)      # price for market clearing                
@variable(model, λ_F[1:3,1:T]>=0)    # price for SCL AS      n=1 for bus 26 ; n=2 for bus 29  ; n=3 for bus 30 

@variable(model, ζᵐᵃˣ¹[1:T]>=0) 
@variable(model, ζᵐᵃˣ²³[1:T]>=0)
@variable(model, ζᵐᵃˣ²⁶[1:T]>=0)

@variable(model, φᵐᵃˣ¹[1:T]>=0)
@variable(model, φᵐᵃˣ²³[1:T]>=0)
@variable(model, φᵐᵃˣ²⁶[1:T]>=0)

@variable(model, μᵐⁱⁿˢᴳ²_1[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ²_1[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ²_2[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ²_2[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ³_1[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ³_1[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ³_2[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ³_2[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ⁴_1[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ⁴_1[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ⁴_2[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ⁴_2[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ⁵_1[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ⁵_1[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ⁵_2[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ⁵_2[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ²⁷_1[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ²⁷_1[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ²⁷_2[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ²⁷_2[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ³⁰_1[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ³⁰_1[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ³⁰_2[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ³⁰_2[1:T]>=0)

@variable(model, πʳᵈˢᴳ²_1[1:T]>=0)
@variable(model, πʳᵘˢᴳ²_1[1:T]>=0)
@variable(model, πʳᵈˢᴳ²_2[1:T]>=0)
@variable(model, πʳᵘˢᴳ²_2[1:T]>=0)
@variable(model, πʳᵈˢᴳ³_1[1:T]>=0)
@variable(model, πʳᵘˢᴳ³_1[1:T]>=0)
@variable(model, πʳᵈˢᴳ³_2[1:T]>=0)
@variable(model, πʳᵘˢᴳ³_2[1:T]>=0)
@variable(model, πʳᵈˢᴳ⁴_1[1:T]>=0)
@variable(model, πʳᵘˢᴳ⁴_1[1:T]>=0)
@variable(model, πʳᵈˢᴳ⁴_2[1:T]>=0)
@variable(model, πʳᵘˢᴳ⁴_2[1:T]>=0)
@variable(model, πʳᵈˢᴳ⁵_1[1:T]>=0)
@variable(model, πʳᵘˢᴳ⁵_1[1:T]>=0)
@variable(model, πʳᵈˢᴳ⁵_2[1:T]>=0)
@variable(model, πʳᵘˢᴳ⁵_2[1:T]>=0)
@variable(model, πʳᵈˢᴳ²⁷_1[1:T]>=0)
@variable(model, πʳᵘˢᴳ²⁷_1[1:T]>=0)
@variable(model, πʳᵈˢᴳ²⁷_2[1:T]>=0)
@variable(model, πʳᵘˢᴳ²⁷_2[1:T]>=0)
@variable(model, πʳᵈˢᴳ³⁰_1[1:T]>=0)
@variable(model, πʳᵘˢᴳ³⁰_1[1:T]>=0)
@variable(model, πʳᵈˢᴳ³⁰_2[1:T]>=0)
@variable(model, πʳᵘˢᴳ³⁰_2[1:T]>=0)

@variable(model, σˢᵗˢᴳ²_1[1:T]>=0)
@variable(model, σˢʰˢᴳ²_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ²_2[1:T]>=0)
@variable(model, σˢʰˢᴳ²_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ³_1[1:T]>=0)
@variable(model, σˢʰˢᴳ³_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ³_2[1:T]>=0)
@variable(model, σˢʰˢᴳ³_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ⁴_1[1:T]>=0)
@variable(model, σˢʰˢᴳ⁴_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ⁴_2[1:T]>=0)
@variable(model, σˢʰˢᴳ⁴_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ⁵_1[1:T]>=0)
@variable(model, σˢʰˢᴳ⁵_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ⁵_2[1:T]>=0)
@variable(model, σˢʰˢᴳ⁵_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ²⁷_1[1:T]>=0)
@variable(model, σˢʰˢᴳ²⁷_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ²⁷_2[1:T]>=0)
@variable(model, σˢʰˢᴳ²⁷_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ³⁰_1[1:T]>=0)
@variable(model, σˢʰˢᴳ³⁰_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ³⁰_2[1:T]>=0)
@variable(model, σˢʰˢᴳ³⁰_2[1:T]>=0)

@variable(model, ψᵐᵃˣˢᴳ²_1[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ²_2[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ³_1[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ³_2[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ⁴_1[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ⁴_2[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ⁵_1[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ⁵_2[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ²⁷_1[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ²⁷_2[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ³⁰_1[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ³⁰_2[1:T]>=0)



#-------Define Primal Constraints
@constraint(model, Pˢᴳ²_1+Pˢᴳ²_2+Pˢᴳ³_1+Pˢᴳ³_2+Pˢᴳ⁴_1+Pˢᴳ⁴_2+Pˢᴳ⁵_1+Pˢᴳ⁵_2+Pˢᴳ²⁷_1+Pˢᴳ²⁷_2+Pˢᴳ³⁰_1+Pˢᴳ³⁰_2+
                   Pᴵᴮᴳ¹+Pᴵᴮᴳ²³+Pᴵᴮᴳ²⁶==Load_total)     # power balance , dual variable: λᴱₜ

@constraint(model, Pˢᴳ²_1.<=yˢᴳ²_1*Pˢᴳₘₐₓ[1])           # bounds for the output of SGs with UC , dual variables: μᵐⁱⁿₜ , μᵐᵃˣₜ
@constraint(model, yˢᴳ²_1*Pˢᴳₘᵢₙ[1].<=Pˢᴳ²_1)
@constraint(model, Pˢᴳ²_2.<=yˢᴳ²_2*Pˢᴳₘₐₓ[1])       
@constraint(model, yˢᴳ²_2*Pˢᴳₘᵢₙ[1].<=Pˢᴳ²_2)             
@constraint(model, Pˢᴳ³_1.<=yˢᴳ³_1*Pˢᴳₘₐₓ[2])       
@constraint(model, yˢᴳ³_1*Pˢᴳₘᵢₙ[2].<=Pˢᴳ³_1)       
@constraint(model, Pˢᴳ³_2.<=yˢᴳ³_2*Pˢᴳₘₐₓ[2])       
@constraint(model, yˢᴳ³_2*Pˢᴳₘᵢₙ[2].<=Pˢᴳ³_2)     
@constraint(model, Pˢᴳ⁴_1.<=yˢᴳ⁴_1*Pˢᴳₘₐₓ[3])       
@constraint(model, yˢᴳ⁴_1*Pˢᴳₘᵢₙ[3].<=Pˢᴳ⁴_1)
@constraint(model, Pˢᴳ⁴_2.<=yˢᴳ⁴_2*Pˢᴳₘₐₓ[3])       
@constraint(model, yˢᴳ⁴_2*Pˢᴳₘᵢₙ[3].<=Pˢᴳ⁴_2)     
@constraint(model, Pˢᴳ⁵_1.<=yˢᴳ⁵_1*Pˢᴳₘₐₓ[4])       
@constraint(model, yˢᴳ⁵_1*Pˢᴳₘᵢₙ[4].<=Pˢᴳ⁵_1)
@constraint(model, Pˢᴳ⁵_2.<=yˢᴳ⁵_2*Pˢᴳₘₐₓ[4])       
@constraint(model, yˢᴳ⁵_2*Pˢᴳₘᵢₙ[4].<=Pˢᴳ⁵_2)
@constraint(model, Pˢᴳ²⁷_1.<=yˢᴳ²⁷_1*Pˢᴳₘₐₓ[5])       
@constraint(model, yˢᴳ²⁷_1*Pˢᴳₘᵢₙ[5].<=Pˢᴳ²⁷_1)
@constraint(model, Pˢᴳ²⁷_2.<=yˢᴳ²⁷_2*Pˢᴳₘₐₓ[5])       
@constraint(model, yˢᴳ²⁷_2*Pˢᴳₘᵢₙ[5].<=Pˢᴳ²⁷_2)
@constraint(model, Pˢᴳ³⁰_1.<=yˢᴳ³⁰_1*Pˢᴳₘₐₓ[6])       
@constraint(model, yˢᴳ³⁰_1*Pˢᴳₘᵢₙ[6].<=Pˢᴳ³⁰_1)
@constraint(model, Pˢᴳ³⁰_2.<=yˢᴳ³⁰_2*Pˢᴳₘₐₓ[6])       
@constraint(model, yˢᴳ³⁰_2*Pˢᴳₘᵢₙ[6].<=Pˢᴳ³⁰_2)

@constraint(model, Pˢᴳ²_1[1]-P_g₀[1]<=Rₘₐₓ[1])        # bounds for the ramp of SGs , dual variables: πʳᵈₜ , πʳᵘₜ
@constraint(model, -Rₘₐₓ[1]<=Pˢᴳ²_1[1]-P_g₀[1])  
@constraint(model, Pˢᴳ²_2[1]-P_g₀[1]<=Rₘₐₓ[1])        
@constraint(model, -Rₘₐₓ[1]<=Pˢᴳ²_2[1]-P_g₀[1]) 
@constraint(model, Pˢᴳ³_1[1]-P_g₀[2]<=Rₘₐₓ[2])
@constraint(model, -Rₘₐₓ[2]<=Pˢᴳ³_1[1]-P_g₀[2])
@constraint(model, Pˢᴳ³_2[1]-P_g₀[2]<=Rₘₐₓ[2])        
@constraint(model, -Rₘₐₓ[2]<=Pˢᴳ³_2[1]-P_g₀[2]) 
@constraint(model, Pˢᴳ⁴_1[1]-P_g₀[3]<=Rₘₐₓ[3])
@constraint(model, -Rₘₐₓ[3]<=Pˢᴳ⁴_1[1]-P_g₀[3])
@constraint(model, Pˢᴳ⁴_2[1]-P_g₀[3]<=Rₘₐₓ[3])
@constraint(model, -Rₘₐₓ[3]<=Pˢᴳ⁴_2[1]-P_g₀[3])
@constraint(model, Pˢᴳ⁵_1[1]-P_g₀[4]<=Rₘₐₓ[4])
@constraint(model, -Rₘₐₓ[4]<=Pˢᴳ⁵_1[1]-P_g₀[4])
@constraint(model, Pˢᴳ⁵_2[1]-P_g₀[4]<=Rₘₐₓ[4])
@constraint(model, -Rₘₐₓ[4]<=Pˢᴳ⁵_2[1]-P_g₀[4])
@constraint(model, Pˢᴳ²⁷_1[1]-P_g₀[5]<=Rₘₐₓ[5])
@constraint(model, -Rₘₐₓ[5]<=Pˢᴳ²⁷_1[1]-P_g₀[5])
@constraint(model, Pˢᴳ²⁷_2[1]-P_g₀[5]<=Rₘₐₓ[5])
@constraint(model, -Rₘₐₓ[5]<=Pˢᴳ²⁷_2[1]-P_g₀[5])
@constraint(model, Pˢᴳ³⁰_1[1]-P_g₀[6]<=Rₘₐₓ[6])
@constraint(model, -Rₘₐₓ[6]<=Pˢᴳ³⁰_1[1]-P_g₀[6])
@constraint(model, Pˢᴳ³⁰_2[1]-P_g₀[6]<=Rₘₐₓ[6])
@constraint(model, -Rₘₐₓ[6]<=Pˢᴳ³⁰_2[1]-P_g₀[6])
for t in 2:T                                                   
    @constraint(model, Pˢᴳ²_1[t]-Pˢᴳ²_1[t-1]<=Rₘₐₓ[1])        
    @constraint(model, -Rₘₐₓ[1]<=Pˢᴳ²_1[t]-Pˢᴳ²_1[t-1])  
    @constraint(model, Pˢᴳ²_2[t]-Pˢᴳ²_2[t-1]<=Rₘₐₓ[1])        
    @constraint(model, -Rₘₐₓ[1]<=Pˢᴳ²_2[t]-Pˢᴳ²_2[t-1]) 

    @constraint(model, Pˢᴳ³_1[t]-Pˢᴳ³_1[t-1]<=Rₘₐₓ[2])
    @constraint(model, -Rₘₐₓ[2]<=Pˢᴳ³_1[t]-Pˢᴳ³_1[t-1])
    @constraint(model, Pˢᴳ³_2[t]-Pˢᴳ³_2[t-1]<=Rₘₐₓ[2])
    @constraint(model, -Rₘₐₓ[2]<=Pˢᴳ³_2[t]-Pˢᴳ³_2[t-1])  
    
    @constraint(model, Pˢᴳ⁴_1[t]-Pˢᴳ⁴_1[t-1]<=Rₘₐₓ[3])
    @constraint(model, -Rₘₐₓ[3]<=Pˢᴳ⁴_1[t]-Pˢᴳ⁴_1[t-1])
    @constraint(model, Pˢᴳ⁴_2[t]-Pˢᴳ⁴_2[t-1]<=Rₘₐₓ[3])
    @constraint(model, -Rₘₐₓ[3]<=Pˢᴳ⁴_2[t]-Pˢᴳ⁴_2[t-1])

    @constraint(model, Pˢᴳ⁵_1[t]-Pˢᴳ⁵_1[t-1]<=Rₘₐₓ[4])
    @constraint(model, -Rₘₐₓ[4]<=Pˢᴳ⁵_1[t]-Pˢᴳ⁵_1[t-1])
    @constraint(model, Pˢᴳ⁵_2[t]-Pˢᴳ⁵_2[t-1]<=Rₘₐₓ[4])
    @constraint(model, -Rₘₐₓ[4]<=Pˢᴳ⁵_2[t]-Pˢᴳ⁵_2[t-1])

    @constraint(model, Pˢᴳ²⁷_1[t]-Pˢᴳ²⁷_1[t-1]<=Rₘₐₓ[5])
    @constraint(model, -Rₘₐₓ[5]<=Pˢᴳ²⁷_1[t]-Pˢᴳ²⁷_1[t-1])
    @constraint(model, Pˢᴳ²⁷_2[t]-Pˢᴳ²⁷_2[t-1]<=Rₘₐₓ[5])
    @constraint(model, -Rₘₐₓ[5]<=Pˢᴳ²⁷_2[t]-Pˢᴳ²⁷_2[t-1])

    @constraint(model, Pˢᴳ³⁰_1[t]-Pˢᴳ³⁰_1[t-1]<=Rₘₐₓ[6])
    @constraint(model, -Rₘₐₓ[6]<=Pˢᴳ³⁰_1[t]-Pˢᴳ³⁰_1[t-1])
    @constraint(model, Pˢᴳ³⁰_2[t]-Pˢᴳ³⁰_2[t-1]<=Rₘₐₓ[6])
    @constraint(model, -Rₘₐₓ[6]<=Pˢᴳ³⁰_2[t]-Pˢᴳ³⁰_2[t-1])
end

@constraint(model, Cᵁ²_1[1]>=(yˢᴳ²_1[1]-yˢᴳ₀[1])*Kˢᵗ[1])        # startup costs and shutdown costs for SGs , dual variables: σˢᵗₜ , σˢʰₜ
@constraint(model, Cᴰ²_1[1]>=(yˢᴳ₀[1]-yˢᴳ²_1[1])*Kˢʰ[1])  
@constraint(model, Cᵁ²_2[1]>=(yˢᴳ²_2[1]-yˢᴳ₀[1])*Kˢᵗ[1])        
@constraint(model, Cᴰ²_2[1]>=(yˢᴳ₀[1]-yˢᴳ²_2[1])*Kˢʰ[1]) 
@constraint(model, Cᵁ³_1[1]>=(yˢᴳ³_1[1]-yˢᴳ₀[2])*Kˢᵗ[2])
@constraint(model, Cᴰ³_1[1]>=(yˢᴳ₀[2]-yˢᴳ³_1[1])*Kˢʰ[2])
@constraint(model, Cᵁ³_2[1]>=(yˢᴳ³_2[1]-yˢᴳ₀[2])*Kˢᵗ[2])
@constraint(model, Cᴰ³_2[1]>=(yˢᴳ₀[2]-yˢᴳ³_2[1])*Kˢʰ[2])
@constraint(model, Cᵁ⁴_1[1]>=(yˢᴳ⁴_1[1]-yˢᴳ₀[3])*Kˢᵗ[3])
@constraint(model, Cᴰ⁴_1[1]>=(yˢᴳ₀[3]-yˢᴳ⁴_1[1])*Kˢʰ[3])
@constraint(model, Cᵁ⁴_2[1]>=(yˢᴳ⁴_2[1]-yˢᴳ₀[3])*Kˢᵗ[3])
@constraint(model, Cᴰ⁴_2[1]>=(yˢᴳ₀[3]-yˢᴳ⁴_2[1])*Kˢʰ[3])
@constraint(model, Cᵁ⁵_1[1]>=(yˢᴳ⁵_1[1]-yˢᴳ₀[4])*Kˢᵗ[4])
@constraint(model, Cᴰ⁵_1[1]>=(yˢᴳ₀[4]-yˢᴳ⁵_1[1])*Kˢʰ[4])
@constraint(model, Cᵁ⁵_2[1]>=(yˢᴳ⁵_2[1]-yˢᴳ₀[4])*Kˢᵗ[4])
@constraint(model, Cᴰ⁵_2[1]>=(yˢᴳ₀[4]-yˢᴳ⁵_2[1])*Kˢʰ[4])
@constraint(model, Cᵁ²⁷_1[1]>=(yˢᴳ²⁷_1[1]-yˢᴳ₀[5])*Kˢᵗ[5])
@constraint(model, Cᴰ²⁷_1[1]>=(yˢᴳ₀[5]-yˢᴳ²⁷_1[1])*Kˢʰ[5])
@constraint(model, Cᵁ²⁷_2[1]>=(yˢᴳ²⁷_2[1]-yˢᴳ₀[5])*Kˢᵗ[5])
@constraint(model, Cᴰ²⁷_2[1]>=(yˢᴳ₀[5]-yˢᴳ²⁷_2[1])*Kˢʰ[5])
@constraint(model, Cᵁ³⁰_1[1]>=(yˢᴳ³⁰_1[1]-yˢᴳ₀[6])*Kˢᵗ[6])
@constraint(model, Cᴰ³⁰_1[1]>=(yˢᴳ₀[6]-yˢᴳ³⁰_1[1])*Kˢʰ[6])
@constraint(model, Cᵁ³⁰_2[1]>=(yˢᴳ³⁰_2[1]-yˢᴳ₀[6])*Kˢᵗ[6])
@constraint(model, Cᴰ³⁰_2[1]>=(yˢᴳ₀[6]-yˢᴳ³⁰_2[1])*Kˢʰ[6]) 
for t in 2:T
    @constraint(model, Cᵁ²_1[t]>=(yˢᴳ²_1[t]-yˢᴳ²_1[t-1])*Kˢᵗ[1])        
    @constraint(model, Cᴰ²_1[t]>=(yˢᴳ²_1[t-1]-yˢᴳ²_1[t])*Kˢʰ[1])  
    @constraint(model, Cᵁ²_2[t]>=(yˢᴳ²_2[t]-yˢᴳ²_2[t-1])*Kˢᵗ[1])        
    @constraint(model, Cᴰ²_2[t]>=(yˢᴳ²_2[t-1]-yˢᴳ²_2[t])*Kˢʰ[1]) 

    @constraint(model, Cᵁ³_1[t]>=(yˢᴳ³_1[t]-yˢᴳ³_1[t-1])*Kˢᵗ[2]) 
    @constraint(model, Cᴰ³_1[t]>=(yˢᴳ³_1[t-1]-yˢᴳ³_1[t])*Kˢʰ[2])
    @constraint(model, Cᵁ³_2[t]>=(yˢᴳ³_2[t]-yˢᴳ³_2[t-1])*Kˢᵗ[2])
    @constraint(model, Cᴰ³_2[t]>=(yˢᴳ³_2[t-1]-yˢᴳ³_2[t])*Kˢʰ[2])

    @constraint(model, Cᵁ⁴_1[t]>=(yˢᴳ⁴_1[t]-yˢᴳ⁴_1[t-1])*Kˢᵗ[3])
    @constraint(model, Cᴰ⁴_1[t]>=(yˢᴳ⁴_1[t-1]-yˢᴳ⁴_1[t])*Kˢʰ[3])
    @constraint(model, Cᵁ⁴_2[t]>=(yˢᴳ⁴_2[t]-yˢᴳ⁴_2[t-1])*Kˢᵗ[3])
    @constraint(model, Cᴰ⁴_2[t]>=(yˢᴳ⁴_2[t-1]-yˢᴳ⁴_2[t])*Kˢʰ[3])

    @constraint(model, Cᵁ⁵_1[t]>=(yˢᴳ⁵_1[t]-yˢᴳ⁵_1[t-1])*Kˢᵗ[4])
    @constraint(model, Cᴰ⁵_1[t]>=(yˢᴳ⁵_1[t-1]-yˢᴳ⁵_1[t])*Kˢʰ[4])
    @constraint(model, Cᵁ⁵_2[t]>=(yˢᴳ⁵_2[t]-yˢᴳ⁵_2[t-1])*Kˢᵗ[4])
    @constraint(model, Cᴰ⁵_2[t]>=(yˢᴳ⁵_2[t-1]-yˢᴳ⁵_2[t])*Kˢʰ[4])

    @constraint(model, Cᵁ²⁷_1[t]>=(yˢᴳ²⁷_1[t]-yˢᴳ²⁷_1[t-1])*Kˢᵗ[5])
    @constraint(model, Cᴰ²⁷_1[t]>=(yˢᴳ²⁷_1[t-1]-yˢᴳ²⁷_1[t])*Kˢʰ[5])
    @constraint(model, Cᵁ²⁷_2[t]>=(yˢᴳ²⁷_2[t]-yˢᴳ²⁷_2[t-1])*Kˢᵗ[5])
    @constraint(model, Cᴰ²⁷_2[t]>=(yˢᴳ²⁷_2[t-1]-yˢᴳ²⁷_2[t])*Kˢʰ[5])

    @constraint(model, Cᵁ³⁰_1[t]>=(yˢᴳ³⁰_1[t]-yˢᴳ³⁰_1[t-1])*Kˢᵗ[6])
    @constraint(model, Cᴰ³⁰_1[t]>=(yˢᴳ³⁰_1[t-1]-yˢᴳ³⁰_1[t])*Kˢʰ[6])
    @constraint(model, Cᵁ³⁰_2[t]>=(yˢᴳ³⁰_2[t]-yˢᴳ³⁰_2[t-1])*Kˢᵗ[6])
    @constraint(model, Cᴰ³⁰_2[t]>=(yˢᴳ³⁰_2[t-1]-yˢᴳ³⁰_2[t])*Kˢʰ[6])
end
    
for t in 1:T
    @constraint(model, Pᴵᴮᴳ¹[t] <= IBG₁[t]*α₁[t])        # wind power limit  , dual variable: ζᵐᵃˣₜ
    @constraint(model, Pᴵᴮᴳ²³[t]<= IBG₂₃[t]*α₂₃[t])       
    @constraint(model, Pᴵᴮᴳ²⁶[t]<= IBG₂₆[t]*α₂₆[t])       

    @constraint(model, α₁[t]<=1)                         # IBR online capacity limit  , dual variable:      φᵐᵃˣₜ
    @constraint(model, α₂₃[t]<=1)                   
    @constraint(model, α₂₆[t]<=1)                   
end

k=26    # for bus 26   
for t in 1:T     # bounds for the SCL of buses  I_₂₆   I_₂₉  I_₃₀
        @constraint(model, I_₂₆[t]==                     # SCL on bus F  , dual variable:   
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]*α₁[t]+ K_c[2,k]*α₂₃[t]+ K_c[3,k]*α₂₆[t])

        @constraint(model, I_₂₆[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F  , dual variable: λ_F  
end

k=29    
for t in 1:T     # bounds for the SCL of buses  I_₂₆   I_₂₉  I_₃₀
        @constraint(model, I_₂₉[t]==                     # SCL on bus F  , dual variable:   
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]*α₁[t]+ K_c[2,k]*α₂₃[t]+ K_c[3,k]*α₂₆[t])

        @constraint(model, I_₂₉[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F  , dual variable: λ_F  
end

k=30    
for t in 1:T     # bounds for the SCL of buses  I_₂₆   I_₂₉  I_₃₀
        @constraint(model, I_₃₀[t]==                     # SCL on bus F  , dual variable:   
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]*α₁[t]+ K_c[2,k]*α₂₃[t]+ K_c[3,k]*α₂₆[t])

        @constraint(model, I_₃₀[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F  , dual variable: λ_F  
end



n=1    # for bus 26
k=26
#-------Define Dual Constraints   
@constraint(model, Oⁿˡ[1] -K_g[1,k]*λ_F[n,T] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_1[T] -Kˢʰ[1]*σˢʰˢᴳ²_1[T]+ ψᵐᵃˣˢᴳ²_1[T] >=0)             # dual constraints for UC, when t==T
@constraint(model, Oⁿˡ[1] -K_g[2,k]*λ_F[n,T] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_2[T] -Kˢʰ[1]*σˢʰˢᴳ²_2[T]+ ψᵐᵃˣˢᴳ²_2[T] >=0)
@constraint(model, Oⁿˡ[2] -K_g[3,k]*λ_F[n,T] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_1[T] -Kˢʰ[2]*σˢʰˢᴳ³_1[T]+ ψᵐᵃˣˢᴳ³_1[T] >=0)
@constraint(model, Oⁿˡ[2] -K_g[4,k]*λ_F[n,T] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_2[T] -Kˢʰ[2]*σˢʰˢᴳ³_2[T]+ ψᵐᵃˣˢᴳ³_2[T] >=0)
@constraint(model, Oⁿˡ[3] -K_g[5,k]*λ_F[n,T] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_1[T] -Kˢʰ[3]*σˢʰˢᴳ⁴_1[T]+ ψᵐᵃˣˢᴳ⁴_1[T] >=0)
@constraint(model, Oⁿˡ[3] -K_g[6,k]*λ_F[n,T] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_2[T] -Kˢʰ[3]*σˢʰˢᴳ⁴_2[T]+ ψᵐᵃˣˢᴳ⁴_2[T] >=0)
@constraint(model, Oⁿˡ[4] -K_g[7,k]*λ_F[n,T] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_1[T] -Kˢʰ[4]*σˢʰˢᴳ⁵_1[T]+ ψᵐᵃˣˢᴳ⁵_1[T] >=0)
@constraint(model, Oⁿˡ[4] -K_g[8,k]*λ_F[n,T] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_2[T] -Kˢʰ[4]*σˢʰˢᴳ⁵_2[T]+ ψᵐᵃˣˢᴳ⁵_2[T] >=0)
@constraint(model, Oⁿˡ[5] -K_g[9,k]*λ_F[n,T] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1[T] -Kˢʰ[5]*σˢʰˢᴳ²⁷_1[T]+ ψᵐᵃˣˢᴳ²⁷_1[T] >=0)
@constraint(model, Oⁿˡ[5] -K_g[10,k]*λ_F[n,T] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2[T] -Kˢʰ[5]*σˢʰˢᴳ²⁷_2[T]+ ψᵐᵃˣˢᴳ²⁷_2[T] >=0)
@constraint(model, Oⁿˡ[6] -K_g[11,k]*λ_F[n,T] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1[T] -Kˢʰ[6]*σˢʰˢᴳ³⁰_1[T]+ ψᵐᵃˣˢᴳ³⁰_1[T] >=0)
@constraint(model, Oⁿˡ[6] -K_g[12,k]*λ_F[n,T] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2[T] -Kˢʰ[6]*σˢʰˢᴳ³⁰_2[T]+ ψᵐᵃˣˢᴳ³⁰_2[T] >=0)
for t in 1:T-1                                                                                                                                                       # dual constraints for UC, when t<=T-1
    @constraint(model, Oⁿˡ[1] -K_g[1,k]*λ_F[n,t] - Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[t]+ Kˢᵗ[1]*(σˢᵗˢᴳ²_1[t]-σˢᵗˢᴳ²_1[t+1])+ Kˢʰ[1]*(σˢᵗˢᴳ²_1[t+1]-σˢᵗˢᴳ²_1[t])+ ψᵐᵃˣˢᴳ²_1[t] >=0)                
    @constraint(model, Oⁿˡ[1] -K_g[2,k]*λ_F[n,t] - Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[t]+ Kˢᵗ[1]*(σˢᵗˢᴳ²_2[t]-σˢᵗˢᴳ²_2[t+1])+ Kˢʰ[1]*(σˢᵗˢᴳ²_2[t+1]-σˢᵗˢᴳ²_2[t])+ ψᵐᵃˣˢᴳ²_2[t] >=0)   
    @constraint(model, Oⁿˡ[2] -K_g[3,k]*λ_F[n,t] - Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[t]+ Kˢᵗ[2]*(σˢᵗˢᴳ³_1[t]-σˢᵗˢᴳ³_1[t+1])+ Kˢʰ[2]*(σˢᵗˢᴳ³_1[t+1]-σˢᵗˢᴳ³_1[t])+ ψᵐᵃˣˢᴳ³_1[t] >=0)
    @constraint(model, Oⁿˡ[2] -K_g[4,k]*λ_F[n,t] - Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[t]+ Kˢᵗ[2]*(σˢᵗˢᴳ³_2[t]-σˢᵗˢᴳ³_2[t+1])+ Kˢʰ[2]*(σˢᵗˢᴳ³_2[t+1]-σˢᵗˢᴳ³_2[t])+ ψᵐᵃˣˢᴳ³_2[t] >=0)
    @constraint(model, Oⁿˡ[3] -K_g[5,k]*λ_F[n,t] - Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[t]+ Kˢᵗ[3]*(σˢᵗˢᴳ⁴_1[t]-σˢᵗˢᴳ⁴_1[t+1])+ Kˢʰ[3]*(σˢᵗˢᴳ⁴_1[t+1]-σˢᵗˢᴳ⁴_1[t])+ ψᵐᵃˣˢᴳ⁴_1[t] >=0)
    @constraint(model, Oⁿˡ[3] -K_g[6,k]*λ_F[n,t] - Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[t]+ Kˢᵗ[3]*(σˢᵗˢᴳ⁴_2[t]-σˢᵗˢᴳ⁴_2[t+1])+ Kˢʰ[3]*(σˢᵗˢᴳ⁴_2[t+1]-σˢᵗˢᴳ⁴_2[t])+ ψᵐᵃˣˢᴳ⁴_2[t] >=0)
    @constraint(model, Oⁿˡ[4] -K_g[7,k]*λ_F[n,t] - Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[t]+ Kˢᵗ[4]*(σˢᵗˢᴳ⁵_1[t]-σˢᵗˢᴳ⁵_1[t+1])+ Kˢʰ[4]*(σˢᵗˢᴳ⁵_1[t+1]-σˢᵗˢᴳ⁵_1[t])+ ψᵐᵃˣˢᴳ⁵_1[t] >=0)
    @constraint(model, Oⁿˡ[4] -K_g[8,k]*λ_F[n,t] - Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[t]+ Kˢᵗ[4]*(σˢᵗˢᴳ⁵_2[t]-σˢᵗˢᴳ⁵_2[t+1])+ Kˢʰ[4]*(σˢᵗˢᴳ⁵_2[t+1]-σˢᵗˢᴳ⁵_2[t])+ ψᵐᵃˣˢᴳ⁵_2[t] >=0)
    @constraint(model, Oⁿˡ[5] -K_g[9,k]*λ_F[n,t] - Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[t]+ Kˢᵗ[5]*(σˢᵗˢᴳ²⁷_1[t]-σˢᵗˢᴳ²⁷_1[t+1])+ Kˢʰ[5]*(σˢᵗˢᴳ²⁷_1[t+1]-σˢᵗˢᴳ²⁷_1[t])+ ψᵐᵃˣˢᴳ²⁷_1[t] >=0)
    @constraint(model, Oⁿˡ[5] -K_g[10,k]*λ_F[n,t] - Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[t]+ Kˢᵗ[5]*(σˢᵗˢᴳ²⁷_2[t]-σˢᵗˢᴳ²⁷_2[t+1])+ Kˢʰ[5]*(σˢᵗˢᴳ²⁷_2[t+1]-σˢᵗˢᴳ²⁷_2[t])+ ψᵐᵃˣˢᴳ²⁷_2[t] >=0)
    @constraint(model, Oⁿˡ[6] -K_g[11,k]*λ_F[n,t] - Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[t]+ Kˢᵗ[6]*(σˢᵗˢᴳ³⁰_1[t]-σˢᵗˢᴳ³⁰_1[t+1])+ Kˢʰ[6]*(σˢᵗˢᴳ³⁰_1[t+1]-σˢᵗˢᴳ³⁰_1[t])+ ψᵐᵃˣˢᴳ³⁰_1[t] >=0)
    @constraint(model, Oⁿˡ[6] -K_g[12,k]*λ_F[n,t] - Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[t]+ Kˢᵗ[6]*(σˢᵗˢᴳ³⁰_2[t]-σˢᵗˢᴳ³⁰_2[t+1])+ Kˢʰ[6]*(σˢᵗˢᴳ³⁰_2[t+1]-σˢᵗˢᴳ³⁰_2[t])+ ψᵐᵃˣˢᴳ³⁰_2[t] >=0)          
end

n=2    # for bus 29
k=29
#-------Define Dual Constraints   
@constraint(model, Oⁿˡ[1] -K_g[1,k]*λ_F[n,T] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_1[T] -Kˢʰ[1]*σˢʰˢᴳ²_1[T]+ ψᵐᵃˣˢᴳ²_1[T] >=0)             # dual constraints for UC, when t==T
@constraint(model, Oⁿˡ[1] -K_g[2,k]*λ_F[n,T] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_2[T] -Kˢʰ[1]*σˢʰˢᴳ²_2[T]+ ψᵐᵃˣˢᴳ²_2[T] >=0)
@constraint(model, Oⁿˡ[2] -K_g[3,k]*λ_F[n,T] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_1[T] -Kˢʰ[2]*σˢʰˢᴳ³_1[T]+ ψᵐᵃˣˢᴳ³_1[T] >=0)
@constraint(model, Oⁿˡ[2] -K_g[4,k]*λ_F[n,T] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_2[T] -Kˢʰ[2]*σˢʰˢᴳ³_2[T]+ ψᵐᵃˣˢᴳ³_2[T] >=0)
@constraint(model, Oⁿˡ[3] -K_g[5,k]*λ_F[n,T] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_1[T] -Kˢʰ[3]*σˢʰˢᴳ⁴_1[T]+ ψᵐᵃˣˢᴳ⁴_1[T] >=0)
@constraint(model, Oⁿˡ[3] -K_g[6,k]*λ_F[n,T] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_2[T] -Kˢʰ[3]*σˢʰˢᴳ⁴_2[T]+ ψᵐᵃˣˢᴳ⁴_2[T] >=0)
@constraint(model, Oⁿˡ[4] -K_g[7,k]*λ_F[n,T] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_1[T] -Kˢʰ[4]*σˢʰˢᴳ⁵_1[T]+ ψᵐᵃˣˢᴳ⁵_1[T] >=0)
@constraint(model, Oⁿˡ[4] -K_g[8,k]*λ_F[n,T] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_2[T] -Kˢʰ[4]*σˢʰˢᴳ⁵_2[T]+ ψᵐᵃˣˢᴳ⁵_2[T] >=0)
@constraint(model, Oⁿˡ[5] -K_g[9,k]*λ_F[n,T] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1[T] -Kˢʰ[5]*σˢʰˢᴳ²⁷_1[T]+ ψᵐᵃˣˢᴳ²⁷_1[T] >=0)
@constraint(model, Oⁿˡ[5] -K_g[10,k]*λ_F[n,T] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2[T] -Kˢʰ[5]*σˢʰˢᴳ²⁷_2[T]+ ψᵐᵃˣˢᴳ²⁷_2[T] >=0)
@constraint(model, Oⁿˡ[6] -K_g[11,k]*λ_F[n,T] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1[T] -Kˢʰ[6]*σˢʰˢᴳ³⁰_1[T]+ ψᵐᵃˣˢᴳ³⁰_1[T] >=0)
@constraint(model, Oⁿˡ[6] -K_g[12,k]*λ_F[n,T] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2[T] -Kˢʰ[6]*σˢʰˢᴳ³⁰_2[T]+ ψᵐᵃˣˢᴳ³⁰_2[T] >=0)
for t in 1:T-1                                                                                                                                                       # dual constraints for UC, when t<=T-1
    @constraint(model, Oⁿˡ[1] -K_g[1,k]*λ_F[n,t] - Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[t]+ Kˢᵗ[1]*(σˢᵗˢᴳ²_1[t]-σˢᵗˢᴳ²_1[t+1])+ Kˢʰ[1]*(σˢᵗˢᴳ²_1[t+1]-σˢᵗˢᴳ²_1[t])+ ψᵐᵃˣˢᴳ²_1[t] >=0)                
    @constraint(model, Oⁿˡ[1] -K_g[2,k]*λ_F[n,t] - Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[t]+ Kˢᵗ[1]*(σˢᵗˢᴳ²_2[t]-σˢᵗˢᴳ²_2[t+1])+ Kˢʰ[1]*(σˢᵗˢᴳ²_2[t+1]-σˢᵗˢᴳ²_2[t])+ ψᵐᵃˣˢᴳ²_2[t] >=0)   
    @constraint(model, Oⁿˡ[2] -K_g[3,k]*λ_F[n,t] - Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[t]+ Kˢᵗ[2]*(σˢᵗˢᴳ³_1[t]-σˢᵗˢᴳ³_1[t+1])+ Kˢʰ[2]*(σˢᵗˢᴳ³_1[t+1]-σˢᵗˢᴳ³_1[t])+ ψᵐᵃˣˢᴳ³_1[t] >=0)
    @constraint(model, Oⁿˡ[2] -K_g[4,k]*λ_F[n,t] - Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[t]+ Kˢᵗ[2]*(σˢᵗˢᴳ³_2[t]-σˢᵗˢᴳ³_2[t+1])+ Kˢʰ[2]*(σˢᵗˢᴳ³_2[t+1]-σˢᵗˢᴳ³_2[t])+ ψᵐᵃˣˢᴳ³_2[t] >=0)
    @constraint(model, Oⁿˡ[3] -K_g[5,k]*λ_F[n,t] - Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[t]+ Kˢᵗ[3]*(σˢᵗˢᴳ⁴_1[t]-σˢᵗˢᴳ⁴_1[t+1])+ Kˢʰ[3]*(σˢᵗˢᴳ⁴_1[t+1]-σˢᵗˢᴳ⁴_1[t])+ ψᵐᵃˣˢᴳ⁴_1[t] >=0)
    @constraint(model, Oⁿˡ[3] -K_g[6,k]*λ_F[n,t] - Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[t]+ Kˢᵗ[3]*(σˢᵗˢᴳ⁴_2[t]-σˢᵗˢᴳ⁴_2[t+1])+ Kˢʰ[3]*(σˢᵗˢᴳ⁴_2[t+1]-σˢᵗˢᴳ⁴_2[t])+ ψᵐᵃˣˢᴳ⁴_2[t] >=0)
    @constraint(model, Oⁿˡ[4] -K_g[7,k]*λ_F[n,t] - Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[t]+ Kˢᵗ[4]*(σˢᵗˢᴳ⁵_1[t]-σˢᵗˢᴳ⁵_1[t+1])+ Kˢʰ[4]*(σˢᵗˢᴳ⁵_1[t+1]-σˢᵗˢᴳ⁵_1[t])+ ψᵐᵃˣˢᴳ⁵_1[t] >=0)
    @constraint(model, Oⁿˡ[4] -K_g[8,k]*λ_F[n,t] - Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[t]+ Kˢᵗ[4]*(σˢᵗˢᴳ⁵_2[t]-σˢᵗˢᴳ⁵_2[t+1])+ Kˢʰ[4]*(σˢᵗˢᴳ⁵_2[t+1]-σˢᵗˢᴳ⁵_2[t])+ ψᵐᵃˣˢᴳ⁵_2[t] >=0)
    @constraint(model, Oⁿˡ[5] -K_g[9,k]*λ_F[n,t] - Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[t]+ Kˢᵗ[5]*(σˢᵗˢᴳ²⁷_1[t]-σˢᵗˢᴳ²⁷_1[t+1])+ Kˢʰ[5]*(σˢᵗˢᴳ²⁷_1[t+1]-σˢᵗˢᴳ²⁷_1[t])+ ψᵐᵃˣˢᴳ²⁷_1[t] >=0)
    @constraint(model, Oⁿˡ[5] -K_g[10,k]*λ_F[n,t] - Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[t]+ Kˢᵗ[5]*(σˢᵗˢᴳ²⁷_2[t]-σˢᵗˢᴳ²⁷_2[t+1])+ Kˢʰ[5]*(σˢᵗˢᴳ²⁷_2[t+1]-σˢᵗˢᴳ²⁷_2[t])+ ψᵐᵃˣˢᴳ²⁷_2[t] >=0)
    @constraint(model, Oⁿˡ[6] -K_g[11,k]*λ_F[n,t] - Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[t]+ Kˢᵗ[6]*(σˢᵗˢᴳ³⁰_1[t]-σˢᵗˢᴳ³⁰_1[t+1])+ Kˢʰ[6]*(σˢᵗˢᴳ³⁰_1[t+1]-σˢᵗˢᴳ³⁰_1[t])+ ψᵐᵃˣˢᴳ³⁰_1[t] >=0)
    @constraint(model, Oⁿˡ[6] -K_g[12,k]*λ_F[n,t] - Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[t]+ Kˢᵗ[6]*(σˢᵗˢᴳ³⁰_2[t]-σˢᵗˢᴳ³⁰_2[t+1])+ Kˢʰ[6]*(σˢᵗˢᴳ³⁰_2[t+1]-σˢᵗˢᴳ³⁰_2[t])+ ψᵐᵃˣˢᴳ³⁰_2[t] >=0)          
end

n=3    # for bus 30
k=30
#-------Define Dual Constraints   
@constraint(model, Oⁿˡ[1] -K_g[1,k]*λ_F[n,T] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_1[T] -Kˢʰ[1]*σˢʰˢᴳ²_1[T]+ ψᵐᵃˣˢᴳ²_1[T] >=0)             # dual constraints for UC, when t==T
@constraint(model, Oⁿˡ[1] -K_g[2,k]*λ_F[n,T] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_2[T] -Kˢʰ[1]*σˢʰˢᴳ²_2[T]+ ψᵐᵃˣˢᴳ²_2[T] >=0)
@constraint(model, Oⁿˡ[2] -K_g[3,k]*λ_F[n,T] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_1[T] -Kˢʰ[2]*σˢʰˢᴳ³_1[T]+ ψᵐᵃˣˢᴳ³_1[T] >=0)
@constraint(model, Oⁿˡ[2] -K_g[4,k]*λ_F[n,T] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_2[T] -Kˢʰ[2]*σˢʰˢᴳ³_2[T]+ ψᵐᵃˣˢᴳ³_2[T] >=0)
@constraint(model, Oⁿˡ[3] -K_g[5,k]*λ_F[n,T] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_1[T] -Kˢʰ[3]*σˢʰˢᴳ⁴_1[T]+ ψᵐᵃˣˢᴳ⁴_1[T] >=0)
@constraint(model, Oⁿˡ[3] -K_g[6,k]*λ_F[n,T] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_2[T] -Kˢʰ[3]*σˢʰˢᴳ⁴_2[T]+ ψᵐᵃˣˢᴳ⁴_2[T] >=0)
@constraint(model, Oⁿˡ[4] -K_g[7,k]*λ_F[n,T] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_1[T] -Kˢʰ[4]*σˢʰˢᴳ⁵_1[T]+ ψᵐᵃˣˢᴳ⁵_1[T] >=0)
@constraint(model, Oⁿˡ[4] -K_g[8,k]*λ_F[n,T] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_2[T] -Kˢʰ[4]*σˢʰˢᴳ⁵_2[T]+ ψᵐᵃˣˢᴳ⁵_2[T] >=0)
@constraint(model, Oⁿˡ[5] -K_g[9,k]*λ_F[n,T] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1[T] -Kˢʰ[5]*σˢʰˢᴳ²⁷_1[T]+ ψᵐᵃˣˢᴳ²⁷_1[T] >=0)
@constraint(model, Oⁿˡ[5] -K_g[10,k]*λ_F[n,T] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2[T] -Kˢʰ[5]*σˢʰˢᴳ²⁷_2[T]+ ψᵐᵃˣˢᴳ²⁷_2[T] >=0)
@constraint(model, Oⁿˡ[6] -K_g[11,k]*λ_F[n,T] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1[T] -Kˢʰ[6]*σˢʰˢᴳ³⁰_1[T]+ ψᵐᵃˣˢᴳ³⁰_1[T] >=0)
@constraint(model, Oⁿˡ[6] -K_g[12,k]*λ_F[n,T] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2[T] -Kˢʰ[6]*σˢʰˢᴳ³⁰_2[T]+ ψᵐᵃˣˢᴳ³⁰_2[T] >=0)
for t in 1:T-1                                                                                                                                                       # dual constraints for UC, when t<=T-1
    @constraint(model, Oⁿˡ[1] -K_g[1,k]*λ_F[n,t] - Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[t]+ Kˢᵗ[1]*(σˢᵗˢᴳ²_1[t]-σˢᵗˢᴳ²_1[t+1])+ Kˢʰ[1]*(σˢᵗˢᴳ²_1[t+1]-σˢᵗˢᴳ²_1[t])+ ψᵐᵃˣˢᴳ²_1[t] >=0)                
    @constraint(model, Oⁿˡ[1] -K_g[2,k]*λ_F[n,t] - Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[t]+ Kˢᵗ[1]*(σˢᵗˢᴳ²_2[t]-σˢᵗˢᴳ²_2[t+1])+ Kˢʰ[1]*(σˢᵗˢᴳ²_2[t+1]-σˢᵗˢᴳ²_2[t])+ ψᵐᵃˣˢᴳ²_2[t] >=0)   
    @constraint(model, Oⁿˡ[2] -K_g[3,k]*λ_F[n,t] - Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[t]+ Kˢᵗ[2]*(σˢᵗˢᴳ³_1[t]-σˢᵗˢᴳ³_1[t+1])+ Kˢʰ[2]*(σˢᵗˢᴳ³_1[t+1]-σˢᵗˢᴳ³_1[t])+ ψᵐᵃˣˢᴳ³_1[t] >=0)
    @constraint(model, Oⁿˡ[2] -K_g[4,k]*λ_F[n,t] - Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[t]+ Kˢᵗ[2]*(σˢᵗˢᴳ³_2[t]-σˢᵗˢᴳ³_2[t+1])+ Kˢʰ[2]*(σˢᵗˢᴳ³_2[t+1]-σˢᵗˢᴳ³_2[t])+ ψᵐᵃˣˢᴳ³_2[t] >=0)
    @constraint(model, Oⁿˡ[3] -K_g[5,k]*λ_F[n,t] - Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[t]+ Kˢᵗ[3]*(σˢᵗˢᴳ⁴_1[t]-σˢᵗˢᴳ⁴_1[t+1])+ Kˢʰ[3]*(σˢᵗˢᴳ⁴_1[t+1]-σˢᵗˢᴳ⁴_1[t])+ ψᵐᵃˣˢᴳ⁴_1[t] >=0)
    @constraint(model, Oⁿˡ[3] -K_g[6,k]*λ_F[n,t] - Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[t]+ Kˢᵗ[3]*(σˢᵗˢᴳ⁴_2[t]-σˢᵗˢᴳ⁴_2[t+1])+ Kˢʰ[3]*(σˢᵗˢᴳ⁴_2[t+1]-σˢᵗˢᴳ⁴_2[t])+ ψᵐᵃˣˢᴳ⁴_2[t] >=0)
    @constraint(model, Oⁿˡ[4] -K_g[7,k]*λ_F[n,t] - Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[t]+ Kˢᵗ[4]*(σˢᵗˢᴳ⁵_1[t]-σˢᵗˢᴳ⁵_1[t+1])+ Kˢʰ[4]*(σˢᵗˢᴳ⁵_1[t+1]-σˢᵗˢᴳ⁵_1[t])+ ψᵐᵃˣˢᴳ⁵_1[t] >=0)
    @constraint(model, Oⁿˡ[4] -K_g[8,k]*λ_F[n,t] - Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[t]+ Kˢᵗ[4]*(σˢᵗˢᴳ⁵_2[t]-σˢᵗˢᴳ⁵_2[t+1])+ Kˢʰ[4]*(σˢᵗˢᴳ⁵_2[t+1]-σˢᵗˢᴳ⁵_2[t])+ ψᵐᵃˣˢᴳ⁵_2[t] >=0)
    @constraint(model, Oⁿˡ[5] -K_g[9,k]*λ_F[n,t] - Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[t]+ Kˢᵗ[5]*(σˢᵗˢᴳ²⁷_1[t]-σˢᵗˢᴳ²⁷_1[t+1])+ Kˢʰ[5]*(σˢᵗˢᴳ²⁷_1[t+1]-σˢᵗˢᴳ²⁷_1[t])+ ψᵐᵃˣˢᴳ²⁷_1[t] >=0)
    @constraint(model, Oⁿˡ[5] -K_g[10,k]*λ_F[n,t] - Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[t]+ Kˢᵗ[5]*(σˢᵗˢᴳ²⁷_2[t]-σˢᵗˢᴳ²⁷_2[t+1])+ Kˢʰ[5]*(σˢᵗˢᴳ²⁷_2[t+1]-σˢᵗˢᴳ²⁷_2[t])+ ψᵐᵃˣˢᴳ²⁷_2[t] >=0)
    @constraint(model, Oⁿˡ[6] -K_g[11,k]*λ_F[n,t] - Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[t]+ Kˢᵗ[6]*(σˢᵗˢᴳ³⁰_1[t]-σˢᵗˢᴳ³⁰_1[t+1])+ Kˢʰ[6]*(σˢᵗˢᴳ³⁰_1[t+1]-σˢᵗˢᴳ³⁰_1[t])+ ψᵐᵃˣˢᴳ³⁰_1[t] >=0)
    @constraint(model, Oⁿˡ[6] -K_g[12,k]*λ_F[n,t] - Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[t]+ Kˢᵗ[6]*(σˢᵗˢᴳ³⁰_2[t]-σˢᵗˢᴳ³⁰_2[t+1])+ Kˢʰ[6]*(σˢᵗˢᴳ³⁰_2[t+1]-σˢᵗˢᴳ³⁰_2[t])+ ψᵐᵃˣˢᴳ³⁰_2[t] >=0)          
end


@constraint(model, Oᵐ₁[1] -λᴱ[T] +μᵐᵃˣˢᴳ²_1[T] -μᵐⁱⁿˢᴳ²_1[T] +πʳᵘˢᴳ²_1[T] -πʳᵈˢᴳ²_1[T]>=0)          # dual constraints for generation, when t==T, assume SGs in bus 2 are strategic
@constraint(model, Oᵐ₂[1] -λᴱ[T] +μᵐᵃˣˢᴳ²_2[T] -μᵐⁱⁿˢᴳ²_2[T] +πʳᵘˢᴳ²_2[T] -πʳᵈˢᴳ²_2[T]>=0)
@constraint(model, Oᵐ₁[2] -λᴱ[T] +μᵐᵃˣˢᴳ³_1[T] -μᵐⁱⁿˢᴳ³_1[T] +πʳᵘˢᴳ³_1[T] -πʳᵈˢᴳ³_1[T]>=0)
@constraint(model, Oᵐ₂[2] -λᴱ[T] +μᵐᵃˣˢᴳ³_2[T] -μᵐⁱⁿˢᴳ³_2[T] +πʳᵘˢᴳ³_2[T] -πʳᵈˢᴳ³_2[T]>=0)
@constraint(model, Oᵐ₁[3] -λᴱ[T] +μᵐᵃˣˢᴳ⁴_1[T] -μᵐⁱⁿˢᴳ⁴_1[T] +πʳᵘˢᴳ⁴_1[T] -πʳᵈˢᴳ⁴_1[T]>=0)
@constraint(model, Oᵐ₂[3] -λᴱ[T] +μᵐᵃˣˢᴳ⁴_2[T] -μᵐⁱⁿˢᴳ⁴_2[T] +πʳᵘˢᴳ⁴_2[T] -πʳᵈˢᴳ⁴_2[T]>=0)
@constraint(model, Oᵐ₁[4] -λᴱ[T] +μᵐᵃˣˢᴳ⁵_1[T] -μᵐⁱⁿˢᴳ⁵_1[T] +πʳᵘˢᴳ⁵_1[T] -πʳᵈˢᴳ⁵_1[T]>=0)
@constraint(model, Oᵐ₂[4] -λᴱ[T] +μᵐᵃˣˢᴳ⁵_2[T] -μᵐⁱⁿˢᴳ⁵_2[T] +πʳᵘˢᴳ⁵_2[T] -πʳᵈˢᴳ⁵_2[T]>=0)
@constraint(model, Oᵐ₁[5] -λᴱ[T] +μᵐᵃˣˢᴳ²⁷_1[T] -μᵐⁱⁿˢᴳ²⁷_1[T] +πʳᵘˢᴳ²⁷_1[T] -πʳᵈˢᴳ²⁷_1[T]>=0)
@constraint(model, Oᵐ₂[5] -λᴱ[T] +μᵐᵃˣˢᴳ²⁷_2[T] -μᵐⁱⁿˢᴳ²⁷_2[T] +πʳᵘˢᴳ²⁷_2[T] -πʳᵈˢᴳ²⁷_2[T]>=0)
@constraint(model, Oᵐ₁[6] -λᴱ[T] +μᵐᵃˣˢᴳ³⁰_1[T] -μᵐⁱⁿˢᴳ³⁰_1[T] +πʳᵘˢᴳ³⁰_1[T] -πʳᵈˢᴳ³⁰_1[T]>=0)
@constraint(model, Oᵐ₂[6] -λᴱ[T] +μᵐᵃˣˢᴳ³⁰_2[T] -μᵐⁱⁿˢᴳ³⁰_2[T] +πʳᵘˢᴳ³⁰_2[T] -πʳᵈˢᴳ³⁰_2[T]>=0)
for t in 1:T-1                                                                                                  # dual constraints for generation, when t<=T-1, assume SGs in bus 2 are strategic
    @constraint(model, Oᵐ₁[1] -λᴱ[t] +μᵐᵃˣˢᴳ²_1[t] -μᵐⁱⁿˢᴳ²_1[t] +πʳᵘˢᴳ²_1[t] -πʳᵘˢᴳ²_1[t+1] -πʳᵈˢᴳ²_1[t] +πʳᵈˢᴳ²_1[t+1]>=0)                
    @constraint(model, Oᵐ₂[1] -λᴱ[t] +μᵐᵃˣˢᴳ²_2[t] -μᵐⁱⁿˢᴳ²_2[t] +πʳᵘˢᴳ²_2[t] -πʳᵘˢᴳ²_2[t+1] -πʳᵈˢᴳ²_2[t] +πʳᵈˢᴳ²_2[t+1]>=0)
    @constraint(model, Oᵐ₁[2] -λᴱ[t] +μᵐᵃˣˢᴳ³_1[t] -μᵐⁱⁿˢᴳ³_1[t] +πʳᵘˢᴳ³_1[t] -πʳᵘˢᴳ³_1[t+1] -πʳᵈˢᴳ³_1[t] +πʳᵈˢᴳ³_1[t+1]>=0)                
    @constraint(model, Oᵐ₂[2] -λᴱ[t] +μᵐᵃˣˢᴳ³_2[t] -μᵐⁱⁿˢᴳ³_2[t] +πʳᵘˢᴳ³_2[t] -πʳᵘˢᴳ³_2[t+1] -πʳᵈˢᴳ³_2[t] +πʳᵈˢᴳ³_2[t+1]>=0)
    @constraint(model, Oᵐ₁[3] -λᴱ[t] +μᵐᵃˣˢᴳ⁴_1[t] -μᵐⁱⁿˢᴳ⁴_1[t] +πʳᵘˢᴳ⁴_1[t] -πʳᵘˢᴳ⁴_1[t+1] -πʳᵈˢᴳ⁴_1[t] +πʳᵈˢᴳ⁴_1[t+1]>=0)
    @constraint(model, Oᵐ₂[3] -λᴱ[t] +μᵐᵃˣˢᴳ⁴_2[t] -μᵐⁱⁿˢᴳ⁴_2[t] +πʳᵘˢᴳ⁴_2[t] -πʳᵘˢᴳ⁴_2[t+1] -πʳᵈˢᴳ⁴_2[t] +πʳᵈˢᴳ⁴_2[t+1]>=0)
    @constraint(model, Oᵐ₁[4] -λᴱ[t] +μᵐᵃˣˢᴳ⁵_1[t] -μᵐⁱⁿˢᴳ⁵_1[t] +πʳᵘˢᴳ⁵_1[t] -πʳᵘˢᴳ⁵_1[t+1] -πʳᵈˢᴳ⁵_1[t] +πʳᵈˢᴳ⁵_1[t+1]>=0)
    @constraint(model, Oᵐ₂[4] -λᴱ[t] +μᵐᵃˣˢᴳ⁵_2[t] -μᵐⁱⁿˢᴳ⁵_2[t] +πʳᵘˢᴳ⁵_2[t] -πʳᵘˢᴳ⁵_2[t+1] -πʳᵈˢᴳ⁵_2[t] +πʳᵈˢᴳ⁵_2[t+1]>=0)
    @constraint(model, Oᵐ₁[5] -λᴱ[t] +μᵐᵃˣˢᴳ²⁷_1[t] -μᵐⁱⁿˢᴳ²⁷_1[t] +πʳᵘˢᴳ²⁷_1[t] -πʳᵘˢᴳ²⁷_1[t+1] -πʳᵈˢᴳ²⁷_1[t] +πʳᵈˢᴳ²⁷_1[t+1]>=0)
    @constraint(model, Oᵐ₂[5] -λᴱ[t] +μᵐᵃˣˢᴳ²⁷_2[t] -μᵐⁱⁿˢᴳ²⁷_2[t] +πʳᵘˢᴳ²⁷_2[t] -πʳᵘˢᴳ²⁷_2[t+1] -πʳᵈˢᴳ²⁷_2[t] +πʳᵈˢᴳ²⁷_2[t+1]>=0)
    @constraint(model, Oᵐ₁[6] -λᴱ[t] +μᵐᵃˣˢᴳ³⁰_1[t] -μᵐⁱⁿˢᴳ³⁰_1[t] +πʳᵘˢᴳ³⁰_1[t] -πʳᵘˢᴳ³⁰_1[t+1] -πʳᵈˢᴳ³⁰_1[t] +πʳᵈˢᴳ³⁰_1[t+1]>=0)
    @constraint(model, Oᵐ₂[6] -λᴱ[t] +μᵐᵃˣˢᴳ³⁰_2[t] -μᵐⁱⁿˢᴳ³⁰_2[t] +πʳᵘˢᴳ³⁰_2[t] -πʳᵘˢᴳ³⁰_2[t+1] -πʳᵈˢᴳ³⁰_2[t] +πʳᵈˢᴳ³⁰_2[t+1]>=0)                
end
                                                                                          
@constraint(model,σˢᵗˢᴳ²_1.<=1)                                 # dual constraints for on/off costs
@constraint(model,σˢᵗˢᴳ²_2.<=1)
@constraint(model,σˢᵗˢᴳ³_1.<=1)
@constraint(model,σˢᵗˢᴳ³_2.<=1)
@constraint(model,σˢᵗˢᴳ⁴_1.<=1)
@constraint(model,σˢᵗˢᴳ⁴_2.<=1)
@constraint(model,σˢᵗˢᴳ⁵_1.<=1)
@constraint(model,σˢᵗˢᴳ⁵_2.<=1)
@constraint(model,σˢᵗˢᴳ²⁷_1.<=1)
@constraint(model,σˢᵗˢᴳ²⁷_2.<=1)
@constraint(model,σˢᵗˢᴳ³⁰_1.<=1)
@constraint(model,σˢᵗˢᴳ³⁰_2.<=1)
@constraint(model,σˢʰˢᴳ²_1.<=1)
@constraint(model,σˢʰˢᴳ²_2.<=1)
@constraint(model,σˢʰˢᴳ³_1.<=1)
@constraint(model,σˢʰˢᴳ³_2.<=1)
@constraint(model,σˢʰˢᴳ⁴_1.<=1)
@constraint(model,σˢʰˢᴳ⁴_2.<=1)
@constraint(model,σˢʰˢᴳ⁵_1.<=1)
@constraint(model,σˢʰˢᴳ⁵_2.<=1)
@constraint(model,σˢʰˢᴳ²⁷_1.<=1)
@constraint(model,σˢʰˢᴳ²⁷_2.<=1)
@constraint(model,σˢʰˢᴳ³⁰_1.<=1)
@constraint(model,σˢʰˢᴳ³⁰_2.<=1)

for t in 1:T                                            # dual constraints for generation of IBRs
    @constraint(model, Oᴱ_c[1] -λᴱ[t] +ζᵐᵃˣ¹[t]>=0)
    @constraint(model, Oᴱ_c[2] -λᴱ[t] +ζᵐᵃˣ²³[t]>=0)
    @constraint(model, Oᴱ_c[3] -λᴱ[t] +ζᵐᵃˣ²⁶[t]>=0)
end

k=26
for t in 1:T                                            # dual constraints for online capacity factor of IBRs
    @constraint(model, φᵐᵃˣ¹[t] -K_c[1,k]*λ_F[1,t] -IBG₁[t]*ζᵐᵃˣ¹[t]>=0)
    @constraint(model, φᵐᵃˣ²³[t]  -K_c[2,k]*λ_F[1,t] -IBG₂₃[t]*ζᵐᵃˣ²³[t]>=0)
    @constraint(model, φᵐᵃˣ²⁶[t]  -K_c[3,k]*λ_F[1,t] -IBG₂₆[t]*ζᵐᵃˣ²⁶[t]>=0)
end

k=29
for t in 1:T                                            # dual constraints for online capacity factor of IBRs
    @constraint(model, φᵐᵃˣ¹[t] -K_c[1,k]*λ_F[2,t] -IBG₁[t]*ζᵐᵃˣ¹[t]>=0)
    @constraint(model, φᵐᵃˣ²³[t]  -K_c[2,k]*λ_F[2,t] -IBG₂₃[t]*ζᵐᵃˣ²³[t]>=0)
    @constraint(model, φᵐᵃˣ²⁶[t]  -K_c[3,k]*λ_F[2,t] -IBG₂₆[t]*ζᵐᵃˣ²⁶[t]>=0)
end

k=30
for t in 1:T                                            # dual constraints for online capacity factor of IBRs
    @constraint(model, φᵐᵃˣ¹[t] -K_c[1,k]*λ_F[3,t] -IBG₁[t]*ζᵐᵃˣ¹[t]>=0)
    @constraint(model, φᵐᵃˣ²³[t]  -K_c[2,k]*λ_F[3,t] -IBG₂₃[t]*ζᵐᵃˣ²³[t]>=0)
    @constraint(model, φᵐᵃˣ²⁶[t]  -K_c[3,k]*λ_F[3,t] -IBG₂₆[t]*ζᵐᵃˣ²⁶[t]>=0)
end



#-------Define Objective Functions with binary expansion
cost_onoff_LL=sum(Cᵁ²_1)+sum(Cᴰ²_1)+sum(Cᵁ³_1)+sum(Cᴰ³_1)+sum(Cᵁ⁴_1)+sum(Cᴰ⁴_1)+sum(Cᵁ⁵_1)+sum(Cᴰ⁵_1)+sum(Cᵁ²⁷_1)+sum(Cᴰ²⁷_1)+sum(Cᵁ³⁰_1)+sum(Cᴰ³⁰_1)  +sum(Cᵁ²_2)+sum(Cᴰ²_2)+sum(Cᵁ³_2)+sum(Cᴰ³_2)+sum(Cᵁ⁴_2)+sum(Cᴰ⁴_2)+sum(Cᵁ⁵_2)+sum(Cᴰ⁵_2)+sum(Cᵁ²⁷_2)+sum(Cᴰ²⁷_2)+sum(Cᵁ³⁰_2)+sum(Cᴰ³⁰_2)       
cost_nl_LL=sum(Oⁿˡ[1].*(yˢᴳ²_1+yˢᴳ²_2))+sum(Oⁿˡ[2].*(yˢᴳ³_1+yˢᴳ³_2))+sum(Oⁿˡ[3].*(yˢᴳ⁴_1+yˢᴳ⁴_2))+sum(Oⁿˡ[4].*(yˢᴳ⁵_1+yˢᴳ⁵_2))+sum(Oⁿˡ[5].*(yˢᴳ²⁷_1+yˢᴳ²⁷_2))+sum(Oⁿˡ[6].*(yˢᴳ³⁰_1+yˢᴳ³⁰_2))    
cost_gene_LL=Oᵐ₁[1]*sum( Pˢᴳ²_1 )+Oᵐ₂[1]*sum( Pˢᴳ²_2 )+sum(Oᵐ₁[2].*Pˢᴳ³_1+Oᵐ₂[2].*Pˢᴳ³_2)+sum(Oᵐ₁[3].*Pˢᴳ⁴_1+Oᵐ₂[3].*Pˢᴳ⁴_2)+sum(Oᵐ₁[4].*Pˢᴳ⁵_1+Oᵐ₂[4].*Pˢᴳ⁵_2)+sum(Oᵐ₁[5].*Pˢᴳ²⁷_1+Oᵐ₂[5].*Pˢᴳ²⁷_2)+sum(Oᵐ₁[6].*Pˢᴳ³⁰_1+Oᵐ₂[6].*Pˢᴳ³⁰_2)   
cost_IBR_LL=sum(Oᴱ_c[1].*Pᴵᴮᴳ¹) +sum(Oᴱ_c[2].*Pᴵᴮᴳ²³) +sum(Oᴱ_c[3].*Pᴵᴮᴳ²⁶)
obj_LL=cost_onoff_LL+cost_nl_LL+cost_gene_LL+cost_IBR_LL

@variable(model, obj_DLL_1[1:T])
for t in 1:T
    @constraint(model, obj_DLL_1[t]== Load_total[t]*λᴱ[t] -(φᵐᵃˣ¹[t] +φᵐᵃˣ²³[t] +φᵐᵃˣ²⁶[t]) -ψᵐᵃˣˢᴳ²_1[t] -ψᵐᵃˣˢᴳ²_2[t] -ψᵐᵃˣˢᴳ³_1[t] -ψᵐᵃˣˢᴳ³_2[t] -ψᵐᵃˣˢᴳ⁴_1[t] -ψᵐᵃˣˢᴳ⁴_2[t] -ψᵐᵃˣˢᴳ⁵_1[t] -ψᵐᵃˣˢᴳ⁵_2[t] -ψᵐᵃˣˢᴳ²⁷_1[t] -ψᵐᵃˣˢᴳ²⁷_2[t] -ψᵐᵃˣˢᴳ³⁰_1[t] -ψᵐᵃˣˢᴳ³⁰_2[t])
end

@variable(model, obj_DLL_stra_1[1:T])
for t in 1:T
    @constraint(model, obj_DLL_stra_1[t]== -Rₘₐₓ[1]*πʳᵈˢᴳ²_1[t] -Rₘₐₓ[1]*πʳᵈˢᴳ²_2[t] -Rₘₐₓ[1]*πʳᵘˢᴳ²_1[t] -Rₘₐₓ[1]*πʳᵘˢᴳ²_2[t])
end

obj_DLL_stra= sum(obj_DLL_stra_1) +(P_g₀[1]*πʳᵈˢᴳ²_1[1] +P_g₀[1]*πʳᵈˢᴳ²_2[1]) -(P_g₀[1]*πʳᵘˢᴳ²_1[1] +P_g₀[1]*πʳᵘˢᴳ²_2[1]) -(yˢᴳ₀[1]*Kˢᵗ[1]*σˢᵗˢᴳ²_1[1] +yˢᴳ₀[1]*Kˢᵗ[1]*σˢᵗˢᴳ²_2[1]) +(yˢᴳ₀[1]*Kˢʰ[1]*σˢʰˢᴳ²_1[1] +yˢᴳ₀[1]*Kˢʰ[1]*σˢʰˢᴳ²_2[1])

@variable(model, obj_DLL_comp_1[1:T])
for t in 1:T
    @constraint(model, obj_DLL_comp_1[t]==  -Rₘₐₓ[2]*πʳᵈˢᴳ³_1[t] -Rₘₐₓ[2]*πʳᵈˢᴳ³_2[t] -Rₘₐₓ[2]*πʳᵘˢᴳ³_1[t] -Rₘₐₓ[2]*πʳᵘˢᴳ³_2[t]
                                            -Rₘₐₓ[3]*πʳᵈˢᴳ⁴_1[t] -Rₘₐₓ[3]*πʳᵈˢᴳ⁴_2[t] -Rₘₐₓ[3]*πʳᵘˢᴳ⁴_1[t] -Rₘₐₓ[3]*πʳᵘˢᴳ⁴_2[t]
                                            -Rₘₐₓ[4]*πʳᵈˢᴳ⁵_1[t] -Rₘₐₓ[4]*πʳᵈˢᴳ⁵_2[t] -Rₘₐₓ[4]*πʳᵘˢᴳ⁵_1[t] -Rₘₐₓ[4]*πʳᵘˢᴳ⁵_2[t]
                                            -Rₘₐₓ[5]*πʳᵈˢᴳ²⁷_1[t] -Rₘₐₓ[5]*πʳᵈˢᴳ²⁷_2[t] -Rₘₐₓ[5]*πʳᵘˢᴳ²⁷_1[t] -Rₘₐₓ[5]*πʳᵘˢᴳ²⁷_2[t]
                                            -Rₘₐₓ[6]*πʳᵈˢᴳ³⁰_1[t] -Rₘₐₓ[6]*πʳᵈˢᴳ³⁰_2[t] -Rₘₐₓ[6]*πʳᵘˢᴳ³⁰_1[t] -Rₘₐₓ[6]*πʳᵘˢᴳ³⁰_2[t])
end

obj_DLL_comp= sum(obj_DLL_comp_1) + (P_g₀[2]*πʳᵈˢᴳ³_1[1] +P_g₀[2]*πʳᵈˢᴳ³_2[1]) -(P_g₀[2]*πʳᵘˢᴳ³_1[1] +P_g₀[2]*πʳᵘˢᴳ³_2[1]) -(yˢᴳ₀[2]*Kˢᵗ[2]*σˢᵗˢᴳ³_1[1] +yˢᴳ₀[2]*Kˢᵗ[2]*σˢᵗˢᴳ³_2[1]) +(yˢᴳ₀[2]*Kˢʰ[2]*σˢʰˢᴳ³_1[1] +yˢᴳ₀[2]*Kˢʰ[2]*σˢʰˢᴳ³_2[1]
                                    +P_g₀[3]*πʳᵈˢᴳ⁴_1[1] +P_g₀[3]*πʳᵈˢᴳ⁴_2[1]) -(P_g₀[3]*πʳᵘˢᴳ⁴_1[1] +P_g₀[3]*πʳᵘˢᴳ⁴_2[1]) -(yˢᴳ₀[3]*Kˢᵗ[3]*σˢᵗˢᴳ⁴_1[1] +yˢᴳ₀[3]*Kˢᵗ[3]*σˢᵗˢᴳ⁴_2[1]) +(yˢᴳ₀[3]*Kˢʰ[3]*σˢʰˢᴳ⁴_1[1] +yˢᴳ₀[3]*Kˢʰ[3]*σˢʰˢᴳ⁴_2[1]
                                    +P_g₀[4]*πʳᵈˢᴳ⁵_1[1] +P_g₀[4]*πʳᵈˢᴳ⁵_2[1]) -(P_g₀[4]*πʳᵘˢᴳ⁵_1[1] +P_g₀[4]*πʳᵘˢᴳ⁵_2[1]) -(yˢᴳ₀[4]*Kˢᵗ[4]*σˢᵗˢᴳ⁵_1[1] +yˢᴳ₀[4]*Kˢᵗ[4]*σˢᵗˢᴳ⁵_2[1]) +(yˢᴳ₀[4]*Kˢʰ[4]*σˢʰˢᴳ⁵_1[1] +yˢᴳ₀[4]*Kˢʰ[4]*σˢʰˢᴳ⁵_2[1]
                                    +P_g₀[5]*πʳᵈˢᴳ²⁷_1[1] +P_g₀[5]*πʳᵈˢᴳ²⁷_2[1]) -(P_g₀[5]*πʳᵘˢᴳ²⁷_1[1] +P_g₀[5]*πʳᵘˢᴳ²⁷_2[1]) -(yˢᴳ₀[5]*Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1[1] +yˢᴳ₀[5]*Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2[1]) +(yˢᴳ₀[5]*Kˢʰ[5]*σˢʰˢᴳ²⁷_1[1] +yˢᴳ₀[5]*Kˢʰ[5]*σˢʰˢᴳ²⁷_2[1]
                                    +P_g₀[6]*πʳᵈˢᴳ³⁰_1[1] +P_g₀[6]*πʳᵈˢᴳ³⁰_2[1]) -(P_g₀[6]*πʳᵘˢᴳ³⁰_1[1] +P_g₀[6]*πʳᵘˢᴳ³⁰_2[1]) -(yˢᴳ₀[6]*Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1[1] +yˢᴳ₀[6]*Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2[1]) +(yˢᴳ₀[6]*Kˢʰ[6]*σˢʰˢᴳ³⁰_1[1] +yˢᴳ₀[6]*Kˢʰ[6]*σˢʰˢᴳ³⁰_2[1])

obj_DLL=sum( obj_DLL_1 ) + obj_DLL_stra +obj_DLL_comp+ sum(Iₗᵢₘ.*λ_F[1,:])+ sum(Iₗᵢₘ.*λ_F[2,:])+ sum(Iₗᵢₘ.*λ_F[3,:])

@constraint(model, obj_LL-obj_DLL>=0)
@objective(model, Min, obj_LL-obj_DLL)  # objective function
#-------Solve and Output Results
set_optimizer(model , Gurobi.Optimizer)
# set_attribute(model, "limits/gap", 0.0280)
# set_time_limit_sec(model, 700.0)
optimize!(model)

obj_LL=JuMP.value(obj_LL)
obj_DLL=JuMP.value(obj_DLL)
DG=obj_LL-obj_DLL

r_DG_matrix=DG/obj_LL



λ_F=JuMP.value.(λ_F) 
marketclearingprices=JuMP.value.(λᴱ) 
I_₂₆=JuMP.value.(I_₂₆)

AS_price_26=λ_F[1,:]
AS_price_29=λ_F[2,:]
AS_price_30=λ_F[3,:]

yˢᴳ²_2=JuMP.value.(yˢᴳ²_2)

bar(λ_F[3,:])
print(λ_F[3,:])
bar(marketclearingprices)

plot!(λ_F[1,:])

bar(I_₂₆)


 

yˢᴳ²_1 = value.(yˢᴳ²_1)
yˢᴳ²_2 = value.(yˢᴳ²_2)
yˢᴳ³_1 = value.(yˢᴳ³_1)
yˢᴳ³_2 = value.(yˢᴳ³_2)
yˢᴳ⁴_1 = value.(yˢᴳ⁴_1)
yˢᴳ⁴_2 = value.(yˢᴳ⁴_2)
yˢᴳ⁵_1 = value.(yˢᴳ⁵_1)
yˢᴳ⁵_2 = value.(yˢᴳ⁵_2)
yˢᴳ²⁷_1 = value.(yˢᴳ²⁷_1)
yˢᴳ²⁷_2 = value.(yˢᴳ²⁷_2)
yˢᴳ³⁰_2 = value.(yˢᴳ³⁰_2)
yˢᴳ³⁰_1 = value.(yˢᴳ³⁰_1)

α₁ = value.(α₁)
α₂₃ = value.(α₂₃)
I_₂₆= value.(I_₂₆)


I_min=zeros(1,30)  
I=zeros(30,T)

for k in 1:30
for t in 1:T     # bounds for the SCL of buses  I_₂₆   I_₂₉  I_₃₀     
    I[k,t]=                  
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]*α₁[t]+ K_c[2,k]*α₂₃[t]+ K_c[3,k]*α₂₆[t]

        I_min[k]=minimum(I[k,:])
end
end

print(I_min)

bar(I_min')

matwrite("I_min_withSCL.mat", Dict("I_min_withSCL" => I_min))
matwrite("marketclearingprices_withSCL.mat", Dict("marketclearingprices_withSCL" => marketclearingprices))
