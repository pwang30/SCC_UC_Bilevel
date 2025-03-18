# Author: Peng Wang       from Technical University of Madrid (UPM)
# Supervisor: Luis Badesa

# Now, this is the bi-level+UC WITHOUT SCL with a modified IEEE-30 bus system, SCL on each bus is calculated in an offline process according to the solution
# The non-convexity of the model is solved by the method proposed by Yujian Ye in Imperial College London
# 18.March.2025

import Pkg
using JuMP,Gurobi, CSV,DataFrames,LinearAlgebra, XLSX, IterTools, DelimitedFiles 
#----------------------------------IEEE-30 Bus System Data Introduction----------------------------------
df = DataFrame(CSV.File( "C:/Users/ME2/Desktop/single_level_UC_SCL_pricing/Windcurves.csv") ) 
windcurves=df[:,:]

include("dataset_gene.jl")
include("offline_trainning.jl")
include("admittance_matrix_calculation.jl")
#-----------------------------------Define Parameters for Calculating SCC
IBG₁=Vector(windcurves[1, 2:end])      # total capacity of IBG, here they are wind turbines 
IBG₂₃=Vector(windcurves[2, 2:end])
IBG₂₆=Vector(windcurves[3, 2:end])
IBG₁=hcat(IBG₁)*10^8*3
IBG₂₃=hcat(IBG₂₃)*10^8*3
IBG₂₆=hcat(IBG₂₆)*10^8*3


I_IBG=1    # pre-defined SCC contribution from IBG
Iₗᵢₘ= 10       # SCC limit
β=0.95
v=1

I_SCC_all_buses_scenarios, matrix_ω = dataset_gene(I_IBG, β)                      # data set generation
K_g, K_c, K_m, N_type_1, N_type_2, err_type_1, err_type_2= offline_trainning(I_SCC_all_buses_scenarios, matrix_ω, Iₗᵢₘ, v)  # offline_trainning



#-----------------------------------Define Parameters for Optimization-----------------------------------
Load_total=[18.42,17.95,18.29,18.51,18.13,17.88,19.46,21.97,23.17,23.87,
23.91,23.77,23.80,23.82,24.23,23.79,26.01,26.91,25.26,23.69,22.12,20.04,18.17,18.01]*0.7   #(MW)

Pˢᴳₘₐₓ=[6.584, 5.760, 3.781, 3.335, 3.252, 2.880]     # SGs, buses:2,3,4,5,27,30
Pˢᴳₘᵢₙ=[3.292, 2.880, 1.512, 0.667, 0.650, 0.288]
Rₘₐₓ=[1.317, 1.152, 1.512, 1.334, 1.951, 1.728]

Kᵁ=[4000, 325, 142.5, 72, 55, 31]*10^3  
Kᴰ=[800, 28.5, 18.5, 14.4, 12, 10]*10^3
Oᵐ₁=[6.20, 32.10, 36.47, 64.28, 84.53, 97.36] 
Oᵐ₂=[7.07, 34.72, 38.49, 72.84, 93.60, 105.02]
Oⁿˡ=[18.431, 17.005, 13.755, 9.930, 9.900, 8.570]*10^3
Oᴱ_c=[2.46 3.19 2.86]      # IBRs, buses:1,23,26
T=length(Load_total)

P_g₀=[5268 4608 3025 2668 2602 0]
yˢᴳ₀=[1 1 1 1 1 0]

kᵐᵐᵃˣ_g=3

#-----------------------------------Define Primal-Dual Model-----------------------------------
model= Model()



#-------Define Primal Variales
@variable(model, kᵐ_g1[1:T])          # bidding variable of UL
@variable(model, kᵐ_g2[1:T])

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



#-------Define Dual Variales
@variable(model, λᴱ[1:T]>=0)                   

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
@constraint(model,kᵐ_g1.>=1)
@constraint(model,kᵐ_g1.<=kᵐᵐᵃˣ_g)
@constraint(model,kᵐ_g2.>=1)
@constraint(model,kᵐ_g2.<=kᵐᵐᵃˣ_g)

@constraint(model, Pˢᴳ²_1+Pˢᴳ²_2+Pˢᴳ³_1+Pˢᴳ³_2+Pˢᴳ⁴_1+Pˢᴳ⁴_2+Pˢᴳ⁵_1+Pˢᴳ⁵_2+Pˢᴳ²⁷_1+Pˢᴳ²⁷_2+Pˢᴳ³⁰_1+Pˢᴳ³⁰_2+
                   Pᴵᴮᴳ¹+Pᴵᴮᴳ²³+Pᴵᴮᴳ²⁶==Load_total)     # power balance , dual variable: λᴱₜ

@constraint(model, Pˢᴳ²_1.<=yˢᴳ²_1*Pˢᴳₘₐₓ[1])           # bounds for the output of SGs with UC , dual variables: μᵐⁱⁿₜ , μᵐᵃˣₜ
@constraint(model, yˢᴳ²_1*Pˢᴳₘᵢₙ[1].<=Pˢᴳ²_1)
@constraint(model, Pˢᴳ²_1.<=yˢᴳ²_1*Pˢᴳₘₐₓ[1])       
@constraint(model, yˢᴳ²_1*Pˢᴳₘᵢₙ[1].<=Pˢᴳ²_1)
@constraint(model, Pˢᴳ²_2.<=yˢᴳ²_2*Pˢᴳₘₐₓ[1])       
@constraint(model, yˢᴳ²_2*Pˢᴳₘᵢₙ[1].<=Pˢᴳ²_2)
@constraint(model, Pˢᴳ²_2.<=yˢᴳ²_2*Pˢᴳₘₐₓ[1])       
@constraint(model, yˢᴳ²_2*Pˢᴳₘᵢₙ[1].<=Pˢᴳ²_2)
@constraint(model, Pˢᴳ³_1.<=yˢᴳ³_1*Pˢᴳₘₐₓ[2])      
@constraint(model, yˢᴳ³_1*Pˢᴳₘᵢₙ[2].<=Pˢᴳ³_1)
@constraint(model, Pˢᴳ³_1.<=yˢᴳ³_1*Pˢᴳₘₐₓ[2])       
@constraint(model, yˢᴳ³_1*Pˢᴳₘᵢₙ[2].<=Pˢᴳ³_1)
@constraint(model, Pˢᴳ³_2.<=yˢᴳ³_2*Pˢᴳₘₐₓ[2])       
@constraint(model, yˢᴳ³_2*Pˢᴳₘᵢₙ[2].<=Pˢᴳ³_2)
@constraint(model, Pˢᴳ³_2.<=yˢᴳ³_2*Pˢᴳₘₐₓ[2])       
@constraint(model, yˢᴳ³_2*Pˢᴳₘᵢₙ[2].<=Pˢᴳ³_2)
@constraint(model, Pˢᴳ⁴_1.<=yˢᴳ⁴_1*Pˢᴳₘₐₓ[3])      
@constraint(model, yˢᴳ⁴_1*Pˢᴳₘᵢₙ[3].<=Pˢᴳ⁴_1)
@constraint(model, Pˢᴳ⁴_1.<=yˢᴳ⁴_1*Pˢᴳₘₐₓ[3])       
@constraint(model, yˢᴳ⁴_1*Pˢᴳₘᵢₙ[3].<=Pˢᴳ⁴_1)
@constraint(model, Pˢᴳ⁴_2.<=yˢᴳ⁴_2*Pˢᴳₘₐₓ[3])       
@constraint(model, yˢᴳ⁴_2*Pˢᴳₘᵢₙ[3].<=Pˢᴳ⁴_2)
@constraint(model, Pˢᴳ⁴_2.<=yˢᴳ⁴_2*Pˢᴳₘₐₓ[3])       
@constraint(model, yˢᴳ⁴_2*Pˢᴳₘᵢₙ[3].<=Pˢᴳ⁴_2)
@constraint(model, Pˢᴳ⁵_1.<=yˢᴳ⁵_1*Pˢᴳₘₐₓ[4])      
@constraint(model, yˢᴳ⁵_1*Pˢᴳₘᵢₙ[4].<=Pˢᴳ⁵_1)
@constraint(model, Pˢᴳ⁵_1.<=yˢᴳ⁵_1*Pˢᴳₘₐₓ[4])       
@constraint(model, yˢᴳ⁵_1*Pˢᴳₘᵢₙ[4].<=Pˢᴳ⁵_1)
@constraint(model, Pˢᴳ⁵_2.<=yˢᴳ⁵_2*Pˢᴳₘₐₓ[4])       
@constraint(model, yˢᴳ⁵_2*Pˢᴳₘᵢₙ[4].<=Pˢᴳ⁵_2)
@constraint(model, Pˢᴳ⁵_2.<=yˢᴳ⁵_2*Pˢᴳₘₐₓ[4])       
@constraint(model, yˢᴳ⁵_2*Pˢᴳₘᵢₙ[4].<=Pˢᴳ⁵_2)
@constraint(model, Pˢᴳ²⁷_1.<=yˢᴳ²⁷_1*Pˢᴳₘₐₓ[5])      
@constraint(model, yˢᴳ²⁷_1*Pˢᴳₘᵢₙ[5].<=Pˢᴳ²⁷_1)
@constraint(model, Pˢᴳ²⁷_1.<=yˢᴳ²⁷_1*Pˢᴳₘₐₓ[5])       
@constraint(model, yˢᴳ²⁷_1*Pˢᴳₘᵢₙ[5].<=Pˢᴳ²⁷_1)
@constraint(model, Pˢᴳ²⁷_2.<=yˢᴳ²⁷_2*Pˢᴳₘₐₓ[5])       
@constraint(model, yˢᴳ²⁷_2*Pˢᴳₘᵢₙ[5].<=Pˢᴳ²⁷_2)
@constraint(model, Pˢᴳ²⁷_2.<=yˢᴳ²⁷_2*Pˢᴳₘₐₓ[5])       
@constraint(model, yˢᴳ²⁷_2*Pˢᴳₘᵢₙ[5].<=Pˢᴳ²⁷_2)
@constraint(model, Pˢᴳ³⁰_1.<=yˢᴳ³⁰_1*Pˢᴳₘₐₓ[6])      
@constraint(model, yˢᴳ³⁰_1*Pˢᴳₘᵢₙ[6].<=Pˢᴳ³⁰_1)
@constraint(model, Pˢᴳ³⁰_1.<=yˢᴳ³⁰_1*Pˢᴳₘₐₓ[6])       
@constraint(model, yˢᴳ³⁰_1*Pˢᴳₘᵢₙ[6].<=Pˢᴳ³⁰_1)
@constraint(model, Pˢᴳ³⁰_2.<=yˢᴳ³⁰_2*Pˢᴳₘₐₓ[6])       
@constraint(model, yˢᴳ³⁰_2*Pˢᴳₘᵢₙ[6].<=Pˢᴳ³⁰_2)
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

@constraint(model, Cᵁ²_1[1]>=(yˢᴳ²_1[1]-yˢᴳ₀[1])*Kᵁ[1])        # startup costs and shutdown costs for SGs , dual variables: σˢᵗₜ , σˢʰₜ
@constraint(model, Cᴰ²_1[1]>=(yˢᴳ₀[1]-yˢᴳ²_1[1])*Kᴰ[1])  
@constraint(model, Cᵁ²_2[1]>=(yˢᴳ²_2[1]-yˢᴳ₀[1])*Kᵁ[1])        
@constraint(model, Cᴰ²_2[1]>=(yˢᴳ₀[1]-yˢᴳ²_2[1])*Kᴰ[1]) 
@constraint(model, Cᵁ³_1[1]>=(yˢᴳ³_1[1]-yˢᴳ₀[2])*Kᵁ[2])
@constraint(model, Cᴰ³_1[1]>=(yˢᴳ₀[2]-yˢᴳ³_1[1])*Kᴰ[2])
@constraint(model, Cᵁ³_2[1]>=(yˢᴳ³_2[1]-yˢᴳ₀[2])*Kᵁ[2])
@constraint(model, Cᴰ³_2[1]>=(yˢᴳ₀[2]-yˢᴳ³_2[1])*Kᴰ[2])
@constraint(model, Cᵁ⁴_1[1]>=(yˢᴳ⁴_1[1]-yˢᴳ₀[3])*Kᵁ[3])
@constraint(model, Cᴰ⁴_1[1]>=(yˢᴳ₀[3]-yˢᴳ⁴_1[1])*Kᴰ[3])
@constraint(model, Cᵁ⁴_2[1]>=(yˢᴳ⁴_2[1]-yˢᴳ₀[3])*Kᵁ[3])
@constraint(model, Cᴰ⁴_2[1]>=(yˢᴳ₀[3]-yˢᴳ⁴_2[1])*Kᴰ[3])
@constraint(model, Cᵁ⁵_1[1]>=(yˢᴳ⁵_1[1]-yˢᴳ₀[4])*Kᵁ[4])
@constraint(model, Cᴰ⁵_1[1]>=(yˢᴳ₀[4]-yˢᴳ⁵_1[1])*Kᴰ[4])
@constraint(model, Cᵁ⁵_2[1]>=(yˢᴳ⁵_2[1]-yˢᴳ₀[4])*Kᵁ[4])
@constraint(model, Cᴰ⁵_2[1]>=(yˢᴳ₀[4]-yˢᴳ⁵_2[1])*Kᴰ[4])
@constraint(model, Cᵁ²⁷_1[1]>=(yˢᴳ²⁷_1[1]-yˢᴳ₀[5])*Kᵁ[5])
@constraint(model, Cᴰ²⁷_1[1]>=(yˢᴳ₀[5]-yˢᴳ²⁷_1[1])*Kᴰ[5])
@constraint(model, Cᵁ²⁷_2[1]>=(yˢᴳ²⁷_2[1]-yˢᴳ₀[5])*Kᵁ[5])
@constraint(model, Cᴰ²⁷_2[1]>=(yˢᴳ₀[5]-yˢᴳ²⁷_2[1])*Kᴰ[5])
@constraint(model, Cᵁ³⁰_1[1]>=(yˢᴳ³⁰_1[1]-yˢᴳ₀[6])*Kᵁ[6])
@constraint(model, Cᴰ³⁰_1[1]>=(yˢᴳ₀[6]-yˢᴳ³⁰_1[1])*Kᴰ[6])
@constraint(model, Cᵁ³⁰_2[1]>=(yˢᴳ³⁰_2[1]-yˢᴳ₀[6])*Kᵁ[6])
@constraint(model, Cᴰ³⁰_2[1]>=(yˢᴳ₀[6]-yˢᴳ³⁰_2[1])*Kᴰ[6]) 
for t in 2:T
    @constraint(model, Cᵁ²_1[t]>=(yˢᴳ²_1[t]-yˢᴳ²_1[t-1])*Kᵁ[1])        
    @constraint(model, Cᴰ²_1[t]>=(yˢᴳ²_1[t-1]-yˢᴳ²_1[t])*Kᴰ[1])  
    @constraint(model, Cᵁ²_2[t]>=(yˢᴳ²_2[t]-yˢᴳ²_2[t-1])*Kᵁ[1])        
    @constraint(model, Cᴰ²_2[t]>=(yˢᴳ²_2[t-1]-yˢᴳ²_2[t])*Kᴰ[1]) 

    @constraint(model, Cᵁ³_1[t]>=(yˢᴳ³_1[t]-yˢᴳ³_1[t-1])*Kᵁ[2]) 
    @constraint(model, Cᴰ³_1[t]>=(yˢᴳ³_1[t-1]-yˢᴳ³_1[t])*Kᴰ[2])
    @constraint(model, Cᵁ³_2[t]>=(yˢᴳ³_2[t]-yˢᴳ³_2[t-1])*Kᵁ[2])
    @constraint(model, Cᴰ³_2[t]>=(yˢᴳ³_2[t-1]-yˢᴳ³_2[t])*Kᴰ[2])

    @constraint(model, Cᵁ⁴_1[t]>=(yˢᴳ⁴_1[t]-yˢᴳ⁴_1[t-1])*Kᵁ[3])
    @constraint(model, Cᴰ⁴_1[t]>=(yˢᴳ⁴_1[t-1]-yˢᴳ⁴_1[t])*Kᴰ[3])
    @constraint(model, Cᵁ⁴_2[t]>=(yˢᴳ⁴_2[t]-yˢᴳ⁴_2[t-1])*Kᵁ[3])
    @constraint(model, Cᴰ⁴_2[t]>=(yˢᴳ⁴_2[t-1]-yˢᴳ⁴_2[t])*Kᴰ[3])

    @constraint(model, Cᵁ⁵_1[t]>=(yˢᴳ⁵_1[t]-yˢᴳ⁵_1[t-1])*Kᵁ[4])
    @constraint(model, Cᴰ⁵_1[t]>=(yˢᴳ⁵_1[t-1]-yˢᴳ⁵_1[t])*Kᴰ[4])
    @constraint(model, Cᵁ⁵_2[t]>=(yˢᴳ⁵_2[t]-yˢᴳ⁵_2[t-1])*Kᵁ[4])
    @constraint(model, Cᴰ⁵_2[t]>=(yˢᴳ⁵_2[t-1]-yˢᴳ⁵_2[t])*Kᴰ[4])

    @constraint(model, Cᵁ²⁷_1[t]>=(yˢᴳ²⁷_1[t]-yˢᴳ²⁷_1[t-1])*Kᵁ[5])
    @constraint(model, Cᴰ²⁷_1[t]>=(yˢᴳ²⁷_1[t-1]-yˢᴳ²⁷_1[t])*Kᴰ[5])
    @constraint(model, Cᵁ²⁷_2[t]>=(yˢᴳ²⁷_2[t]-yˢᴳ²⁷_2[t-1])*Kᵁ[5])
    @constraint(model, Cᴰ²⁷_2[t]>=(yˢᴳ²⁷_2[t-1]-yˢᴳ²⁷_2[t])*Kᴰ[5])

    @constraint(model, Cᵁ³⁰_1[t]>=(yˢᴳ³⁰_1[t]-yˢᴳ³⁰_1[t-1])*Kᵁ[6])
    @constraint(model, Cᴰ³⁰_1[t]>=(yˢᴳ³⁰_1[t-1]-yˢᴳ³⁰_1[t])*Kᴰ[6])
    @constraint(model, Cᵁ³⁰_2[t]>=(yˢᴳ³⁰_2[t]-yˢᴳ³⁰_2[t-1])*Kᵁ[6])
    @constraint(model, Cᴰ³⁰_2[t]>=(yˢᴳ³⁰_2[t-1]-yˢᴳ³⁰_2[t])*Kᴰ[6])
end
    
for t in 1:T
    @constraint(model, Pᴵᴮᴳ¹[t] <= IBG₁[t]*α₁[t])        # wind power limit  , dual variable: ζᵐᵃˣₜ
    @constraint(model, Pᴵᴮᴳ²³[t]<= IBG₂₃[t]*α₂₃[t])       
    @constraint(model, Pᴵᴮᴳ²⁶[t]<= IBG₂₆[t]*α₂₆[t])       

    @constraint(model, α₁[t]<=1)                         # IBR online capacity limit  , dual variable:      φᵐᵃˣₜ
    @constraint(model, α₂₃[t]<=1)                   
    @constraint(model, α₂₆[t]<=1)                   
end



#-------Define Dual Constraints
@constraint(model, Oᵐ₁[1] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[1] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[1] +Kᵁ[1]*σˢᵗˢᴳ²_1[1] -Kᴰ[1]*σˢʰˢᴳ²_1[1]+ ψᵐᵃˣˢᴳ²_1[1] >=0)  # dual constraints for UC, when t==1
@constraint(model, Oᵐ₂[1] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[1] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[1] +Kᵁ[1]*σˢᵗˢᴳ²_2[1] -Kᴰ[1]*σˢʰˢᴳ²_2[1]+ ψᵐᵃˣˢᴳ²_2[1] >=0)
@constraint(model, Oᵐ₁[2] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[1] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[1] +Kᵁ[2]*σˢᵗˢᴳ³_1[1] -Kᴰ[2]*σˢʰˢᴳ³_1[1]+ ψᵐᵃˣˢᴳ³_1[1] >=0)
@constraint(model, Oᵐ₂[2] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[1] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[1] +Kᵁ[2]*σˢᵗˢᴳ³_2[1] -Kᴰ[2]*σˢʰˢᴳ³_2[1]+ ψᵐᵃˣˢᴳ³_2[1] >=0)
@constraint(model, Oᵐ₁[3] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[1] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[1] +Kᵁ[3]*σˢᵗˢᴳ⁴_1[1] -Kᴰ[3]*σˢʰˢᴳ⁴_1[1]+ ψᵐᵃˣˢᴳ⁴_1[1] >=0)
@constraint(model, Oᵐ₂[3] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[1] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[1] +Kᵁ[3]*σˢᵗˢᴳ⁴_2[1] -Kᴰ[3]*σˢʰˢᴳ⁴_2[1]+ ψᵐᵃˣˢᴳ⁴_2[1] >=0)
@constraint(model, Oᵐ₁[4] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[1] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[1] +Kᵁ[4]*σˢᵗˢᴳ⁵_1[1] -Kᴰ[4]*σˢʰˢᴳ⁵_1[1]+ ψᵐᵃˣˢᴳ⁵_1[1] >=0)
@constraint(model, Oᵐ₂[4] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[1] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[1] +Kᵁ[4]*σˢᵗˢᴳ⁵_2[1] -Kᴰ[4]*σˢʰˢᴳ⁵_2[1]+ ψᵐᵃˣˢᴳ⁵_2[1] >=0)
@constraint(model, Oᵐ₁[5] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[1] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[1] +Kᵁ[5]*σˢᵗˢᴳ²⁷_1[1] -Kᴰ[5]*σˢʰˢᴳ²⁷_1[1]+ ψᵐᵃˣˢᴳ²⁷_1[1] >=0)
@constraint(model, Oᵐ₂[5] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[1] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[1] +Kᵁ[5]*σˢᵗˢᴳ²⁷_2[1] -Kᴰ[5]*σˢʰˢᴳ²⁷_2[1]+ ψᵐᵃˣˢᴳ²⁷_2[1] >=0)
@constraint(model, Oᵐ₁[6] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[1] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[1] +Kᵁ[6]*σˢᵗˢᴳ³⁰_1[1] -Kᴰ[6]*σˢʰˢᴳ³⁰_1[1]+ ψᵐᵃˣˢᴳ³⁰_1[1] >=0)
@constraint(model, Oᵐ₂[6] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[1] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[1] +Kᵁ[6]*σˢᵗˢᴳ³⁰_2[1] -Kᴰ[6]*σˢʰˢᴳ³⁰_2[1]+ ψᵐᵃˣˢᴳ³⁰_2[1] >=0)
for t in 2:T                                                                                                                        # dual constraints for UC, when t>=2
    @constraint(model, Oᵐ₁[1]- Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[t]+ Kᵁ[1]*(σˢᵗˢᴳ²_1[t]-σˢᵗˢᴳ²_1[t-1])+ Kᴰ[1]*(σˢᵗˢᴳ²_1[t-1]-σˢᵗˢᴳ²_1[t])+ ψᵐᵃˣˢᴳ²_1[t] >=0)                
    @constraint(model, Oᵐ₂[1]- Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[t]+ Kᵁ[1]*(σˢᵗˢᴳ²_2[t]-σˢᵗˢᴳ²_2[t-1])+ Kᴰ[1]*(σˢᵗˢᴳ²_2[t-1]-σˢᵗˢᴳ²_2[t])+ ψᵐᵃˣˢᴳ²_2[t] >=0)   
    @constraint(model, Oᵐ₁[2]- Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[t]+ Kᵁ[2]*(σˢᵗˢᴳ³_1[t]-σˢᵗˢᴳ³_1[t-1])+ Kᴰ[2]*(σˢᵗˢᴳ³_1[t-1]-σˢᵗˢᴳ³_1[t])+ ψᵐᵃˣˢᴳ³_1[t] >=0)
    @constraint(model, Oᵐ₂[2]- Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[t]+ Kᵁ[2]*(σˢᵗˢᴳ³_2[t]-σˢᵗˢᴳ³_2[t-1])+ Kᴰ[2]*(σˢᵗˢᴳ³_2[t-1]-σˢᵗˢᴳ³_2[t])+ ψᵐᵃˣˢᴳ³_2[t] >=0)
    @constraint(model, Oᵐ₁[3]- Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[t]+ Kᵁ[3]*(σˢᵗˢᴳ⁴_1[t]-σˢᵗˢᴳ⁴_1[t-1])+ Kᴰ[3]*(σˢᵗˢᴳ⁴_1[t-1]-σˢᵗˢᴳ⁴_1[t])+ ψᵐᵃˣˢᴳ⁴_1[t] >=0)
    @constraint(model, Oᵐ₂[3]- Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[t]+ Kᵁ[3]*(σˢᵗˢᴳ⁴_2[t]-σˢᵗˢᴳ⁴_2[t-1])+ Kᴰ[3]*(σˢᵗˢᴳ⁴_2[t-1]-σˢᵗˢᴳ⁴_2[t])+ ψᵐᵃˣˢᴳ⁴_2[t] >=0)
    @constraint(model, Oᵐ₁[4]- Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[t]+ Kᵁ[4]*(σˢᵗˢᴳ⁵_1[t]-σˢᵗˢᴳ⁵_1[t-1])+ Kᴰ[4]*(σˢᵗˢᴳ⁵_1[t-1]-σˢᵗˢᴳ⁵_1[t])+ ψᵐᵃˣˢᴳ⁵_1[t] >=0)
    @constraint(model, Oᵐ₂[4]- Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[t]+ Kᵁ[4]*(σˢᵗˢᴳ⁵_2[t]-σˢᵗˢᴳ⁵_2[t-1])+ Kᴰ[4]*(σˢᵗˢᴳ⁵_2[t-1]-σˢᵗˢᴳ⁵_2[t])+ ψᵐᵃˣˢᴳ⁵_2[t] >=0)
    @constraint(model, Oᵐ₁[5]- Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[t]+ Kᵁ[5]*(σˢᵗˢᴳ²⁷_1[t]-σˢᵗˢᴳ²⁷_1[t-1])+ Kᴰ[5]*(σˢᵗˢᴳ²⁷_1[t-1]-σˢᵗˢᴳ²⁷_1[t])+ ψᵐᵃˣˢᴳ²⁷_1[t] >=0)
    @constraint(model, Oᵐ₂[5]- Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[t]+ Kᵁ[5]*(σˢᵗˢᴳ²⁷_2[t]-σˢᵗˢᴳ²⁷_2[t-1])+ Kᴰ[5]*(σˢᵗˢᴳ²⁷_2[t-1]-σˢᵗˢᴳ²⁷_2[t])+ ψᵐᵃˣˢᴳ²⁷_2[t] >=0)
    @constraint(model, Oᵐ₁[6]- Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[t]+ Kᵁ[6]*(σˢᵗˢᴳ³⁰_1[t]-σˢᵗˢᴳ³⁰_1[t-1])+ Kᴰ[6]*(σˢᵗˢᴳ³⁰_1[t-1]-σˢᵗˢᴳ³⁰_1[t])+ ψᵐᵃˣˢᴳ³⁰_1[t] >=0)
    @constraint(model, Oᵐ₂[6]- Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[t]+ Kᵁ[6]*(σˢᵗˢᴳ³⁰_2[t]-σˢᵗˢᴳ³⁰_2[t-1])+ Kᴰ[6]*(σˢᵗˢᴳ³⁰_2[t-1]-σˢᵗˢᴳ³⁰_2[t])+ ψᵐᵃˣˢᴳ³⁰_2[t] >=0)          
end

@constraint(model, Oᵐ₁[1]*kᵐ_g1[1] -λᴱ[1] +μᵐᵃˣˢᴳ²_1[1] -μᵐⁱⁿˢᴳ²_1[1] +πʳᵘˢᴳ²_1[1] -πʳᵈˢᴳ²_1[1]>=0)  # dual constraints for generation, when t==1, assume SGs in bus 2 are strategic
@constraint(model, Oᵐ₁[1]*kᵐ_g2[1] -λᴱ[1] +μᵐᵃˣˢᴳ²_2[1] -μᵐⁱⁿˢᴳ²_2[1] +πʳᵘˢᴳ²_2[1] -πʳᵈˢᴳ²_2[1]>=0)
@constraint(model, Oᵐ₁[2] -λᴱ[1] +μᵐᵃˣˢᴳ³_1[1] -μᵐⁱⁿˢᴳ³_1[1] +πʳᵘˢᴳ³_1[1] -πʳᵈˢᴳ³_1[1]>=0)
@constraint(model, Oᵐ₁[2] -λᴱ[1] +μᵐᵃˣˢᴳ³_2[1] -μᵐⁱⁿˢᴳ³_2[1] +πʳᵘˢᴳ³_2[1] -πʳᵈˢᴳ³_2[1]>=0)
@constraint(model, Oᵐ₁[3] -λᴱ[1] +μᵐᵃˣˢᴳ⁴_1[1] -μᵐⁱⁿˢᴳ⁴_1[1] +πʳᵘˢᴳ⁴_1[1] -πʳᵈˢᴳ⁴_1[1]>=0)
@constraint(model, Oᵐ₁[3] -λᴱ[1] +μᵐᵃˣˢᴳ⁴_2[1] -μᵐⁱⁿˢᴳ⁴_2[1] +πʳᵘˢᴳ⁴_2[1] -πʳᵈˢᴳ⁴_2[1]>=0)
@constraint(model, Oᵐ₁[4] -λᴱ[1] +μᵐᵃˣˢᴳ⁵_1[1] -μᵐⁱⁿˢᴳ⁵_1[1] +πʳᵘˢᴳ⁵_1[1] -πʳᵈˢᴳ⁵_1[1]>=0)
@constraint(model, Oᵐ₁[4] -λᴱ[1] +μᵐᵃˣˢᴳ⁵_2[1] -μᵐⁱⁿˢᴳ⁵_2[1] +πʳᵘˢᴳ⁵_2[1] -πʳᵈˢᴳ⁵_2[1]>=0)
@constraint(model, Oᵐ₁[5] -λᴱ[1] +μᵐᵃˣˢᴳ²⁷_1[1] -μᵐⁱⁿˢᴳ²⁷_1[1] +πʳᵘˢᴳ²⁷_1[1] -πʳᵈˢᴳ²⁷_1[1]>=0)
@constraint(model, Oᵐ₁[5] -λᴱ[1] +μᵐᵃˣˢᴳ²⁷_2[1] -μᵐⁱⁿˢᴳ²⁷_2[1] +πʳᵘˢᴳ²⁷_2[1] -πʳᵈˢᴳ²⁷_2[1]>=0)
@constraint(model, Oᵐ₁[6] -λᴱ[1] +μᵐᵃˣˢᴳ³⁰_1[1] -μᵐⁱⁿˢᴳ³⁰_1[1] +πʳᵘˢᴳ³⁰_1[1] -πʳᵈˢᴳ³⁰_1[1]>=0)
@constraint(model, Oᵐ₁[6] -λᴱ[1] +μᵐᵃˣˢᴳ³⁰_2[1] -μᵐⁱⁿˢᴳ³⁰_2[1] +πʳᵘˢᴳ³⁰_2[1] -πʳᵈˢᴳ³⁰_2[1]>=0)
for t in 2:T                                                                                           # dual constraints for generation, when t>=2, assume SGs in bus 2 are strategic
    @constraint(model, Oᵐ₁[1]*kᵐ_g1[t] -λᴱ[t] +μᵐᵃˣˢᴳ²_1[t] -μᵐⁱⁿˢᴳ²_1[t] +πʳᵘˢᴳ²_1[t] -πʳᵘˢᴳ²_1[t-1] -πʳᵈˢᴳ²_1[t] +πʳᵈˢᴳ²_1[t-1]>=0)                
    @constraint(model, Oᵐ₁[1]*kᵐ_g2[t] -λᴱ[t] +μᵐᵃˣˢᴳ²_2[t] -μᵐⁱⁿˢᴳ²_2[t] +πʳᵘˢᴳ²_2[t] -πʳᵘˢᴳ²_2[t-1] -πʳᵈˢᴳ²_2[t] +πʳᵈˢᴳ²_2[t-1]>=0)
    @constraint(model, Oᵐ₁[2] -λᴱ[t] +μᵐᵃˣˢᴳ³_1[t] -μᵐⁱⁿˢᴳ³_1[t] +πʳᵘˢᴳ³_1[t] -πʳᵘˢᴳ³_1[t-1] -πʳᵈˢᴳ³_1[t] +πʳᵈˢᴳ³_1[t-1]>=0)                
    @constraint(model, Oᵐ₁[2] -λᴱ[t] +μᵐᵃˣˢᴳ³_2[t] -μᵐⁱⁿˢᴳ³_2[t] +πʳᵘˢᴳ³_2[t] -πʳᵘˢᴳ³_2[t-1] -πʳᵈˢᴳ³_2[t] +πʳᵈˢᴳ³_2[t-1]>=0)
    @constraint(model, Oᵐ₁[3] -λᴱ[t] +μᵐᵃˣˢᴳ⁴_1[t] -μᵐⁱⁿˢᴳ⁴_1[t] +πʳᵘˢᴳ⁴_1[t] -πʳᵘˢᴳ⁴_1[t-1] -πʳᵈˢᴳ⁴_1[t] +πʳᵈˢᴳ⁴_1[t-1]>=0)
    @constraint(model, Oᵐ₁[3] -λᴱ[t] +μᵐᵃˣˢᴳ⁴_2[t] -μᵐⁱⁿˢᴳ⁴_2[t] +πʳᵘˢᴳ⁴_2[t] -πʳᵘˢᴳ⁴_2[t-1] -πʳᵈˢᴳ⁴_2[t] +πʳᵈˢᴳ⁴_2[t-1]>=0)
    @constraint(model, Oᵐ₁[4] -λᴱ[t] +μᵐᵃˣˢᴳ⁵_1[t] -μᵐⁱⁿˢᴳ⁵_1[t] +πʳᵘˢᴳ⁵_1[t] -πʳᵘˢᴳ⁵_1[t-1] -πʳᵈˢᴳ⁵_1[t] +πʳᵈˢᴳ⁵_1[t-1]>=0)
    @constraint(model, Oᵐ₁[4] -λᴱ[t] +μᵐᵃˣˢᴳ⁵_2[t] -μᵐⁱⁿˢᴳ⁵_2[t] +πʳᵘˢᴳ⁵_2[t] -πʳᵘˢᴳ⁵_2[t-1] -πʳᵈˢᴳ⁵_2[t] +πʳᵈˢᴳ⁵_2[t-1]>=0)
    @constraint(model, Oᵐ₁[5] -λᴱ[t] +μᵐᵃˣˢᴳ²⁷_1[t] -μᵐⁱⁿˢᴳ²⁷_1[t] +πʳᵘˢᴳ²⁷_1[t] -πʳᵘˢᴳ²⁷_1[t-1] -πʳᵈˢᴳ²⁷_1[t] +πʳᵈˢᴳ²⁷_1[t-1]>=0)
    @constraint(model, Oᵐ₁[5] -λᴱ[t] +μᵐᵃˣˢᴳ²⁷_2[t] -μᵐⁱⁿˢᴳ²⁷_2[t] +πʳᵘˢᴳ²⁷_2[t] -πʳᵘˢᴳ²⁷_2[t-1] -πʳᵈˢᴳ²⁷_2[t] +πʳᵈˢᴳ²⁷_2[t-1]>=0)
    @constraint(model, Oᵐ₁[6] -λᴱ[t] +μᵐᵃˣˢᴳ³⁰_1[t] -μᵐⁱⁿˢᴳ³⁰_1[t] +πʳᵘˢᴳ³⁰_1[t] -πʳᵘˢᴳ³⁰_1[t-1] -πʳᵈˢᴳ³⁰_1[t] +πʳᵈˢᴳ³⁰_1[t-1]>=0)
    @constraint(model, Oᵐ₁[6] -λᴱ[t] +μᵐᵃˣˢᴳ³⁰_2[t] -μᵐⁱⁿˢᴳ³⁰_2[t] +πʳᵘˢᴳ³⁰_2[t] -πʳᵘˢᴳ³⁰_2[t-1] -πʳᵈˢᴳ³⁰_2[t] +πʳᵈˢᴳ³⁰_2[t-1]>=0)                
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

for t in 1:T                                            # dual constraints for online capacity factor of IBRs
    @constraint(model, φᵐᵃˣ¹[t] -IBG₁[t]*ζᵐᵃˣ¹[t]>=0)
    @constraint(model, φᵐᵃˣ²³[t] -IBG₂₃[t]*ζᵐᵃˣ²³[t]>=0)
    @constraint(model, φᵐᵃˣ²⁶[t] -IBG₂₆[t]*ζᵐᵃˣ²⁶[t]>=0)
end



#-------Define Objective Functions
@variable(model, z₁[1:T])    # define auxiliary variables of McCormick envelopes
@variable(model, z₂[1:T])
@variable(model, z₃[1:T])
@variable(model, z₄[1:T])

@constraint(model, z₁[t]>=λᴱ[t]*Pˢᴳₘᵢₙ[1])
@constraint(model, z₁[t]>=Oᵐ₁[1]*kᵐᵐᵃˣ_g*(Pˢᴳ²_1[t]-Pˢᴳₘₐₓ[1]) +λᴱ[t]*Pˢᴳₘₐₓ[1])
@constraint(model, z₁[t]<=λᴱ[t]*Pˢᴳₘₐₓ[1])
@constraint(model, z₁[t]<=Oᵐ₁[1]*kᵐᵐᵃˣ_g*(Pˢᴳ²_1[t]-Pˢᴳₘᵢₙ[1]) +λᴱ[t]*Pˢᴳₘᵢₙ[1])

@constraint(model, z₂[t]>=λᴱ[t]*Pˢᴳₘᵢₙ[1])
@constraint(model, z₂[t]>=Oᵐ₂[1]*kᵐᵐᵃˣ_g*(Pˢᴳ²_2[t]-Pˢᴳₘₐₓ[1]) +λᴱ[t]*Pˢᴳₘₐₓ[1])
@constraint(model, z₂[t]<=λᴱ[t]*Pˢᴳₘₐₓ[1])
@constraint(model, z₂[t]<=Oᵐ₂[1]*kᵐᵐᵃˣ_g*(Pˢᴳ²_2[t]-Pˢᴳₘᵢₙ[1]) +λᴱ[t]*Pˢᴳₘᵢₙ[1])

@constraint(model, z₃[t]>=Oᵐ₁[1]*(Pˢᴳ²_1[t]+kᵐ_g1[t]*Pˢᴳₘᵢₙ[1]-Pˢᴳₘᵢₙ[1]))
@constraint(model, z₃[t]>=Oᵐ₁[1]*(kᵐᵐᵃˣ_g*Pˢᴳ²_1[t]+kᵐ_g1[t]*Pˢᴳₘₐₓ[1]-kᵐᵐᵃˣ_g*Pˢᴳₘₐₓ[1]))
@constraint(model, z₃[t]<=Oᵐ₁[1]*(kᵐᵐᵃˣ_g*Pˢᴳ²_1[t]+kᵐ_g1[t]*Pˢᴳₘᵢₙ[1]-kᵐᵐᵃˣ_g*Pˢᴳₘᵢₙ[1]))
@constraint(model, z₃[t]<=Oᵐ₁[1]*(Pˢᴳ²_1[t]+kᵐ_g1[t]*Pˢᴳₘₐₓ[1]-Pˢᴳₘₐₓ[1]))

@constraint(model, z₄[t]>=Oᵐ₂[1]*(Pˢᴳ²_2[t]+kᵐ_g2[t]*Pˢᴳₘᵢₙ[1]-Pˢᴳₘᵢₙ[1]))
@constraint(model, z₄[t]>=Oᵐ₂[1]*(kᵐᵐᵃˣ_g*Pˢᴳ²_2[t]+kᵐ_g2[t]*Pˢᴳₘₐₓ[1]-kᵐᵐᵃˣ_g*Pˢᴳₘₐₓ[1]))
@constraint(model, z₄[t]<=Oᵐ₂[1]*(kᵐᵐᵃˣ_g*Pˢᴳ²_2[t]+kᵐ_g2[t]*Pˢᴳₘᵢₙ[1]-kᵐᵐᵃˣ_g*Pˢᴳₘᵢₙ[1]))
@constraint(model, z₄[t]<=Oᵐ₂[1]*(Pˢᴳ²_2[t]+kᵐ_g2[t]*Pˢᴳₘₐₓ[1]-Pˢᴳₘₐₓ[1]))

cost_nl_UL=sum(Oⁿˡ[1].*(yˢᴳ²_1+yˢᴳ²_2))                       # no-load cost of strategic SGs
cost_gene_UL=sum(Oᵐ₁[1].*Pˢᴳ²_1+Oᵐ₂[1].*Pˢᴳ²_2)               # generation cost of strategic SGs
cost_onoff_UL=sum(Cᵁ²_1)+sum(Cᴰ²_1)+sum(Cᵁ²_2)+sum(Cᴰ²_2)     # on/off cost of strategic SGs
revenue_marketclearing_UL=sum(z₁)+sum(z₂)                         # revenue from market clearing
obj_UL=revenue_marketclearing_UL -cost_nl_UL -cost_gene_UL -cost_onoff_UL    # objective function of UL

cost_onoff_LL=sum(Cᵁ²_1)+sum(Cᴰ²_1)+sum(Cᵁ³_1)+sum(Cᴰ³_1)+sum(Cᵁ⁴_1)+sum(Cᴰ⁴_1)+sum(Cᵁ⁵_1)+sum(Cᴰ⁵_1)+sum(Cᵁ²⁷_1)+sum(Cᴰ²⁷_1)+sum(Cᵁ³⁰_1)+sum(Cᴰ³⁰_1)  +sum(Cᵁ²_2)+sum(Cᴰ²_2)+sum(Cᵁ³_2)+sum(Cᴰ³_2)+sum(Cᵁ⁴_2)+sum(Cᴰ⁴_2)+sum(Cᵁ⁵_2)+sum(Cᴰ⁵_2)+sum(Cᵁ²⁷_2)+sum(Cᴰ²⁷_2)+sum(Cᵁ³⁰_2)+sum(Cᴰ³⁰_2)       
cost_nl_LL=sum(Oⁿˡ[1].*(yˢᴳ²_1+yˢᴳ²_2))+sum(Oⁿˡ[2].*(yˢᴳ³_1+yˢᴳ³_2))+sum(Oⁿˡ[3].*(yˢᴳ⁴_1+yˢᴳ⁴_2))+sum(Oⁿˡ[4].*(yˢᴳ⁵_1+yˢᴳ⁵_2))+sum(Oⁿˡ[5].*(yˢᴳ²⁷_1+yˢᴳ²⁷_2))+sum(Oⁿˡ[6].*(yˢᴳ³⁰_1+yˢᴳ³⁰_2))    
cost_gene_LL=sum(z₃[t]+z₄[t])+sum(Oᵐ₁[2].*Pˢᴳ³_1+Oᵐ₂[2].*Pˢᴳ³_2)+sum(Oᵐ₁[3].*Pˢᴳ⁴_1+Oᵐ₂[3].*Pˢᴳ⁴_2)+sum(Oᵐ₁[4].*Pˢᴳ⁵_1+Oᵐ₂[4].*Pˢᴳ⁵_2)+sum(Oᵐ₁[5].*Pˢᴳ²⁷_1+Oᵐ₂[5].*Pˢᴳ²⁷_2)+sum(Oᵐ₁[6].*Pˢᴳ³⁰_1+Oᵐ₂[6].*Pˢᴳ³⁰_2)   
cost_IBR_LL=sum(Oᴱ_c[1].*Pᴵᴮᴳ¹) +sum(Oᴱ_c[2].*Pᴵᴮᴳ²³) +sum(Oᴱ_c[3].*Pᴵᴮᴳ²⁶)
obj_LL=cost_onoff_LL+cost_nl_LL+cost_gene_LL+cost_IBR_LL

obj_DLL=sum( Load_total.*λᴱ .-φᵐᵃˣ¹ .-φᵐᵃˣ²³ .-φᵐᵃˣ²⁶ ) +

W=10
@objective(model, Max, obj_UL-W*(obj_LL-obj_DLL))  # objective function
#-------Solve and Output Results
set_optimizer(model , Gurobi.Optimizer)
# set_attribute(model, "limits/gap", 0.0280)
# set_time_limit_sec(model, 700.0)

optimize!(model)






# for i in 1:T
    # println("对偶变量 λ$i (constraint_$i): ", dual(constraints[i]))
# end


#-----------------------------------Calculate SCC at each bus





# I_scc=JuMP.value.(I_scc)
# min_value, min_index= findmin(I_scc)

# λ_scc=JuMP.value.(λ_scc)
# max_value, max_index= findmax(λ_scc)

# α₂₆=JuMP.value.(α₂₆)
 # yˢᴳ³=JuMP.value.(yˢᴳ³)
 # yˢᴳ⁴=JuMP.value.(yˢᴳ⁴)
 # yˢᴳ⁵=JuMP.value.(yˢᴳ⁵)
 # yˢᴳ²⁷=JuMP.value.(yˢᴳ²⁷)
 # yˢᴳ³⁰=JuMP.value.(yˢᴳ³⁰)

 # α₁=JuMP.value.(α₁)
 # α₂₃=JuMP.value.(α₂₃)
 # α₂₆=JuMP.value.(α₂₆)
# α₂₆=JuMP.value.(α₂₆)
#Z=JuMP.value.(Z)
#println(value.( Z[1,3,1]))

