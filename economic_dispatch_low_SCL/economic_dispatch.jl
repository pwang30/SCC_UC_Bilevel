# Author: Peng Wang       from Technical University of Madrid (UPM)
# Supervisor: Luis Badesa

# Now, this is the economic_dispatch+UC WITHOUT SCL constraints on a modified IEEE-30 bus system, SCL on each bus is calculated in an offline process according to the solution
# 20.March.2025

import Pkg
Pkg.add("MAT")
using JuMP,Gurobi, CSV,DataFrames,LinearAlgebra, XLSX, IterTools, DelimitedFiles ,Plots, MAT
include("dataset_gene.jl")
include("offline_trainning.jl")
include("admittance_matrix_calculation.jl")



#-----------------------------------Define Parameters for Calculating SCL-----------------------------------

I_IBG=1    # pre-defined SCC contribution from IBG
Iₗᵢₘ=7       # SCC limit
β=1
v_n=1
v=0.9

I_SCC_all_buses_scenarios, matrix_ω = dataset_gene(I_IBG, β,v_n) # data set generation

Total_N_type_1=zeros(3,1)
Total_N_type_2=zeros(3,1)
Total_err_type_1=zeros(3,1)
Total_err_type_2=zeros(3,1)

LIM=[5 6 7]

for i in 1:3
    
    Iₗᵢₘ=LIM[i]

                   
K_g, K_c, K_m, N_type_1, N_type_2, err_type_1, err_type_2= offline_trainning(I_SCC_all_buses_scenarios, matrix_ω, Iₗᵢₘ, v)  # offline_trainning

Total_N_type_1[i]=N_type_1
Total_N_type_2[i]=N_type_2
Total_err_type_1[i]=err_type_1
Total_err_type_2[i]=err_type_2

end



#-----------------------------------Define Parameters for Optimization-----------------------------------
IBG₁=[2,1.5,1.6,1.8,1.3,0.6,2.8,3.3,3.9,4,3.3,2.9,2.7,2,0.2,3.2,5.1,3.1,1.8,2,1.3,1,2,3.8]*10^3                    # Wind_bus_1
IBG₂₃=[4.7,5.1,4.3,4.1,3.8,3.9,4,5,5,4.8,3.9,4.3,5,5.2,5.8,5.6,1.6,0.9,5.8,4.1,3.6,3.5,3.1,3.8]*10^3               # Wind_bus_23
IBG₂₆=[9.3,10.1,7.2,7.5,7.9,6.4,7.1,6.9,5.6,5.4,5.2,4,3.8,3,2.8,3.2,2.5,1.1,2.1,2.9,2.7,3,4.6,5.5]*10^3            # Wind_bus_26

Load_total=[18.42,17.95,18.29,18.51,18.13,17.88,19.46,21.97,23.17,23.87,
23.91,23.77,23.80,23.82,24.23,23.79,26.01,26.91,25.26,23.69,22.12,20.04,18.17,18.01]*10^3   # (MW)

#Pˢᴳₘₐₓ=[6.584, 5.760, 3.781, 3.335, 3.252, 2.880]*10^3     # SGs, buses:2,3,4,5,27,30
#Pˢᴳₘᵢₙ=[3.292, 2.880, 1.512, 0.667, 0.650, 0.288]*10^3
#Rₘₐₓ=[1.317, 1.152, 1.512, 1.334, 1.951, 1.728]*10^3

#Kˢᵗ=[4000, 325, 142.5, 72, 55, 31]*10^3  
#Kˢʰ=[800, 28.5, 18.5, 14.4, 12, 10]*10^3
#Oᵐ₁=[6.20, 32.10, 36.47, 64.28, 84.53, 97.36] 
#Oᵐ₂=[7.07, 34.72, 38.49, 72.84, 93.60, 105.02]
#Oⁿˡ=[18.431, 17.005 , 13.755, 9.930, 9.900, 8.570]*10^3


Pˢᴳₘₐₓ=[4.584, 4.760, 4.781, 4.335, 4.852, 4.810]*10^3     # SGs, buses:2,3,4,5,27,30
Pˢᴳₘᵢₙ=[1.292, 1.880, 1.512, 1.650, 1.667, 1.288]*10^3
Rₘₐₓ=[2.317, 2.452, 2.512, 2.551, 2.334, 2.428]*10^3


Kˢᵗ=[60, 65, 57, 55, 51, 54]*10^3  
Kˢʰ=[16, 18, 18.5, 15.4, 15, 16]*10^3
Oᵐ₁=[6.20, 7.10, 8.47, 6.13, 5.28, 7.36] 
Oᵐ₂=[7.60, 8.72, 8.49, 7.84, 7.07, 8.02]
Oⁿˡ=[9.431, 9.005 , 9.755, 9.930, 8.570, 9.900]*10^3

Oᴱ_c=[2.46 3.19 2.86]      # IBRs, buses:1,23,26
T=length(Load_total)

P_g₀=[1.292, 1.880, 1.512, 1.650, 1.667, 1.288]*10^3
yˢᴳ₀=[1 1 1 1 1 1]



#-----------------------------------Define Primal Model-----------------------------------
model= Model()



#-------Define Primal Variales
@variable(model, Pˢᴳ²_1[1:T])        # generation of SGs , buses:2,3,4,5,27,30.  
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

@variable(model, Pᴵᴮᴳ¹[1:T]>=0)         # generation of IBRs (WT) , buses:1, 23, 26 
@variable(model, Pᴵᴮᴳ²³[1:T]>=0)              
@variable(model, Pᴵᴮᴳ²⁶[1:T]>=0)  

@variable(model, yˢᴳ²_1[1:T],Bin)      # status of SGs, buses:2,3,4,5,27,30.  
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

@variable(model, α₁[1:T]>=0)                   # percentage of IBGs' online capacity 
@variable(model, α₂₃[1:T]>=0)
@variable(model, α₂₆[1:T]>=0)

@variable(model, Cᵁ²_1[1:T]>=0)                 # startup costs and shutdown costs for SGs 
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



#-------Define Primal Constraints
@constraint(model, Pˢᴳ²_1+Pˢᴳ²_2+Pˢᴳ³_1+Pˢᴳ³_2+Pˢᴳ⁴_1+Pˢᴳ⁴_2+Pˢᴳ⁵_1+Pˢᴳ⁵_2+Pˢᴳ²⁷_1+Pˢᴳ²⁷_2+Pˢᴳ³⁰_1+Pˢᴳ³⁰_2+
                   Pᴵᴮᴳ¹+Pᴵᴮᴳ²³+Pᴵᴮᴳ²⁶==Load_total)     # power balance 

@constraint(model, Pˢᴳ²_1.<=yˢᴳ²_1*Pˢᴳₘₐₓ[1])           # bounds for the output of SGs with UC 
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

@constraint(model, Pˢᴳ²_1[1]-P_g₀[1]<=Rₘₐₓ[1])        # bounds for the ramp of SGs 
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

@constraint(model, Cᵁ²_1[1]>=(yˢᴳ²_1[1]-yˢᴳ₀[1])*Kˢᵗ[1])        # startup costs and shutdown costs for SGs
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
    @constraint(model, Pᴵᴮᴳ¹[t] <= IBG₁[t]*α₁[t])        # wind power limit
    @constraint(model, Pᴵᴮᴳ²³[t]<= IBG₂₃[t]*α₂₃[t])       
    @constraint(model, Pᴵᴮᴳ²⁶[t]<= IBG₂₆[t]*α₂₆[t])       

    @constraint(model, α₁[t]<=1)                         # IBR online capacity limit
    @constraint(model, α₂₃[t]<=1)                   
    @constraint(model, α₂₆[t]<=1)                   
end

@variable(model, I_scc[1:30, 1:24]) 

for k in 1:30       # bounds for the SCL of buses
    for t in 1:T   
        @constraint(model, I_scc[k,t]==K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+

        K_c[1,k]*α₁[t]+ K_c[2,k]*α₂₃[t]+ K_c[3,k]*α₂₆[t]+
        
K_m[1,k]*yˢᴳ²_1[t]*yˢᴳ²_2[t]+ K_m[2,k]*yˢᴳ²_1[t]*yˢᴳ³_1[t]+ K_m[3,k]*yˢᴳ²_1[t]*yˢᴳ³_2[t]+ K_m[4,k]*yˢᴳ²_1[t]*yˢᴳ⁴_1[t]+
K_m[5,k]*yˢᴳ²_1[t]*yˢᴳ⁴_2[t]+ K_m[6,k]*yˢᴳ²_1[t]*yˢᴳ⁵_1[t]+ K_m[7,k]*yˢᴳ²_1[t]*yˢᴳ⁵_2[t]+ K_m[8,k]*yˢᴳ²_1[t]*yˢᴳ²⁷_1[t]+
K_m[9,k]*yˢᴳ²_1[t]*yˢᴳ²⁷_2[t]+ K_m[10,k]*yˢᴳ²_1[t]*yˢᴳ³⁰_1[t]+ K_m[11,k]*yˢᴳ²_1[t]*yˢᴳ³⁰_2[t]+

K_m[12,k]*yˢᴳ²_2[t]*yˢᴳ³_1[t]+ K_m[13,k]*yˢᴳ²_2[t]*yˢᴳ³_2[t]+ K_m[14,k]*yˢᴳ²_2[t]*yˢᴳ⁴_1[t]+
K_m[15,k]*yˢᴳ²_2[t]*yˢᴳ⁴_2[t]+ K_m[16,k]*yˢᴳ²_2[t]*yˢᴳ⁵_1[t]+ K_m[17,k]*yˢᴳ²_2[t]*yˢᴳ⁵_2[t]+ K_m[18,k]*yˢᴳ²_2[t]*yˢᴳ²⁷_1[t]+
K_m[19,k]*yˢᴳ²_2[t]*yˢᴳ²⁷_2[t]+ K_m[20,k]*yˢᴳ²_2[t]*yˢᴳ³⁰_1[t]+ K_m[21,k]*yˢᴳ²_2[t]*yˢᴳ³⁰_2[t]+

K_m[22,k]*yˢᴳ³_1[t]*yˢᴳ³_2[t]+ K_m[23,k]*yˢᴳ³_1[t]*yˢᴳ⁴_1[t]+
K_m[24,k]*yˢᴳ³_1[t]*yˢᴳ⁴_2[t]+ K_m[25,k]*yˢᴳ³_1[t]*yˢᴳ⁵_1[t]+ K_m[26,k]*yˢᴳ³_1[t]*yˢᴳ⁵_2[t]+ K_m[27,k]*yˢᴳ³_1[t]*yˢᴳ²⁷_1[t]+
K_m[28,k]*yˢᴳ³_1[t]*yˢᴳ²⁷_2[t]+ K_m[29,k]*yˢᴳ³_1[t]*yˢᴳ³⁰_1[t]+ K_m[30,k]*yˢᴳ³_1[t]*yˢᴳ³⁰_2[t]+

K_m[31,k]*yˢᴳ³_2[t]*yˢᴳ⁴_1[t]+
K_m[32,k]*yˢᴳ³_2[t]*yˢᴳ⁴_2[t]+ K_m[33,k]*yˢᴳ³_2[t]*yˢᴳ⁵_1[t]+ K_m[34,k]*yˢᴳ³_2[t]*yˢᴳ⁵_2[t]+ K_m[35,k]*yˢᴳ³_2[t]*yˢᴳ²⁷_1[t]+
K_m[36,k]*yˢᴳ³_2[t]*yˢᴳ²⁷_2[t]+ K_m[37,k]*yˢᴳ³_2[t]*yˢᴳ³⁰_1[t]+ K_m[38,k]*yˢᴳ³_2[t]*yˢᴳ³⁰_2[t]+


K_m[39,k]*yˢᴳ⁴_1[t]*yˢᴳ⁴_2[t]+ K_m[40,k]*yˢᴳ⁴_1[t]*yˢᴳ⁵_1[t]+ K_m[41,k]*yˢᴳ⁴_1[t]*yˢᴳ⁵_2[t]+ K_m[42,k]*yˢᴳ⁴_1[t]*yˢᴳ²⁷_1[t]+
K_m[43,k]*yˢᴳ⁴_1[t]*yˢᴳ²⁷_2[t]+ K_m[44,k]*yˢᴳ⁴_1[t]*yˢᴳ³⁰_1[t]+ K_m[45,k]*yˢᴳ⁴_1[t]*yˢᴳ³⁰_2[t]+


K_m[46,k]*yˢᴳ⁴_2[t]*yˢᴳ⁵_1[t]+ K_m[47,k]*yˢᴳ⁴_2[t]*yˢᴳ⁵_2[t]+ K_m[48,k]*yˢᴳ⁴_2[t]*yˢᴳ²⁷_1[t]+
K_m[49,k]*yˢᴳ⁴_2[t]*yˢᴳ²⁷_2[t]+ K_m[50,k]*yˢᴳ⁴_2[t]*yˢᴳ³⁰_1[t]+ K_m[51,k]*yˢᴳ⁴_2[t]*yˢᴳ³⁰_2[t]+

K_m[52,k]*yˢᴳ⁵_1[t]*yˢᴳ⁵_2[t]+ K_m[53,k]*yˢᴳ⁵_1[t]*yˢᴳ²⁷_1[t]+
K_m[54,k]*yˢᴳ⁵_1[t]*yˢᴳ²⁷_2[t]+ K_m[55,k]*yˢᴳ⁵_1[t]*yˢᴳ³⁰_1[t]+ K_m[56,k]*yˢᴳ⁵_1[t]*yˢᴳ³⁰_2[t]+

K_m[57,k]*yˢᴳ⁵_2[t]*yˢᴳ²⁷_1[t]+
K_m[58,k]*yˢᴳ⁵_2[t]*yˢᴳ²⁷_2[t]+ K_m[59,k]*yˢᴳ⁵_2[t]*yˢᴳ³⁰_1[t]+ K_m[60,k]*yˢᴳ⁵_2[t]*yˢᴳ³⁰_2[t]+

K_m[61,k]*yˢᴳ²⁷_1[t]*yˢᴳ²⁷_2[t]+ K_m[62,k]*yˢᴳ²⁷_1[t]*yˢᴳ³⁰_1[t]+ K_m[63,k]*yˢᴳ²⁷_1[t]*yˢᴳ³⁰_2[t]+

K_m[64,k]*yˢᴳ²⁷_2[t]*yˢᴳ³⁰_1[t]+ K_m[65,k]*yˢᴳ²⁷_2[t]*yˢᴳ³⁰_2[t]+

K_m[66,k]*yˢᴳ³⁰_1[t]*yˢᴳ³⁰_2[t])

        
        if k== 1 || k==2 || k==3 || k==4 || k==5 || k==6 || k==7 || k==27
            continue
        end
            @constraint(model, I_scc[k,t]
            >=Iₗᵢₘ)
    end
end




#-------Define Objective Function
No_load_cost=sum(Oⁿˡ[1].*(yˢᴳ²_1+yˢᴳ²_2))+sum(Oⁿˡ[2].*(yˢᴳ³_1+yˢᴳ³_2))+sum(Oⁿˡ[3].*(yˢᴳ⁴_1+yˢᴳ⁴_2))+sum(Oⁿˡ[4].*(yˢᴳ⁵_1+yˢᴳ⁵_2))+sum(Oⁿˡ[5].*(yˢᴳ²⁷_1+yˢᴳ²⁷_2))+sum(Oⁿˡ[6].*(yˢᴳ³⁰_1+yˢᴳ³⁰_2))    

Generation_cost=sum(Oᵐ₁[1].*Pˢᴳ²_1+Oᵐ₂[1].*Pˢᴳ²_2)+sum(Oᵐ₁[2].*Pˢᴳ³_1+Oᵐ₂[2].*Pˢᴳ³_2)+sum(Oᵐ₁[3].*Pˢᴳ⁴_1+Oᵐ₂[3].*Pˢᴳ⁴_2)+sum(Oᵐ₁[4].*Pˢᴳ⁵_1+Oᵐ₂[4].*Pˢᴳ⁵_2)+sum(Oᵐ₁[5].*Pˢᴳ²⁷_1+Oᵐ₂[5].*Pˢᴳ²⁷_2)+sum(Oᵐ₁[6].*Pˢᴳ³⁰_1+Oᵐ₂[6].*Pˢᴳ³⁰_2)     

Onoff_cost=sum(Cᵁ²_1)+sum(Cᴰ²_1)+sum(Cᵁ³_1)+sum(Cᴰ³_1)+sum(Cᵁ⁴_1)+sum(Cᴰ⁴_1)+sum(Cᵁ⁵_1)+sum(Cᴰ⁵_1)+sum(Cᵁ²⁷_1)+sum(Cᴰ²⁷_1)+sum(Cᵁ³⁰_1)+sum(Cᴰ³⁰_1)  +sum(Cᵁ²_2)+sum(Cᴰ²_2)+sum(Cᵁ³_2)+sum(Cᴰ³_2)+sum(Cᵁ⁴_2)+sum(Cᴰ⁴_2)+sum(Cᵁ⁵_2)+sum(Cᴰ⁵_2)+sum(Cᵁ²⁷_2)+sum(Cᴰ²⁷_2)+sum(Cᵁ³⁰_2)+sum(Cᴰ³⁰_2)       

IBR_cost=sum(Oᴱ_c[1]*Pᴵᴮᴳ¹)+sum(Oᴱ_c[2]*Pᴵᴮᴳ²³)+sum(Oᴱ_c[3]*Pᴵᴮᴳ²⁶)

@objective(model, Min, No_load_cost+ Generation_cost+ Onoff_cost+IBR_cost)  # objective function

#-------Solve and Output Results
set_optimizer(model , Gurobi.Optimizer)
# set_attribute(model, "limits/gap", 0.0280)
# set_time_limit_sec(model, 700.0)
optimize!(model)

yˢᴳ²_1_value = value.(yˢᴳ²_1)
yˢᴳ²_2_value = value.(yˢᴳ²_2)
yˢᴳ³_1_value = value.(yˢᴳ³_1)
yˢᴳ³_2_value = value.(yˢᴳ³_2)
yˢᴳ⁴_1_value = value.(yˢᴳ⁴_1)
yˢᴳ⁴_2_value = value.(yˢᴳ⁴_2)
yˢᴳ⁵_1_value = value.(yˢᴳ⁵_1)
yˢᴳ⁵_2_value = value.(yˢᴳ⁵_2)
yˢᴳ²⁷_1_value = value.(yˢᴳ²⁷_1)
yˢᴳ²⁷_2_value = value.(yˢᴳ²⁷_2)
yˢᴳ³⁰_2_value = value.(yˢᴳ³⁰_2)
yˢᴳ³⁰_1_value = value.(yˢᴳ³⁰_1)

α₁_value = value.(α₁)
α₂₃_value = value.(α₂₃)
α₂₆_value = value.(α₂₆)

commitment_SGs=zeros(24,12)
commitment_SGs[:,1]=yˢᴳ²_1_value
commitment_SGs[:,2]=yˢᴳ²_2_value
commitment_SGs[:,3]=yˢᴳ³_1_value
commitment_SGs[:,4]=yˢᴳ³_2_value
commitment_SGs[:,5]=yˢᴳ⁴_1_value
commitment_SGs[:,6]=yˢᴳ⁴_2_value
commitment_SGs[:,7]=yˢᴳ⁵_1_value
commitment_SGs[:,8]=yˢᴳ⁵_2_value
commitment_SGs[:,9]=yˢᴳ²⁷_1_value
commitment_SGs[:,10]=yˢᴳ²⁷_2_value
commitment_SGs[:,11]=yˢᴳ³⁰_1_value
commitment_SGs[:,12]=yˢᴳ³⁰_2_value

#----------------------------------Calculate min SCL in each bus----------------------------------
#include("min_SCC_each_bus.jl")
SCL_eachbus_noSCL=min_SCC_each_bus(commitment_SGs,I_IBG,β,v_n)

I_scc_value = value.(I_scc)

I_scc_min=zeros(30,1)
for i in 1:30
    I_scc_min[i]=minimum(I_scc_value[i,:])
end
minimum(I_scc_min)

I_scc_max=zeros(30,1)
for i in 1:30
    I_scc_max[i]=maximum(I_scc_value[i,:])
end
maximum(I_scc_max)

I_26_SCL_linearized=zeros(1,24)
I_26_SCL_simplified=zeros(1,24)

I_26_SCL_linearized=I_scc_value[26,:]
I_26_SCL_simplified=I_scc_value[26,:]


plot(I_26_SCL_linearized)
plot!(I_26_SCL_simplified)
bar(SCL_eachbus_noSCL', xlabel="Category", ylabel="Value", title="Bar Chart")
plot!(I_scc_min, xlabel="Category", ylabel="Value", title="Bar Chart")

matwrite("SCL_eachbus_noSCL.mat", Dict("SCL_eachbus_noSCL" => SCL_eachbus_noSCL))
matwrite("I_26_SCL_simplified.mat", Dict("I_26_SCL_simplified" => I_26_SCL_simplified))
