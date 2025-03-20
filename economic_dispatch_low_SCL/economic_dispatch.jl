# Author: Peng Wang       from Technical University of Madrid (UPM)
# Supervisor: Luis Badesa

# Now, this is the economic_dispatch+UC WITHOUT SCL constraints on a modified IEEE-30 bus system, SCL on each bus is calculated in an offline process according to the solution
# 20.March.2025

import Pkg
using JuMP,Gurobi, CSV,DataFrames,LinearAlgebra, XLSX, IterTools, DelimitedFiles 
#----------------------------------IEEE-30 Bus System Data Introduction----------------------------------
df = DataFrame(CSV.File( "C:/Users/ME2/Desktop/economic_dispatch_weak_area/Windcurves.csv") ) 
windcurves=df[:,:]



#-----------------------------------Define Parameters for Calculating SCC-----------------------------------
IBG₁=Vector(windcurves[1, 2:end])      # total capacity of IBG, here they are wind turbines 
IBG₂₃=Vector(windcurves[2, 2:end])
IBG₂₆=Vector(windcurves[3, 2:end])
IBG₁=hcat(IBG₁)*10^3
IBG₂₃=hcat(IBG₂₃)*10^3
IBG₂₆=hcat(IBG₂₆)*10^3


I_IBG=1    # pre-defined SCC contribution from IBG
Iₗᵢₘ= 10       # SCC limit
β=0.95
v=1

I_SCC_all_buses_scenarios, matrix_ω = dataset_gene(I_IBG, β)                      # data set generation
K_g, K_c, K_m, N_type_1, N_type_2, err_type_1, err_type_2= offline_trainning(I_SCC_all_buses_scenarios, matrix_ω, Iₗᵢₘ, v)  # offline_trainning



#-----------------------------------Define Parameters for Optimization-----------------------------------
Load_total=[18.42,17.95,18.29,18.51,18.13,17.88,19.46,21.97,23.17,23.87,
23.91,23.77,23.80,23.82,24.23,23.79,26.01,26.91,25.26,23.69,22.12,20.04,18.17,18.01]*10^3*1.2   # (MW)

Pˢᴳₘₐₓ=[6.584, 5.760, 3.781, 3.335, 3.252, 2.880]*10^3     # SGs, buses:2,3,4,5,27,30
Pˢᴳₘᵢₙ=[3.292, 2.880, 1.512, 0.667, 0.650, 0.288]*10^3
Rₘₐₓ=[1.317, 1.152, 1.512, 1.334, 1.951, 1.728]*10^3

Kˢᵗ=[4000, 325, 142.5, 72, 55, 31]*10^3  
Kˢʰ=[800, 28.5, 18.5, 14.4, 12, 10]*10^3
Oᵐ₁=[6.20, 32.10, 36.47, 64.28, 84.53, 97.36] 
Oᵐ₂=[7.07, 34.72, 38.49, 72.84, 93.60, 105.02]
Oⁿˡ=[18.431, 17.005 , 13.755, 9.930, 9.900, 8.570]*10^3
Oᴱ_c=[2.46 3.19 2.86]      # IBRs, buses:1,23,26
T=length(Load_total)

P_g₀=[5.268 4.608 3.025 2.668 2.602 0]*10^3
yˢᴳ₀=[1 1 1 1 1 0]



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

