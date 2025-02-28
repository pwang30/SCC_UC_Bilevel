# Author: Peng Wang       from Technical University of Madrid (UPM)
# Supervisor: Luis Badesa

# Now, this is single-level+UC+SCL TEST CASE with a modified IEEE-30 bus system
# 03.Dec.2024

import Pkg
using JuMP,Gurobi, CSV,DataFrames,LinearAlgebra, XLSX, IterTools, DelimitedFiles 
#----------------------------------IEEE-30 Bus System Data Introduction----------------------------------
df = DataFrame(CSV.File("/Users/kl/Desktop/single_level_UC_SCL/Loadcurves.csv"))        
loadcurve=df[:,:]  
df = DataFrame(CSV.File( "/Users/kl/Desktop/single_level_UC_SCL/Windcurves.csv") ) 
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
Iₗᵢₘ= 10     # SCC limit
β=0.95
v=1

I_SCC_all_buses_scenarios, matrix_ω = dataset_gene(I_IBG, β)  # data set generation
# matwrite("C:/Users/ME2/Desktop/scl_ieee30/I_scc_all.mat", Dict("I_scc_all" => I_scc_all))   # save as format of .mat in Matlab

K_g, K_c, K_m, N_type_1, N_type_2, err_type_1, err_type_2= offline_trainning(I_SCC_all_buses_scenarios, matrix_ω, Iₗᵢₘ, v)  # offline_trainning



#-----------------------------------Define Parameters for Optimization
Load_total=[18.42,17.95,18.29,18.51,18.13,17.88,19.46,21.97,23.17,23.87,
23.91,23.77,23.80,23.82,24.23,23.79,26.01,26.91,25.26,23.69,22.12,20.04,18.17,18.01]*10^9
Pˢᴳₘₐₓ=[6.584, 5.760, 3.781, 3.335, 3.252, 2.880]*10^9     # SGs, buses:2,3,4,5,27,30
Pˢᴳₘᵢₙ=[3.292, 2.880, 1.512, 0.667, 0.650, 0.288]*10^9
Rₘₐₓ=[1.317, 1.152, 1.512, 1.334, 1.951, 1.728]*10^9

Kᵁ=[4000, 325, 142.5, 72, 55, 31]/5*10^3
Kᴰ=[800, 28.5, 18.5, 14.4, 12, 10]/5*10^3
Cᵍᵐ₁=[6.20, 32.10, 36.47, 64.28, 84.53, 97.36]/10^6
Cᵍᵐ₂=[7.07, 34.72, 38.49, 72.84, 93.60, 105.02]/10^6
Cⁿˡ=[18.431, 17.005, 13.755, 9.930, 9.900, 8.570]/5*10^3
T=length(Load_total)


     
#-----------------------------------Define Model
model= Model()


#-------Define Variales
@variable(model, Pˢᴳ²_1[1:T])     # generation of SG 
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

@variable(model, yˢᴳ²_1[1:T],Bin)   # status of SGs
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

@variable(model, Pᴵᴮᴳ¹[1:T])               # generation of IBG (WT) 
@variable(model, Pᴵᴮᴳ²³[1:T])              
@variable(model, Pᴵᴮᴳ²⁶[1:T])               

@variable(model, Cᵁ²_1[1:T]>=0)                 # startup costs for SG
@variable(model, Cᵁ²_2[1:T]>=0)                 
@variable(model, Cᴰ²_1[1:T]>=0)                 # shutdown costs for SG 
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

@variable(model, α₁[1:1])                   # percentage of IBGs' online capacity  
@variable(model, α₂₃[1:1])
@variable(model, α₂₆[1:1])



#-------Define Constraints
@constraint(model, Pˢᴳ²_1+Pˢᴳ²_2+Pˢᴳ³_1+Pˢᴳ³_2+Pˢᴳ⁴_1+Pˢᴳ⁴_2+Pˢᴳ⁵_1+Pˢᴳ⁵_2+Pˢᴳ²⁷_1+Pˢᴳ²⁷_2+Pˢᴳ³⁰_1+Pˢᴳ³⁰_2+
                    Pᴵᴮᴳ¹+Pᴵᴮᴳ²³+Pᴵᴮᴳ²⁶==Load_total)     # power balance              

@constraint(model, Pᴵᴮᴳ¹ .<= IBG₁*α₁)        # wind power limit
@constraint(model, Pᴵᴮᴳ¹>=0)
@constraint(model, Pᴵᴮᴳ²³.<= IBG₂₃*α₂₃)       
@constraint(model, Pᴵᴮᴳ²³>=0)
@constraint(model, Pᴵᴮᴳ²⁶.<= IBG₂₆*α₂₆)       
@constraint(model, Pᴵᴮᴳ²⁶>=0)

@constraint(model, α₁.<=1)                   # IBG online capacity limit
@constraint(model, α₁>=0)
@constraint(model, α₂₃.<=1)                   
@constraint(model, α₂₃>=0)
@constraint(model, α₂₆.<=1)                   
@constraint(model, α₂₆>=0)


@constraint(model, Pˢᴳ²_1.<=yˢᴳ²_1*Pˢᴳₘₐₓ[1])       # bounds for the output of SGs
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
 

for t in 1:T-1
@constraint(model, Cᵁ²_1[t]>=(yˢᴳ²_1[t+1]-yˢᴳ²_1[t])*Kᵁ[1])        
@constraint(model, Cᴰ²_1[t]>=(yˢᴳ²_1[t]-yˢᴳ²_1[t+1])*Kᴰ[1])  
@constraint(model, Cᵁ²_2[t]>=(yˢᴳ²_2[t+1]-yˢᴳ²_2[t])*Kᵁ[1])        
@constraint(model, Cᴰ²_2[t]>=(yˢᴳ²_2[t]-yˢᴳ²_2[t+1])*Kᴰ[1])  
@constraint(model, Cᵁ³_1[t]>=(yˢᴳ³_1[t+1]-yˢᴳ³_1[t])*Kᵁ[2])        
@constraint(model, Cᴰ³_1[t]>=(yˢᴳ³_1[t]-yˢᴳ³_1[t+1])*Kᴰ[2])  
@constraint(model, Cᵁ³_2[t]>=(yˢᴳ³_2[t+1]-yˢᴳ³_2[t])*Kᵁ[2])        
@constraint(model, Cᴰ³_2[t]>=(yˢᴳ³_2[t]-yˢᴳ³_2[t+1])*Kᴰ[2]) 
@constraint(model, Cᵁ⁴_1[t]>=(yˢᴳ⁴_1[t+1]-yˢᴳ⁴_1[t])*Kᵁ[3])        
@constraint(model, Cᴰ⁴_1[t]>=(yˢᴳ⁴_1[t]-yˢᴳ⁴_1[t+1])*Kᴰ[3]) 
@constraint(model, Cᵁ⁴_2[t]>=(yˢᴳ⁴_2[t+1]-yˢᴳ⁴_2[t])*Kᵁ[3])        
@constraint(model, Cᴰ⁴_2[t]>=(yˢᴳ⁴_2[t]-yˢᴳ⁴_2[t+1])*Kᴰ[3]) 
@constraint(model, Cᵁ⁵_1[t]>=(yˢᴳ⁵_1[t+1]-yˢᴳ⁵_1[t])*Kᵁ[4])        
@constraint(model, Cᴰ⁵_1[t]>=(yˢᴳ⁵_1[t]-yˢᴳ⁵_1[t+1])*Kᴰ[4])   
@constraint(model, Cᵁ⁵_2[t]>=(yˢᴳ⁵_2[t+1]-yˢᴳ⁵_2[t])*Kᵁ[4])        
@constraint(model, Cᴰ⁵_2[t]>=(yˢᴳ⁵_2[t]-yˢᴳ⁵_2[t+1])*Kᴰ[4]) 
@constraint(model, Cᵁ²⁷_1[t]>=(yˢᴳ²⁷_1[t+1]-yˢᴳ²⁷_1[t])*Kᵁ[5])        
@constraint(model, Cᴰ²⁷_1[t]>=(yˢᴳ²⁷_1[t]-yˢᴳ²⁷_1[t+1])*Kᴰ[5])
@constraint(model, Cᵁ²⁷_2[t]>=(yˢᴳ²⁷_2[t+1]-yˢᴳ²⁷_2[t])*Kᵁ[5])        
@constraint(model, Cᴰ²⁷_2[t]>=(yˢᴳ²⁷_2[t]-yˢᴳ²⁷_2[t+1])*Kᴰ[5])
@constraint(model, Cᵁ³⁰_1[t]>=(yˢᴳ³⁰_1[t+1]-yˢᴳ³⁰_1[t])*Kᵁ[6])        
@constraint(model, Cᴰ³⁰_1[t]>=(yˢᴳ³⁰_1[t]-yˢᴳ³⁰_1[t+1])*Kᴰ[6])
@constraint(model, Cᵁ³⁰_2[t]>=(yˢᴳ³⁰_2[t+1]-yˢᴳ³⁰_2[t])*Kᵁ[6])        
@constraint(model, Cᴰ³⁰_2[t]>=(yˢᴳ³⁰_2[t]-yˢᴳ³⁰_2[t+1])*Kᴰ[6])
end


for t in 1:T-1          # bounds for the ramp of SGs
    @constraint(model, Pˢᴳ²_1[t+1]-Pˢᴳ²_1[t]<=Rₘₐₓ[1])        
    @constraint(model, -Rₘₐₓ[1]<=Pˢᴳ²_1[t+1]-Pˢᴳ²_1[t])  
    @constraint(model, Pˢᴳ²_2[t+1]-Pˢᴳ²_2[t]<=Rₘₐₓ[1])        
    @constraint(model, -Rₘₐₓ[1]<=Pˢᴳ²_2[t+1]-Pˢᴳ²_2[t]) 

    @constraint(model, Pˢᴳ³_1[t+1]-Pˢᴳ³_1[t]<=Rₘₐₓ[2])        
    @constraint(model, -Rₘₐₓ[2]<=Pˢᴳ³_1[t+1]-Pˢᴳ³_1[t]) 
    @constraint(model, Pˢᴳ³_2[t+1]-Pˢᴳ³_2[t]<=Rₘₐₓ[2])        
    @constraint(model, -Rₘₐₓ[2]<=Pˢᴳ³_2[t+1]-Pˢᴳ³_2[t]) 

    @constraint(model, Pˢᴳ⁴_1[t+1]-Pˢᴳ⁴_1[t]<=Rₘₐₓ[3])        
    @constraint(model, -Rₘₐₓ[3]<=Pˢᴳ⁴_1[t+1]-Pˢᴳ⁴_1[t]) 
    @constraint(model, Pˢᴳ⁴_2[t+1]-Pˢᴳ⁴_2[t]<=Rₘₐₓ[3])        
    @constraint(model, -Rₘₐₓ[3]<=Pˢᴳ⁴_2[t+1]-Pˢᴳ⁴_2[t])

    @constraint(model, Pˢᴳ⁵_1[t+1]-Pˢᴳ⁵_1[t]<=Rₘₐₓ[4])        
    @constraint(model, -Rₘₐₓ[4]<=Pˢᴳ⁵_1[t+1]-Pˢᴳ⁵_1[t])    
    @constraint(model, Pˢᴳ⁵_2[t+1]-Pˢᴳ⁵_2[t]<=Rₘₐₓ[4])        
    @constraint(model, -Rₘₐₓ[4]<=Pˢᴳ⁵_2[t+1]-Pˢᴳ⁵_2[t])    

    @constraint(model, Pˢᴳ²⁷_1[t+1]-Pˢᴳ²⁷_1[t]<=Rₘₐₓ[5])        
    @constraint(model, -Rₘₐₓ[5]<=Pˢᴳ²⁷_1[t+1]-Pˢᴳ²⁷_1[t])  
    @constraint(model, Pˢᴳ²⁷_2[t+1]-Pˢᴳ²⁷_2[t]<=Rₘₐₓ[5])        
    @constraint(model, -Rₘₐₓ[5]<=Pˢᴳ²⁷_2[t+1]-Pˢᴳ²⁷_2[t])  

    @constraint(model, Pˢᴳ³⁰_1[t+1]-Pˢᴳ³⁰_1[t]<=Rₘₐₓ[6])        
    @constraint(model, -Rₘₐₓ[6]<=Pˢᴳ³⁰_1[t+1]-Pˢᴳ³⁰_1[t])       
    @constraint(model, Pˢᴳ³⁰_2[t+1]-Pˢᴳ³⁰_2[t]<=Rₘₐₓ[6])        
    @constraint(model, -Rₘₐₓ[6]<=Pˢᴳ³⁰_2[t+1]-Pˢᴳ³⁰_2[t])           
end
         



for k in 1:size(K_g,2)       # bounds for the SCC of SGs
    for t in 1:T   
        @constraint(model, I_scc[k,t]==K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+

        K_c[1,k]*α₁[1]+ K_c[2,k]*α₂₃[1]+ K_c[3,k]*α₂₆[1]+
        
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

        

        @constraint(model, I_scc[k,t]

        >=Iₗᵢₘ)
    end
end

#-------Define Objective Functions
No_load_cost=sum(Cⁿˡ[1].*(yˢᴳ²_1+yˢᴳ²_2))+sum(Cⁿˡ[2].*(yˢᴳ³_1+yˢᴳ³_2))+sum(Cⁿˡ[3].*(yˢᴳ⁴_1+yˢᴳ⁴_2))+sum(Cⁿˡ[4].*(yˢᴳ⁵_1+yˢᴳ⁵_2))+sum(Cⁿˡ[5].*(yˢᴳ²⁷_1+yˢᴳ²⁷_2))+sum(Cⁿˡ[6].*(yˢᴳ³⁰_1+yˢᴳ³⁰_2))    

Generation_cost=sum(Cᵍᵐ₁[1].*Pˢᴳ²_1+Cᵍᵐ₂[1].*Pˢᴳ²_2)+sum(Cᵍᵐ₁[2].*Pˢᴳ³_1+Cᵍᵐ₂[2].*Pˢᴳ³_2)+sum(Cᵍᵐ₁[3].*Pˢᴳ⁴_1+Cᵍᵐ₂[3].*Pˢᴳ⁴_2)+sum(Cᵍᵐ₁[4].*Pˢᴳ⁵_1+Cᵍᵐ₂[4].*Pˢᴳ⁵_2)+sum(Cᵍᵐ₁[5].*Pˢᴳ²⁷_1+Cᵍᵐ₂[5].*Pˢᴳ²⁷_2)+sum(Cᵍᵐ₁[6].*Pˢᴳ³⁰_1+Cᵍᵐ₂[6].*Pˢᴳ³⁰_2)      # no-load cost

Onoff_cost=sum(Cᵁ²_1)+sum(Cᴰ²_1)+sum(Cᵁ³_1)+sum(Cᴰ³_1)+sum(Cᵁ⁴_1)+sum(Cᴰ⁴_1)+sum(Cᵁ⁵_1)+sum(Cᴰ⁵_1)+sum(Cᵁ²⁷_1)+sum(Cᴰ²⁷_1)+sum(Cᵁ³⁰_1)+sum(Cᴰ³⁰_1)  +sum(Cᵁ²_2)+sum(Cᴰ²_2)+sum(Cᵁ³_2)+sum(Cᴰ³_2)+sum(Cᵁ⁴_2)+sum(Cᴰ⁴_2)+sum(Cᵁ⁵_2)+sum(Cᴰ⁵_2)+sum(Cᵁ²⁷_2)+sum(Cᴰ²⁷_2)+sum(Cᵁ³⁰_2)+sum(Cᴰ³⁰_2)       

@objective(model, Min, No_load_cost+ Generation_cost+ Onoff_cost)  # objective function
#-----------------------------------Solve and Output Results
set_optimizer(model , Gurobi.Optimizer)
# set_attribute(model, "limits/gap", 0.0280)
# set_time_limit_sec(model, 700.0)
optimize!(model)

#-----------------------------------Calculate SCC at each bus





 # yˢᴳ²=JuMP.value.(yˢᴳ²)
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
