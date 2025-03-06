# Author:     Peng Wang     from Technical University of Madrid (UPM)
# Supervisor: Luis Badesa   from Technical University of Madrid (UPM)
# This program is the dispatchable method for pricing SCC services in paper: "Pricing of Short Circuit Current in High IBR-Penetrated System".
# Note:   SGs  buses:2,3,4,5,27,30    IBRs buses:1,23,26
# 04.Mar.2025

import Pkg
#Pkg.add("Plots")
using JuMP,Gurobi, CSV,DataFrames,LinearAlgebra, XLSX, IterTools, DelimitedFiles, Plots

#----------------------------------Introduction of IEEE-30 Bus System Data and Functions----------------------------------
# SGs  buses:2,3,4,5,27,30
# IBRs buses:1,23,26

df = DataFrame(CSV.File("/Users/kl/Desktop/single_level_UC_SCL/Loadcurves.csv"))        
loadcurve=df[:,:]  
df = DataFrame(CSV.File( "/Users/kl/Desktop/single_level_UC_SCL/Windcurves.csv") ) 
windcurves=df[:,:]

include("dataset_gene.jl")
include("offline_trainning.jl")
include("admittance_matrix_calculation.jl")

#-----------------------------------Define Parameters for Calculating SCC----------------------------------
IBG₁=Vector(windcurves[1, 2:end])      # total capacity of IBRs, here they are wind turbines 
IBG₂₃=Vector(windcurves[2, 2:end])
IBG₂₆=Vector(windcurves[3, 2:end])
IBG₁=hcat(IBG₁)*10^8*3               # the coffecients here can be modified
IBG₂₃=hcat(IBG₂₃)*10^8*3
IBG₂₆=hcat(IBG₂₆)*10^8*3

#plot(Load_total)
#plot!(IBG₁+IBG₂₃+IBG₂₆)

I_IBG=1     # pre-defined SCC contribution from IBG
Iₗᵢₘ= 6      # SCC limit/SCL
β=0.95      # fluctuation coffecient of nominal voltage 
v=1         # nominal voltage 

I_SCC_all_buses_scenarios, matrix_ω = dataset_gene(I_IBG,β,v)   # data set generation


# signal=ones(size(I_SCC_all_buses_scenarios,1),1)
# for i in 1:size(I_SCC_all_buses_scenarios,1)
  #   for j in 1:size(I_SCC_all_buses_scenarios,2)
    #     if I_SCC_all_buses_scenarios[i,j]<Iₗᵢₘ
      #       signal[i]=signal[i]*0
        # end
    # end
# end

# j=0
# for i in 1:size(signal,1)
  #   if signal[i]==1
    #     j=j+1
    # end
# end


#signal=[]
# for i in 1:size(I_SCC_all_buses_scenarios,1)
  #   for j in 1:size(I_SCC_all_buses_scenarios,2)
    #     if I_SCC_all_buses_scenarios[i,j]<Iₗᵢₘ
      #       signal[i]=signal[i]*0
        # end
    # end
# end


#for i in 1:size(I_SCC_all_buses_scenarios,1)
    #if I_SCC_all_buses_scenarios[i,1]<Iₗᵢₘ
       # push!(signal, i)
   # end
#end
#signal = hcat(signal...)' 





K_g, K_c, K_m, N_type_1, N_type_2, err_type_1, err_type_2= offline_trainning(I_SCC_all_buses_scenarios, matrix_ω, Iₗᵢₘ, v)  # offline_trainning

#-----------------------------------Define Parameters for Optimization----------------------------------
Load_total=[18.42,17.95,18.29,18.51,18.13,17.88,19.46,21.97,23.17,23.87,
23.91,23.77,23.80,23.82,24.23,23.79,26.01,26.91,25.26,23.69,22.12,20.04,18.17,18.01]*10^9*0.7

Pˢᴳₘₐₓ=[6.584, 5.760, 3.781, 3.335, 3.252, 2.880]*10^9    # maximum output of SGs in corresponding bus 
Pˢᴳₘᵢₙ=[3.292, 2.880, 1.512, 0.667, 0.650, 0.288]*10^9    # minimum output of SGs in corresponding bus 
Rₘₐₓ=[1.317, 1.152, 1.512, 1.334, 1.951, 1.728]*10^9      # ramp rates

Kᵁ=[4000, 325, 142.5, 72, 55, 31]/5*10^3                  # startup cost
Kᴰ=[800, 28.5, 18.5, 14.4, 12, 10]/5*10^3                 # shutdown cost
Cᵍᵐ₁=[6.20, 32.10, 36.47, 64.28, 84.53, 97.36]/10^6       # marginal cost for SG1 in each SGs bus
Cᵍᵐ₂=[7.07, 34.72, 38.49, 72.84, 93.60, 105.02]/10^6      # marginal cost for SG2 in each SGs bus
# Cᵍᵐ₁=[6.20, 7.10, 9.47, 11.28, 18.53, 24.36]/10^6       # marginal cost for SG1 in each SGs bus
# Cᵍᵐ₂=[7.07, 9.72, 10.49, 13.84, 20.60, 25.02]/10^6      # marginal cost for SG2 in each SGs bus
Cⁿˡ=[18.431, 17.005, 13.755, 9.930, 9.900, 8.570]/5*10^3  # no-load cost for SGs in buses
T=length(Load_total)


shadow_prices_all_buses=zeros(T,size(K_g,2))             # matrix for saving shadow prices in each bus at all hours

for f in 1:size(K_g,2)
#---------------------------------------Define Model--------------------------------------------------------------------
model= Model()

#----------------------------------Define Variales
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

@variable(model, yˢᴳ²_1[1:T]>=0)   # status of SGs, here they are relaxed to be continuous ones [0,1]
@variable(model, yˢᴳ²_2[1:T]>=0)          
@variable(model, yˢᴳ³_1[1:T]>=0)  
@variable(model, yˢᴳ³_2[1:T]>=0)          
@variable(model, yˢᴳ⁴_1[1:T]>=0)      
@variable(model, yˢᴳ⁴_2[1:T]>=0)      
@variable(model, yˢᴳ⁵_1[1:T]>=0)      
@variable(model, yˢᴳ⁵_2[1:T]>=0)      
@variable(model, yˢᴳ²⁷_1[1:T]>=0)  
@variable(model, yˢᴳ²⁷_2[1:T]>=0)          
@variable(model, yˢᴳ³⁰_1[1:T]>=0)   
@variable(model, yˢᴳ³⁰_2[1:T]>=0)          


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

@variable(model, α₁>=0)                   # percentage of IBGs' online capacity  
@variable(model, α₂₃>=0)
@variable(model, α₂₆>=0)
#α₁=0.4
#α₂₃=0.3
#α₂₆=0.4


#----------------------------------Define Constraints
@constraint(model, yˢᴳ²_1[1:T].<=1)   # status of SGs, here they are relaxed to be continuous ones [0,1]
@constraint(model, yˢᴳ²_2[1:T].<=1)          
@constraint(model, yˢᴳ³_1[1:T].<=1)  
@constraint(model, yˢᴳ³_2[1:T].<=1)          
@constraint(model, yˢᴳ⁴_1[1:T].<=1)      
@constraint(model, yˢᴳ⁴_2[1:T].<=1)      
@constraint(model, yˢᴳ⁵_1[1:T].<=1)      
@constraint(model, yˢᴳ⁵_2[1:T].<=1)      
@constraint(model, yˢᴳ²⁷_1[1:T].<=1)  
@constraint(model, yˢᴳ²⁷_2[1:T].<=1)          
@constraint(model, yˢᴳ³⁰_1[1:T].<=1)   
@constraint(model, yˢᴳ³⁰_2[1:T].<=1) 

@constraint(model, Pˢᴳ²_1+Pˢᴳ²_2+Pˢᴳ³_1+Pˢᴳ³_2+Pˢᴳ⁴_1+Pˢᴳ⁴_2+Pˢᴳ⁵_1+Pˢᴳ⁵_2+Pˢᴳ²⁷_1+Pˢᴳ²⁷_2+Pˢᴳ³⁰_1+Pˢᴳ³⁰_2+
                    Pᴵᴮᴳ¹+Pᴵᴮᴳ²³+Pᴵᴮᴳ²⁶==Load_total)     # power balance              

@constraint(model, Pᴵᴮᴳ¹ .<= IBG₁*α₁)        # wind power limit
@constraint(model, Pᴵᴮᴳ¹>=0)
@constraint(model, Pᴵᴮᴳ²³.<= IBG₂₃*α₂₃)       
@constraint(model, Pᴵᴮᴳ²³>=0)
@constraint(model, Pᴵᴮᴳ²⁶.<= IBG₂₆*α₂₆)       
@constraint(model, Pᴵᴮᴳ²⁶>=0)

@constraint(model, α₁<=1)                   # IBG online capacity limit
@constraint(model, α₂₃<=1)                   
@constraint(model, α₂₆<=1)                   


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
         

constraints = Dict()    # dictionary for saving the names of SCL constraints
@variable(model, I_scc[1:T])

k=f                     # bus that needs to buy SCC services to matain at a certain SCL
    for t in 1:T  
        @constraint(model, I_scc[t]==
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+

        K_c[1,k]*α₁+ K_c[2,k]*α₂₃+ K_c[3,k]*α₂₆)

        constraints[t] = @constraint(model, I_scc[t]-Iₗᵢₘ >=0)
    end




#----------------------------------Define Objective Functions
No_load_cost=sum(Cⁿˡ[1].*(yˢᴳ²_1+yˢᴳ²_2))+sum(Cⁿˡ[2].*(yˢᴳ³_1+yˢᴳ³_2))+sum(Cⁿˡ[3].*(yˢᴳ⁴_1+yˢᴳ⁴_2))+sum(Cⁿˡ[4].*(yˢᴳ⁵_1+yˢᴳ⁵_2))+sum(Cⁿˡ[5].*(yˢᴳ²⁷_1+yˢᴳ²⁷_2))+sum(Cⁿˡ[6].*(yˢᴳ³⁰_1+yˢᴳ³⁰_2))    

Generation_cost=sum(Cᵍᵐ₁[1].*Pˢᴳ²_1+Cᵍᵐ₂[1].*Pˢᴳ²_2)+sum(Cᵍᵐ₁[2].*Pˢᴳ³_1+Cᵍᵐ₂[2].*Pˢᴳ³_2)+sum(Cᵍᵐ₁[3].*Pˢᴳ⁴_1+Cᵍᵐ₂[3].*Pˢᴳ⁴_2)+sum(Cᵍᵐ₁[4].*Pˢᴳ⁵_1+Cᵍᵐ₂[4].*Pˢᴳ⁵_2)+sum(Cᵍᵐ₁[5].*Pˢᴳ²⁷_1+Cᵍᵐ₂[5].*Pˢᴳ²⁷_2)+sum(Cᵍᵐ₁[6].*Pˢᴳ³⁰_1+Cᵍᵐ₂[6].*Pˢᴳ³⁰_2)      # no-load cost

Onoff_cost=sum(Cᵁ²_1)+sum(Cᴰ²_1)+sum(Cᵁ³_1)+sum(Cᴰ³_1)+sum(Cᵁ⁴_1)+sum(Cᴰ⁴_1)+sum(Cᵁ⁵_1)+sum(Cᴰ⁵_1)+sum(Cᵁ²⁷_1)+sum(Cᴰ²⁷_1)+sum(Cᵁ³⁰_1)+sum(Cᴰ³⁰_1)  +sum(Cᵁ²_2)+sum(Cᴰ²_2)+sum(Cᵁ³_2)+sum(Cᴰ³_2)+sum(Cᵁ⁴_2)+sum(Cᴰ⁴_2)+sum(Cᵁ⁵_2)+sum(Cᴰ⁵_2)+sum(Cᵁ²⁷_2)+sum(Cᴰ²⁷_2)+sum(Cᵁ³⁰_2)+sum(Cᴰ³⁰_2)       

@objective(model, Min, No_load_cost+ Generation_cost+ Onoff_cost)  # objective function



#-------------------------------------Solve and Output Results--------------------------------------------------------------------
set_optimizer(model , Gurobi.Optimizer)
# set_attribute(model, "limits/gap", 0.0280)
# set_time_limit_sec(model, 700.0)
optimize!(model)

# I_scc=JuMP.value.(I_scc)

#----------------------------------Shadow Prices for each bus at different hours
shadow_prices=zeros(1,T)
for i in 1:T
     # println("shadow price λ$i (constraint_$i): ", dual(constraints[i]))
     shadow_prices[i]=JuMP.value.(dual(constraints[i]))
end

shadow_prices_all_buses[:,f]=shadow_prices'

end


plot(shadow_prices_all_buses[:,6],label="bus 6",title="SCC prices at other buses")
plot!(shadow_prices_all_buses[:,7],label="bus 7")
plot!(shadow_prices_all_buses[:,8],label="bus 8")
plot!(shadow_prices_all_buses[:,28],label="bus 28")

plot(shadow_prices_all_buses[:,9],label="bus 9",title="SCC prices at other buses")
plot!(shadow_prices_all_buses[:,10],label="bus 10")
plot!(shadow_prices_all_buses[:,11],label="bus 11")
plot!(shadow_prices_all_buses[:,12],label="bus 12")
plot!(shadow_prices_all_buses[:,13],label="bus 13")
plot!(shadow_prices_all_buses[:,14],label="bus 14")
plot!(shadow_prices_all_buses[:,15],label="bus 15")
plot!(shadow_prices_all_buses[:,16],label="bus 16")
plot!(shadow_prices_all_buses[:,17],label="bus 17")
plot!(shadow_prices_all_buses[:,18],label="bus 18")
plot!(shadow_prices_all_buses[:,19],label="bus 19")
plot!(shadow_prices_all_buses[:,20],label="bus 20")
plot!(shadow_prices_all_buses[:,21],label="bus 21")
plot!(shadow_prices_all_buses[:,22],label="bus 22")
plot!(shadow_prices_all_buses[:,24],label="bus 24")
plot!(shadow_prices_all_buses[:,25],label="bus 25")
plot!(shadow_prices_all_buses[:,29],label="bus 29")

plot(shadow_prices_all_buses[:,1],label="bus 1",title="SCC prices at IBRs buses",linestyle=:solid, marker=:circle, markersize=5)
plot!(shadow_prices_all_buses[:,23],label="bus 23",linestyle=:solid, marker=:circle, markersize=5)
plot!(shadow_prices_all_buses[:,26],label="bus 26",linestyle=:solid, marker=:circle, markersize=5)

plot(shadow_prices_all_buses[:,2],label="bus 2",title="SCC prices at SGs buses",linestyle=:solid, marker=:circle, markersize=5)
plot!(shadow_prices_all_buses[:,3],label="bus 3",linestyle=:solid, marker=:circle, markersize=5)
plot!(shadow_prices_all_buses[:,4],label="bus 4",linestyle=:solid, marker=:circle, markersize=5)
plot!(shadow_prices_all_buses[:,5],label="bus 5",linestyle=:solid, marker=:circle, markersize=5)
plot!(shadow_prices_all_buses[:,27],label="bus 27",linestyle=:solid, marker=:circle, markersize=5)
plot!(shadow_prices_all_buses[:,30],label="bus 30",linestyle=:solid, marker=:circle, markersize=5)



#----------------------------------SCC payment from different sinks
#SGs_bus=[2 3 4 5 27 30]
#IBRs_bus=[23 26]
#payment=zeros(2,6)

#k=0
#for j in 1:2:size(K_g,1)-1
    #k=k+1
    #for i in 1: size(IBRs_bus,2)
        #payment[i,k]=sum(  (K_g[j,IBRs_bus[i]] +  K_g[j+1,IBRs_bus[i]] ) / sum(K_g[:,IBRs_bus[i]]) * Iₗᵢₘ.*shadow_prices_all_buses[:,IBRs_bus[i]])/T
    #end
#end
