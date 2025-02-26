# Author: Peng Wang       from Technical University of Madrid (UPM)
# Supervisor: Luis Badesa

# Now, this is single-level+UC+SCL TEST CASE with a modified IEEE-30 bus system
# 03.Dec.2024

import Pkg 
using JuMP,Gurobi, CSV,DataFrames,LinearAlgebra, XLSX, IterTools,MAT # (here we choose Gurobi as it can solve the product of binary and continuous variables, we don't need to linearlize the nonlinear terms manually)

#----------------------------------IEEE-30 Bus System Data Introduction----------------------------------
df = DataFrame(CSV.File("/Users/ME2/Desktop/scl_ieee30/Loadcurves.csv"))        
loadcurve=df[:,:]  
df = DataFrame(CSV.File( "/Users/ME2/Desktop/scl_ieee30/Windcurves.csv") ) 
windcurves=df[:,:]

include("dataset_gene.jl")
include("offline_trainning.jl")

#-----------------------------------Define Parameters for Calculating SCC
IBG₁=Vector(windcurves[1, 2:end])
IBG₂₃=Vector(windcurves[2, 2:end])
IBG₂₆=Vector(windcurves[3, 2:end])

I_IBG=1  # (pre-defined)
Iₗᵢₘ= 6     # SCC limit
β=0.95
v=1

I_scc_all, matrix_ω = dataset_gene(I_IBG, β)  # data set generation
# matwrite("C:/Users/ME2/Desktop/scl_ieee30/I_scc_all.mat", Dict("I_scc_all" => I_scc_all))


I_scc_all=I_scc_all[2:end, :]                  # delete the first row as the first row is not realistic
matrix_ω=matrix_ω[2:end, :]

K_g, K_c, K_m, N_type_1, N_type_2, err_type_1, err_type_2= offline_trainning(I_scc_all, matrix_ω, Iₗᵢₘ, v)  # offline_trainning


#-----------------------------------Define Parameters for Optimization
Pˢᴳₘₐₓ=[26 22 18 20 23 18]
Pˢᴳₘᵢₙ=[10 8 4 6 10 5]
Rₘₐₓ=Pˢᴳₘᵢₙ/5+3*ones(1,6)

Kᵁ=[3.25,2.72,1.43,2.03,  3.31,2.11]*10000
Kᴰ=[0.285,0.201,0.153,0.201,  0.312,0.189]*10000
Cᵍᵐ=[0.9,0.6,0.5,0.7,  0.8,0.6]*1000 
Cⁿˡ=[1.2,1,0.8,1.1,  1.1,0.9]*1000
T=24


     
#-----------------------------------Define Model
model= Model()

#-------Define Variales
@variable(model, Pˢᴳ²[1:T])                # generation of SG in bus 2
@variable(model, Pˢᴳ³[1:T])                # generation of SG in bus 3
@variable(model, Pˢᴳ⁴[1:T])                # generation of SG in bus 4
@variable(model, Pˢᴳ⁵[1:T])                # generation of SG in bus 5
@variable(model, Pˢᴳ²⁷[1:T])                # generation of SG in bus 27
@variable(model, Pˢᴳ³⁰[1:T])                # generation of SG in bus 30

@variable(model, yˢᴳ²[1:T],Bin)            # on/off status of SG in bus 2
@variable(model, yˢᴳ³[1:T],Bin)            # on/off status of SG in bus 3
@variable(model, yˢᴳ⁴[1:T],Bin)            # on/off status of SG in bus 4
@variable(model, yˢᴳ⁵[1:T],Bin)            # on/off status of SG in bus 5
@variable(model, yˢᴳ²⁷[1:T],Bin)            # on/off status of SG in bus 27
@variable(model, yˢᴳ³⁰[1:T],Bin)            # on/off status of SG in bus 30

@variable(model, Pᴵᴮᴳ¹[1:T])               # generation of IBG (WT) in bus 1
@variable(model, Pᴵᴮᴳ²³[1:T])              # generation of IBG (WT) in bus 23
@variable(model, Pᴵᴮᴳ²⁶[1:T])               # generation of IBG (WT) in bus 26

@variable(model, Cᵁ²[1:T])                 # startup costs for SG in bus 2
@variable(model, Cᴰ²[1:T])                 # shutdown costs for SG in bus 2
@variable(model, Cᵁ³[1:T])                 # startup costs for SG in bus 3
@variable(model, Cᴰ³[1:T])                 # shutdown costs for SG in bus 3
@variable(model, Cᵁ⁴[1:T])                 # startup costs for SG in bus 4
@variable(model, Cᴰ⁴[1:T])                 # shutdown costs for SG in bus 4
@variable(model, Cᵁ⁵[1:T])                 # startup costs for SG in bus 5
@variable(model, Cᴰ⁵[1:T])                 # shutdown costs for SG in bus 5
@variable(model, Cᵁ²⁷[1:T])                 # startup costs for SG in bus 27
@variable(model, Cᴰ²⁷[1:T])                 # shutdown costs for SG in bus 27
@variable(model, Cᵁ³⁰[1:T])                 # startup costs for SG in bus 30
@variable(model, Cᴰ³⁰[1:T])                 # shutdown costs for SG in bus 30

@variable(model, α₁[1:T]>=0)                   # percentage of IBG penetration  
@variable(model, α₂₃[1:T]>=0)
@variable(model, α₂₆[1:T]>=0)

#-------Define Constraints
Pᴰ=zeros(1,T)
for t in 1:T
    Pᴰ[t]=Pᴰ[t]+sum(loadcurve[:,t+2])
end

@constraint(model, Pˢᴳ²+Pˢᴳ³+Pˢᴳ⁴+Pˢᴳ⁵+Pˢᴳ²⁷+Pˢᴳ³⁰+Pᴵᴮᴳ¹+Pᴵᴮᴳ²³+Pᴵᴮᴳ²⁶== 0.25.*Pᴰ')     # power balance     0.4.*         

@constraint(model, Pᴵᴮᴳ¹ <= α₁.*IBG₁*5)        # wind power limit
@constraint(model, Pᴵᴮᴳ¹>=0)
@constraint(model, Pᴵᴮᴳ²³<= α₁.*IBG₂₃*5)       
@constraint(model, Pᴵᴮᴳ²³>=0)
@constraint(model, Pᴵᴮᴳ²⁶<= α₁.*IBG₂₆*5)       
@constraint(model, Pᴵᴮᴳ²⁶>=0)

@constraint(model, α₁ .<= 1)        # wind power limit
@constraint(model, α₂₃ .<= 1)       
@constraint(model, α₂₆ .<= 1)       


# @constraint(model, α₁.*IBG₁==Pᴵᴮᴳ¹)                   # IBG online capacity limit
# @constraint(model, α₂₃.*IBG₂₃==Pᴵᴮᴳ²³)                   
# @constraint(model, α₂₆.*IBG₂₆==Pᴵᴮᴳ²⁶)                   


@constraint(model, Pˢᴳ².<=yˢᴳ²*Pˢᴳₘₐₓ[1])       # bounds for the output of SGs
@constraint(model, yˢᴳ²*Pˢᴳₘᵢₙ[1].<=Pˢᴳ²)
@constraint(model, Pˢᴳ³.<=yˢᴳ³*Pˢᴳₘₐₓ[2])   
@constraint(model, yˢᴳ³*Pˢᴳₘᵢₙ[2].<=Pˢᴳ³)        
@constraint(model, Pˢᴳ⁴.<=yˢᴳ⁴*Pˢᴳₘₐₓ[3])   
@constraint(model, yˢᴳ⁴*Pˢᴳₘᵢₙ[3].<=Pˢᴳ⁴)  
@constraint(model, Pˢᴳ⁵.<=yˢᴳ⁵*Pˢᴳₘₐₓ[4])   
@constraint(model, yˢᴳ⁵*Pˢᴳₘᵢₙ[4].<=Pˢᴳ⁵)  
@constraint(model, Pˢᴳ²⁷.<=yˢᴳ²⁷*Pˢᴳₘₐₓ[5])   
@constraint(model, yˢᴳ²⁷*Pˢᴳₘᵢₙ[5].<=Pˢᴳ²⁷)  
@constraint(model, Pˢᴳ³⁰.<=yˢᴳ³⁰*Pˢᴳₘₐₓ[6])   
@constraint(model, yˢᴳ³⁰*Pˢᴳₘᵢₙ[6].<=Pˢᴳ³⁰)  

@constraint(model, Cᵁ²>=0)     # lower bounds for the startup and shutdown costs of SG                     
@constraint(model, Cᴰ²>=0) 
@constraint(model, Cᵁ³>=0)                   
@constraint(model, Cᴰ³>=0)                            
@constraint(model, Cᵁ⁴>=0)                   
@constraint(model, Cᴰ⁴>=0) 
@constraint(model, Cᵁ⁵>=0)                   
@constraint(model, Cᴰ⁵>=0) 
@constraint(model, Cᵁ²⁷>=0)                   
@constraint(model, Cᴰ²⁷>=0) 
@constraint(model, Cᵁ³⁰>=0)                   
@constraint(model, Cᴰ³⁰>=0) 

for t in 1:T-1
@constraint(model, Cᵁ²[t]>=(yˢᴳ²[t+1]-yˢᴳ²[t])*Kᵁ[1])        
@constraint(model, Cᴰ²[t]>=(yˢᴳ²[t]-yˢᴳ²[t+1])*Kᴰ[1])  
@constraint(model, Cᵁ³[t]>=(yˢᴳ³[t+1]-yˢᴳ³[t])*Kᵁ[2])        
@constraint(model, Cᴰ³[t]>=(yˢᴳ³[t]-yˢᴳ³[t+1])*Kᴰ[2])  
@constraint(model, Cᵁ⁴[t]>=(yˢᴳ⁴[t+1]-yˢᴳ⁴[t])*Kᵁ[3])        
@constraint(model, Cᴰ⁴[t]>=(yˢᴳ⁴[t]-yˢᴳ⁴[t+1])*Kᴰ[3]) 
@constraint(model, Cᵁ⁵[t]>=(yˢᴳ⁵[t+1]-yˢᴳ⁵[t])*Kᵁ[4])        
@constraint(model, Cᴰ⁵[t]>=(yˢᴳ⁵[t]-yˢᴳ⁵[t+1])*Kᴰ[4])             
@constraint(model, Cᵁ²⁷[t]>=(yˢᴳ²⁷[t+1]-yˢᴳ²⁷[t])*Kᵁ[5])        
@constraint(model, Cᴰ²⁷[t]>=(yˢᴳ²⁷[t]-yˢᴳ²⁷[t+1])*Kᴰ[5])
@constraint(model, Cᵁ³⁰[t]>=(yˢᴳ³⁰[t+1]-yˢᴳ³⁰[t])*Kᵁ[6])        
@constraint(model, Cᴰ³⁰[t]>=(yˢᴳ³⁰[t]-yˢᴳ³⁰[t+1])*Kᴰ[6])
end

for t in 1:T-1       # bounds for the ramp of SGs
    @constraint(model, Pˢᴳ²[t+1]-Pˢᴳ²[t]<=Rₘₐₓ[1])        
    @constraint(model, -Rₘₐₓ[1]<=Pˢᴳ²[t+1]-Pˢᴳ²[t])  
    @constraint(model, Pˢᴳ³[t+1]-Pˢᴳ³[t]<=Rₘₐₓ[2])        
    @constraint(model, -Rₘₐₓ[2]<=Pˢᴳ³[t+1]-Pˢᴳ³[t]) 
    @constraint(model, Pˢᴳ⁴[t+1]-Pˢᴳ⁴[t]<=Rₘₐₓ[3])        
    @constraint(model, -Rₘₐₓ[3]<=Pˢᴳ⁴[t+1]-Pˢᴳ⁴[t]) 
    @constraint(model, Pˢᴳ⁵[t+1]-Pˢᴳ⁵[t]<=Rₘₐₓ[4])        
    @constraint(model, -Rₘₐₓ[4]<=Pˢᴳ⁵[t+1]-Pˢᴳ⁵[t])    
    @constraint(model, Pˢᴳ²⁷[t+1]-Pˢᴳ²⁷[t]<=Rₘₐₓ[5])        
    @constraint(model, -Rₘₐₓ[5]<=Pˢᴳ²⁷[t+1]-Pˢᴳ²⁷[t])  
    @constraint(model, Pˢᴳ³⁰[t+1]-Pˢᴳ³⁰[t]<=Rₘₐₓ[6])        
    @constraint(model, -Rₘₐₓ[6]<=Pˢᴳ³⁰[t+1]-Pˢᴳ³⁰[t])             
end
         
for k in 1:size(K_g,2)
    for t in 1:T   # bounds for the SCC of SGs
        @constraint(model, K_g[1,k]*yˢᴳ²[t]+ K_g[2,k]*yˢᴳ³[t]+ K_g[3,k]*yˢᴳ⁴[t]+ K_g[4,k]*yˢᴳ⁵[t]+K_g[5,k]*yˢᴳ²⁷[t]+ K_g[6,k]*yˢᴳ³⁰[t]+ 
        K_c[1,k]*α₁[t]+ K_c[2,k]*α₁[t]+ K_c[3,k]*α₁[t]+
        K_m[1,k]*yˢᴳ²[t]*yˢᴳ³[t]+ K_m[2,k]*yˢᴳ²[t]*yˢᴳ⁴[t]+ K_m[3,k]*yˢᴳ²[t]*yˢᴳ⁵[t]+ K_m[4,k]*yˢᴳ²[t]*yˢᴳ²⁷[t]+ K_m[5,k]*yˢᴳ²[t]*yˢᴳ³⁰[t]+      
        K_m[6,k]*yˢᴳ³[t]*yˢᴳ⁴[t]+ K_m[7,k]*yˢᴳ³[t]*yˢᴳ⁵[t]+ K_m[8,k]*yˢᴳ³[t]*yˢᴳ²⁷[t]+ K_m[9,k]*yˢᴳ³[t]*yˢᴳ³⁰[t]+
        K_m[10,k]*yˢᴳ⁴[t]*yˢᴳ⁵[t]+ K_m[11,k]*yˢᴳ⁴[t]*yˢᴳ²⁷[t]+ K_m[12,k]*yˢᴳ⁴[t]*yˢᴳ³⁰[t]+
        K_m[13,k]*yˢᴳ⁵[t]*yˢᴳ²⁷[t]+ K_m[14,k]*yˢᴳ⁵[t]*yˢᴳ³⁰[t]+
        K_m[15,k]*yˢᴳ²⁷[t]*yˢᴳ³⁰[t] 
        >=Iₗᵢₘ)
    end
end

#-------Define Objective Functions
No_load_cost=sum(Cⁿˡ[1].*yˢᴳ²)+sum(Cⁿˡ[2].*yˢᴳ³)+sum(Cⁿˡ[3].*yˢᴳ⁴)+sum(Cⁿˡ[4].*yˢᴳ⁵)+sum(Cⁿˡ[5].*yˢᴳ²⁷)+sum(Cⁿˡ[6].*yˢᴳ³⁰)      # no-load cost
Generation_cost=sum(Cᵍᵐ[1].*Pˢᴳ²)+sum(Cᵍᵐ[2].*Pˢᴳ³)+sum(Cᵍᵐ[3].*Pˢᴳ⁴)+sum(Cᵍᵐ[4].*Pˢᴳ⁵)+sum(Cᵍᵐ[5].*Pˢᴳ²⁷)+sum(Cᵍᵐ[6].*Pˢᴳ³⁰)    # generation cost
Onoff_cost=sum(Cᵁ²)+sum(Cᴰ²)+sum(Cᵁ³)+sum(Cᴰ³)+sum(Cᵁ⁴)+sum(Cᴰ⁴)+sum(Cᵁ⁵)+sum(Cᴰ⁵)+sum(Cᵁ²⁷)+sum(Cᴰ²⁷)+sum(Cᵁ³⁰)+sum(Cᴰ³⁰)     # on/off cost
@objective(model, Min, No_load_cost+ Generation_cost+ Onoff_cost)  # objective function
#-----------------------------------Solve and Output Results
set_optimizer(model , Gurobi.Optimizer)
# set_attribute(model, "limits/gap", 0.0280)
# set_time_limit_sec(model, 700.0)
optimize!(model)

#-----------------------------------Calculate SCC at each bus





 yˢᴳ²=JuMP.value.(yˢᴳ²)
 yˢᴳ³=JuMP.value.(yˢᴳ³)
 yˢᴳ⁴=JuMP.value.(yˢᴳ⁴)
 yˢᴳ⁵=JuMP.value.(yˢᴳ⁵)
 yˢᴳ²⁷=JuMP.value.(yˢᴳ²⁷)
 yˢᴳ³⁰=JuMP.value.(yˢᴳ³⁰)

 α₁=JuMP.value.(α₁)
 α₂₃=JuMP.value.(α₂₃)
 α₂₆=JuMP.value.(α₂₆)
  α₂₆=JuMP.value.(α₂₆)
#Z=JuMP.value.(Z)
#println(value.( Z[1,3,1]))
