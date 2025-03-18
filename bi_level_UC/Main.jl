# A Bilevel problem with UC. 
# Solve this by CCG 

import Pkg
using JuMP,Gurobi, CSV,DataFrames,LinearAlgebra, XLSX, IterTools, DelimitedFiles 

#include("MP.jl")   # master problem
#include("SP1.jl")  # first subproblem
#include("SP2.jl")  # second subproblem


#--------------------------- R&D procedure for solving the  Mi-Bilevel problem ------------

UB=Inf           # initial UB 
LB=-Inf          # initial LB 
tol=2;           # optimality tolerance
l=10

x₂₁=[]     # initialization, so the input for MP.jl is empty to have    OPTIMAL ones for UL problem
x₂₂=[]     # binary variables for the 12 SGs 
x₃₁=[]     
x₃₂=[]
x₄₁=[]
x₄₂=[]
x₅₁=[]
x₅₂=[]
x₂₇₁=[]
x₂₇₂=[]
x₃₀₁=[]
x₃₀₂=[]

kₙₗ=[]
kₘ=[]
kₛₜ=[]
kₛₕ=[] 


for j in 1:l

    LB,  λᵇ, λˢ            =  MP( x₂₁,x₂₂,x₃₁,x₃₂,x₄₁,x₄₂,x₅₁,x₅₂,x₂₇₁,x₂₇₂,x₃₀₁,x₃₀₂, LB  )  # update the input (binary variables) for MP
    ψ_for_SP2, Vᴾ          = SP1(  x₂₁,x₂₂,x₃₁,x₃₂,x₄₁,x₄₂,x₅₁,x₅₂,x₂₇₁,x₂₇₂,x₃₀₁,x₃₀₂  )    # fixed binary variables for SP1
    u¹,u²,u³ ,UB                  = SP2( ψ_for_SP2, λᵇ, λˢ, UB)   

    if UB-LB <=tol
        break  
    end
    
end
