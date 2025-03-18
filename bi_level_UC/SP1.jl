function SP1(   yˢᴳ²_1,yˢᴳ²_2,yˢᴳ³_1,yˢᴳ³_2,yˢᴳ⁴_1,yˢᴳ⁴_2,yˢᴳ⁵_1,yˢᴳ⁵_2,yˢᴳ²⁷_1,yˢᴳ²⁷_2,yˢᴳ³⁰_1,yˢᴳ³⁰_2  )
# solve SP1 for online status of 12 SGs given in advance
    
#----------------------------------IEEE-30 Bus System Data Introduction----------------------------------
    df = DataFrame(CSV.File( "C:/Users/ME2/Desktop/single_level_UC_SCL_pricing/Windcurves.csv") ) 
    windcurves=df[:,:]

    IBG₁=Vector(windcurves[1, 2:end])      # total capacity of IBRs, here they are wind turbines 
    IBG₂₃=Vector(windcurves[2, 2:end])
    IBG₂₆=Vector(windcurves[3, 2:end])
    IBG₁=hcat(IBG₁)*10^8*3                 # the coffecients here can be modified
    IBG₂₃=hcat(IBG₂₃)*10^8*3
    IBG₂₆=hcat(IBG₂₆)*10^8*3

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
    
#---------------------------------------Define Model--------------------------------------------------------------------
    model_SP1= Model()
    


#-------Define Variales
    @variable(model_SP1, Pˢᴳ²_1[1:T])     # generation of SG 
    @variable(model_SP1, Pˢᴳ²_2[1:T])   
    @variable(model_SP1, Pˢᴳ³_1[1:T])     
    @variable(model_SP1, Pˢᴳ³_2[1:T])                
    @variable(model_SP1, Pˢᴳ⁴_1[1:T])   
    @variable(model_SP1, Pˢᴳ⁴_2[1:T])             
    @variable(model_SP1, Pˢᴳ⁵_1[1:T])    
    @variable(model_SP1, Pˢᴳ⁵_2[1:T])            
    @variable(model_SP1, Pˢᴳ²⁷_1[1:T]) 
    @variable(model_SP1, Pˢᴳ²⁷_2[1:T])                
    @variable(model_SP1, Pˢᴳ³⁰_1[1:T])                
    @variable(model_SP1, Pˢᴳ³⁰_2[1:T])  

    @variable(model_SP1, Pᴵᴮᴳ¹[1:T]>=0)               # generation of IBG (WT) 
    @variable(model_SP1, Pᴵᴮᴳ²³[1:T]>=0)              
    @variable(model_SP1, Pᴵᴮᴳ²⁶[1:T]>=0)               

    @variable(model_SP1, Cᵁ²_1[1:T]>=0)                 # startup costs for SG
    @variable(model_SP1, Cᵁ²_2[1:T]>=0)                 
    @variable(model_SP1, Cᴰ²_1[1:T]>=0)                 # shutdown costs for SG 
    @variable(model_SP1, Cᴰ²_2[1:T]>=0)                 
    @variable(model_SP1, Cᵁ³_1[1:T]>=0)    
    @variable(model_SP1, Cᵁ³_2[1:T]>=0)             
    @variable(model_SP1, Cᴰ³_1[1:T]>=0)  
    @variable(model_SP1, Cᴰ³_2[1:T]>=0)               
    @variable(model_SP1, Cᵁ⁴_1[1:T]>=0)     
    @variable(model_SP1, Cᵁ⁴_2[1:T]>=0)            
    @variable(model_SP1, Cᴰ⁴_1[1:T]>=0)   
    @variable(model_SP1, Cᴰ⁴_2[1:T]>=0)              
    @variable(model_SP1, Cᵁ⁵_1[1:T]>=0)  
    @variable(model_SP1, Cᵁ⁵_2[1:T]>=0)               
    @variable(model_SP1, Cᴰ⁵_1[1:T]>=0) 
    @variable(model_SP1, Cᴰ⁵_2[1:T]>=0)                
    @variable(model_SP1, Cᵁ²⁷_1[1:T]>=0)    
    @variable(model_SP1, Cᵁ²⁷_2[1:T]>=0)             
    @variable(model_SP1, Cᴰ²⁷_1[1:T]>=0)    
    @variable(model_SP1, Cᴰ²⁷_2[1:T]>=0)             
    @variable(model_SP1, Cᵁ³⁰_1[1:T]>=0)      
    @variable(model_SP1, Cᵁ³⁰_2[1:T]>=0)            
    @variable(model_SP1, Cᴰ³⁰_1[1:T]>=0)    
    @variable(model_SP1, Cᴰ³⁰_2[1:T]>=0)            

    @variable(model_SP1, α₁[1:T]>=0)                   # percentage of IBGs' online capacity  
    @variable(model_SP1, α₂₃[1:T]>=0)
    @variable(model_SP1, α₂₆[1:T]>=0)



#-------Define Constraints
@constraint(model_SP1, Pˢᴳ²_1+Pˢᴳ²_2+Pˢᴳ³_1+Pˢᴳ³_2+Pˢᴳ⁴_1+Pˢᴳ⁴_2+Pˢᴳ⁵_1+Pˢᴳ⁵_2+Pˢᴳ²⁷_1+Pˢᴳ²⁷_2+Pˢᴳ³⁰_1+Pˢᴳ³⁰_2+Pᴵᴮᴳ¹+Pᴵᴮᴳ²³+Pᴵᴮᴳ²⁶==Load_total)  # power balance              

for t in 1:T
@constraint(model_SP1, Pᴵᴮᴳ¹[t]<= IBG₁[t]*α₁[t])        # wind power limit
@constraint(model_SP1, Pᴵᴮᴳ²³[t]<= IBG₂₃[t]*α₂₃[t])       
@constraint(model_SP1, Pᴵᴮᴳ²⁶[t]<= IBG₂₆[t]*α₂₆[t])       
@constraint(model_SP1, α₁[t]<=1)                        # IBG online capacity limit
@constraint(model_SP1, α₂₃[t]<=1)                   
@constraint(model_SP1, α₂₆[t]<=1)                
end                



@constraint(model_SP1, Pˢᴳ²_1.<=yˢᴳ²_1*Pˢᴳₘₐₓ[1])          # bounds for the output of SGs
@constraint(model_SP1, yˢᴳ²_1*Pˢᴳₘᵢₙ[1].<=Pˢᴳ²_1)
@constraint(model_SP1, Pˢᴳ²_1.<=yˢᴳ²_1*Pˢᴳₘₐₓ[1])       
@constraint(model_SP1, yˢᴳ²_1*Pˢᴳₘᵢₙ[1].<=Pˢᴳ²_1)
@constraint(model_SP1, Pˢᴳ²_2.<=yˢᴳ²_2*Pˢᴳₘₐₓ[1])       
@constraint(model_SP1, yˢᴳ²_2*Pˢᴳₘᵢₙ[1].<=Pˢᴳ²_2)
@constraint(model_SP1, Pˢᴳ²_2.<=yˢᴳ²_2*Pˢᴳₘₐₓ[1])       
@constraint(model_SP1, yˢᴳ²_2*Pˢᴳₘᵢₙ[1].<=Pˢᴳ²_2)
@constraint(model_SP1, Pˢᴳ³_1.<=yˢᴳ³_1*Pˢᴳₘₐₓ[2])      
@constraint(model_SP1, yˢᴳ³_1*Pˢᴳₘᵢₙ[2].<=Pˢᴳ³_1)
@constraint(model_SP1, Pˢᴳ³_1.<=yˢᴳ³_1*Pˢᴳₘₐₓ[2])       
@constraint(model_SP1, yˢᴳ³_1*Pˢᴳₘᵢₙ[2].<=Pˢᴳ³_1)
@constraint(model_SP1, Pˢᴳ³_2.<=yˢᴳ³_2*Pˢᴳₘₐₓ[2])       
@constraint(model_SP1, yˢᴳ³_2*Pˢᴳₘᵢₙ[2].<=Pˢᴳ³_2)
@constraint(model_SP1, Pˢᴳ³_2.<=yˢᴳ³_2*Pˢᴳₘₐₓ[2])       
@constraint(model_SP1, yˢᴳ³_2*Pˢᴳₘᵢₙ[2].<=Pˢᴳ³_2)
@constraint(model_SP1, Pˢᴳ⁴_1.<=yˢᴳ⁴_1*Pˢᴳₘₐₓ[3])      
@constraint(model_SP1, yˢᴳ⁴_1*Pˢᴳₘᵢₙ[3].<=Pˢᴳ⁴_1)
@constraint(model_SP1, Pˢᴳ⁴_1.<=yˢᴳ⁴_1*Pˢᴳₘₐₓ[3])       
@constraint(model_SP1, yˢᴳ⁴_1*Pˢᴳₘᵢₙ[3].<=Pˢᴳ⁴_1)
@constraint(model_SP1, Pˢᴳ⁴_2.<=yˢᴳ⁴_2*Pˢᴳₘₐₓ[3])       
@constraint(model_SP1, yˢᴳ⁴_2*Pˢᴳₘᵢₙ[3].<=Pˢᴳ⁴_2)
@constraint(model_SP1, Pˢᴳ⁴_2.<=yˢᴳ⁴_2*Pˢᴳₘₐₓ[3])       
@constraint(model_SP1, yˢᴳ⁴_2*Pˢᴳₘᵢₙ[3].<=Pˢᴳ⁴_2)
@constraint(model_SP1, Pˢᴳ⁵_1.<=yˢᴳ⁵_1*Pˢᴳₘₐₓ[4])      
@constraint(model_SP1, yˢᴳ⁵_1*Pˢᴳₘᵢₙ[4].<=Pˢᴳ⁵_1)
@constraint(model_SP1, Pˢᴳ⁵_1.<=yˢᴳ⁵_1*Pˢᴳₘₐₓ[4])       
@constraint(model_SP1, yˢᴳ⁵_1*Pˢᴳₘᵢₙ[4].<=Pˢᴳ⁵_1)
@constraint(model_SP1, Pˢᴳ⁵_2.<=yˢᴳ⁵_2*Pˢᴳₘₐₓ[4])       
@constraint(model_SP1, yˢᴳ⁵_2*Pˢᴳₘᵢₙ[4].<=Pˢᴳ⁵_2)
@constraint(model_SP1, Pˢᴳ⁵_2.<=yˢᴳ⁵_2*Pˢᴳₘₐₓ[4])       
@constraint(model_SP1, yˢᴳ⁵_2*Pˢᴳₘᵢₙ[4].<=Pˢᴳ⁵_2)
@constraint(model_SP1, Pˢᴳ²⁷_1.<=yˢᴳ²⁷_1*Pˢᴳₘₐₓ[5])      
@constraint(model_SP1, yˢᴳ²⁷_1*Pˢᴳₘᵢₙ[5].<=Pˢᴳ²⁷_1)
@constraint(model_SP1, Pˢᴳ²⁷_1.<=yˢᴳ²⁷_1*Pˢᴳₘₐₓ[5])       
@constraint(model_SP1, yˢᴳ²⁷_1*Pˢᴳₘᵢₙ[5].<=Pˢᴳ²⁷_1)
@constraint(model_SP1, Pˢᴳ²⁷_2.<=yˢᴳ²⁷_2*Pˢᴳₘₐₓ[5])       
@constraint(model_SP1, yˢᴳ²⁷_2*Pˢᴳₘᵢₙ[5].<=Pˢᴳ²⁷_2)
@constraint(model_SP1, Pˢᴳ²⁷_2.<=yˢᴳ²⁷_2*Pˢᴳₘₐₓ[5])       
@constraint(model_SP1, yˢᴳ²⁷_2*Pˢᴳₘᵢₙ[5].<=Pˢᴳ²⁷_2)
@constraint(model_SP1, Pˢᴳ³⁰_1.<=yˢᴳ³⁰_1*Pˢᴳₘₐₓ[6])      
@constraint(model_SP1, yˢᴳ³⁰_1*Pˢᴳₘᵢₙ[6].<=Pˢᴳ³⁰_1)
@constraint(model_SP1, Pˢᴳ³⁰_1.<=yˢᴳ³⁰_1*Pˢᴳₘₐₓ[6])       
@constraint(model_SP1, yˢᴳ³⁰_1*Pˢᴳₘᵢₙ[6].<=Pˢᴳ³⁰_1)
@constraint(model_SP1, Pˢᴳ³⁰_2.<=yˢᴳ³⁰_2*Pˢᴳₘₐₓ[6])       
@constraint(model_SP1, yˢᴳ³⁰_2*Pˢᴳₘᵢₙ[6].<=Pˢᴳ³⁰_2)
@constraint(model_SP1, Pˢᴳ³⁰_2.<=yˢᴳ³⁰_2*Pˢᴳₘₐₓ[6])       
@constraint(model_SP1, yˢᴳ³⁰_2*Pˢᴳₘᵢₙ[6].<=Pˢᴳ³⁰_2)
 

for t in 1:T-1
@constraint(model_SP1, Cᵁ²_1[t]>=(yˢᴳ²_1[t+1]-yˢᴳ²_1[t])*Kᵁ[1])        
@constraint(model_SP1, Cᴰ²_1[t]>=(yˢᴳ²_1[t]-yˢᴳ²_1[t+1])*Kᴰ[1])  
@constraint(model_SP1, Cᵁ²_2[t]>=(yˢᴳ²_2[t+1]-yˢᴳ²_2[t])*Kᵁ[1])        
@constraint(model_SP1, Cᴰ²_2[t]>=(yˢᴳ²_2[t]-yˢᴳ²_2[t+1])*Kᴰ[1])  
@constraint(model_SP1, Cᵁ³_1[t]>=(yˢᴳ³_1[t+1]-yˢᴳ³_1[t])*Kᵁ[2])        
@constraint(model_SP1, Cᴰ³_1[t]>=(yˢᴳ³_1[t]-yˢᴳ³_1[t+1])*Kᴰ[2])  
@constraint(model_SP1, Cᵁ³_2[t]>=(yˢᴳ³_2[t+1]-yˢᴳ³_2[t])*Kᵁ[2])        
@constraint(model_SP1, Cᴰ³_2[t]>=(yˢᴳ³_2[t]-yˢᴳ³_2[t+1])*Kᴰ[2]) 
@constraint(model_SP1, Cᵁ⁴_1[t]>=(yˢᴳ⁴_1[t+1]-yˢᴳ⁴_1[t])*Kᵁ[3])        
@constraint(model_SP1, Cᴰ⁴_1[t]>=(yˢᴳ⁴_1[t]-yˢᴳ⁴_1[t+1])*Kᴰ[3]) 
@constraint(model_SP1, Cᵁ⁴_2[t]>=(yˢᴳ⁴_2[t+1]-yˢᴳ⁴_2[t])*Kᵁ[3])        
@constraint(model_SP1, Cᴰ⁴_2[t]>=(yˢᴳ⁴_2[t]-yˢᴳ⁴_2[t+1])*Kᴰ[3]) 
@constraint(model_SP1, Cᵁ⁵_1[t]>=(yˢᴳ⁵_1[t+1]-yˢᴳ⁵_1[t])*Kᵁ[4])        
@constraint(model_SP1, Cᴰ⁵_1[t]>=(yˢᴳ⁵_1[t]-yˢᴳ⁵_1[t+1])*Kᴰ[4])   
@constraint(model_SP1, Cᵁ⁵_2[t]>=(yˢᴳ⁵_2[t+1]-yˢᴳ⁵_2[t])*Kᵁ[4])        
@constraint(model_SP1, Cᴰ⁵_2[t]>=(yˢᴳ⁵_2[t]-yˢᴳ⁵_2[t+1])*Kᴰ[4]) 
@constraint(model_SP1, Cᵁ²⁷_1[t]>=(yˢᴳ²⁷_1[t+1]-yˢᴳ²⁷_1[t])*Kᵁ[5])        
@constraint(model_SP1, Cᴰ²⁷_1[t]>=(yˢᴳ²⁷_1[t]-yˢᴳ²⁷_1[t+1])*Kᴰ[5])
@constraint(model_SP1, Cᵁ²⁷_2[t]>=(yˢᴳ²⁷_2[t+1]-yˢᴳ²⁷_2[t])*Kᵁ[5])        
@constraint(model_SP1, Cᴰ²⁷_2[t]>=(yˢᴳ²⁷_2[t]-yˢᴳ²⁷_2[t+1])*Kᴰ[5])
@constraint(model_SP1, Cᵁ³⁰_1[t]>=(yˢᴳ³⁰_1[t+1]-yˢᴳ³⁰_1[t])*Kᵁ[6])        
@constraint(model_SP1, Cᴰ³⁰_1[t]>=(yˢᴳ³⁰_1[t]-yˢᴳ³⁰_1[t+1])*Kᴰ[6])
@constraint(model_SP1, Cᵁ³⁰_2[t]>=(yˢᴳ³⁰_2[t+1]-yˢᴳ³⁰_2[t])*Kᵁ[6])        
@constraint(model_SP1, Cᴰ³⁰_2[t]>=(yˢᴳ³⁰_2[t]-yˢᴳ³⁰_2[t+1])*Kᴰ[6])
end


for t in 1:T-1          # bounds for the ramp of SGs
    @constraint(model_SP1, Pˢᴳ²_1[t+1]-Pˢᴳ²_1[t]<=Rₘₐₓ[1])        
    @constraint(model_SP1, -Rₘₐₓ[1]<=Pˢᴳ²_1[t+1]-Pˢᴳ²_1[t])  
    @constraint(model_SP1, Pˢᴳ²_2[t+1]-Pˢᴳ²_2[t]<=Rₘₐₓ[1])        
    @constraint(model_SP1, -Rₘₐₓ[1]<=Pˢᴳ²_2[t+1]-Pˢᴳ²_2[t]) 

    @constraint(model_SP1, Pˢᴳ³_1[t+1]-Pˢᴳ³_1[t]<=Rₘₐₓ[2])        
    @constraint(model_SP1, -Rₘₐₓ[2]<=Pˢᴳ³_1[t+1]-Pˢᴳ³_1[t]) 
    @constraint(model_SP1, Pˢᴳ³_2[t+1]-Pˢᴳ³_2[t]<=Rₘₐₓ[2])        
    @constraint(model_SP1, -Rₘₐₓ[2]<=Pˢᴳ³_2[t+1]-Pˢᴳ³_2[t]) 

    @constraint(model_SP1, Pˢᴳ⁴_1[t+1]-Pˢᴳ⁴_1[t]<=Rₘₐₓ[3])        
    @constraint(model_SP1, -Rₘₐₓ[3]<=Pˢᴳ⁴_1[t+1]-Pˢᴳ⁴_1[t]) 
    @constraint(model_SP1, Pˢᴳ⁴_2[t+1]-Pˢᴳ⁴_2[t]<=Rₘₐₓ[3])        
    @constraint(model_SP1, -Rₘₐₓ[3]<=Pˢᴳ⁴_2[t+1]-Pˢᴳ⁴_2[t])

    @constraint(model_SP1, Pˢᴳ⁵_1[t+1]-Pˢᴳ⁵_1[t]<=Rₘₐₓ[4])        
    @constraint(model_SP1, -Rₘₐₓ[4]<=Pˢᴳ⁵_1[t+1]-Pˢᴳ⁵_1[t])    
    @constraint(model_SP1, Pˢᴳ⁵_2[t+1]-Pˢᴳ⁵_2[t]<=Rₘₐₓ[4])        
    @constraint(model_SP1, -Rₘₐₓ[4]<=Pˢᴳ⁵_2[t+1]-Pˢᴳ⁵_2[t])    

    @constraint(model_SP1, Pˢᴳ²⁷_1[t+1]-Pˢᴳ²⁷_1[t]<=Rₘₐₓ[5])        
    @constraint(model_SP1, -Rₘₐₓ[5]<=Pˢᴳ²⁷_1[t+1]-Pˢᴳ²⁷_1[t])  
    @constraint(model_SP1, Pˢᴳ²⁷_2[t+1]-Pˢᴳ²⁷_2[t]<=Rₘₐₓ[5])        
    @constraint(model_SP1, -Rₘₐₓ[5]<=Pˢᴳ²⁷_2[t+1]-Pˢᴳ²⁷_2[t])  

    @constraint(model_SP1, Pˢᴳ³⁰_1[t+1]-Pˢᴳ³⁰_1[t]<=Rₘₐₓ[6])        
    @constraint(model_SP1, -Rₘₐₓ[6]<=Pˢᴳ³⁰_1[t+1]-Pˢᴳ³⁰_1[t])       
    @constraint(model_SP1, Pˢᴳ³⁰_2[t+1]-Pˢᴳ³⁰_2[t]<=Rₘₐₓ[6])        
    @constraint(model_SP1, -Rₘₐₓ[6]<=Pˢᴳ³⁰_2[t+1]-Pˢᴳ³⁰_2[t])           
end



#-------Define Objective Functions
No_load_cost=sum(Cⁿˡ[1].*(yˢᴳ²_1+yˢᴳ²_2))+sum(Cⁿˡ[2].*(yˢᴳ³_1+yˢᴳ³_2))+sum(Cⁿˡ[3].*(yˢᴳ⁴_1+yˢᴳ⁴_2))+sum(Cⁿˡ[4].*(yˢᴳ⁵_1+yˢᴳ⁵_2))+sum(Cⁿˡ[5].*(yˢᴳ²⁷_1+yˢᴳ²⁷_2))+sum(Cⁿˡ[6].*(yˢᴳ³⁰_1+yˢᴳ³⁰_2))    

Generation_cost=sum(Cᵍᵐ₁[1].*Pˢᴳ²_1+Cᵍᵐ₂[1].*Pˢᴳ²_2)+sum(Cᵍᵐ₁[2].*Pˢᴳ³_1+Cᵍᵐ₂[2].*Pˢᴳ³_2)+sum(Cᵍᵐ₁[3].*Pˢᴳ⁴_1+Cᵍᵐ₂[3].*Pˢᴳ⁴_2)+sum(Cᵍᵐ₁[4].*Pˢᴳ⁵_1+Cᵍᵐ₂[4].*Pˢᴳ⁵_2)+sum(Cᵍᵐ₁[5].*Pˢᴳ²⁷_1+Cᵍᵐ₂[5].*Pˢᴳ²⁷_2)+sum(Cᵍᵐ₁[6].*Pˢᴳ³⁰_1+Cᵍᵐ₂[6].*Pˢᴳ³⁰_2)  

Onoff_cost=sum(Cᵁ²_1)+sum(Cᴰ²_1)+sum(Cᵁ³_1)+sum(Cᴰ³_1)+sum(Cᵁ⁴_1)+sum(Cᴰ⁴_1)+sum(Cᵁ⁵_1)+sum(Cᴰ⁵_1)+sum(Cᵁ²⁷_1)+sum(Cᴰ²⁷_1)+sum(Cᵁ³⁰_1)+sum(Cᴰ³⁰_1)  +sum(Cᵁ²_2)+sum(Cᴰ²_2)+sum(Cᵁ³_2)+sum(Cᴰ³_2)+sum(Cᵁ⁴_2)+sum(Cᴰ⁴_2)+sum(Cᵁ⁵_2)+sum(Cᴰ⁵_2)+sum(Cᵁ²⁷_2)+sum(Cᴰ²⁷_2)+sum(Cᵁ³⁰_2)+sum(Cᴰ³⁰_2)       

@objective(model_SP1, Min, No_load_cost+ Generation_cost+ Onoff_cost)  # objective function

#-----------------------------------Solve and Output Results
set_optimizer(model_SP1 , Gurobi.Optimizer)
# set_attribute(model, "limits/gap", 0.0280)
# set_time_limit_sec(model, 700.0)
optimize!(model_SP1)
   
    
    
    
    return JuMP.value.(No_load_cost+ Generation_cost+ Onoff_cost)
end