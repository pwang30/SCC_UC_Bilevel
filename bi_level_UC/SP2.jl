function SP2(  ψ_for_SP2, λᵇ, λˢ ,LB  )
    # We solve SP2 to calculate/choose the solution that is in favour of UL program
    
    λ_PMb= [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.75,0.75,0.75,0.75,1.2,1.2,1.2,0.75,0.75,0.75,0.75,1.2,1.2,1.2,1.2,0.4,0.4] 
    λ_PMs= [0,0,0,0,0,0,0,0.35,0.35,0.35,0.35,0.5,0.5,0.5,0.35,0.35,0.35,0.35,0.5,0.5,0.5,0.5,0,0]   
    
    P_Wmax_1=[2,1.5,1.6,1.8,1.3,0.6,2.8,3.3,3.9,4,3.3,2.9,2.7,2,0.2,3.2,5.1,3.1,1.8,2,1.3,1,2,3.8]                    # Wind_VPP1
    load_1=[2.2,1.8,3,6,5.8,5.2,5.6,3.8,2.5,2.7,3,2.6,2.2,2.1,4.2,5.8,6.2,6.3,6.5,6.6,6.3,6.2,6,5.7]                  # Load_VPP1
        
    P_Wmax_2=[4.7,5.1,4.3,4.1,3.8,3.9,4,5,5,4.8,3.9,4.3,5,5.2,5.8,5.6,1.6,0.9,5.8,4.1,3.6,3.5,3.1,3.8]                # Wind_VPP2
    load_2=[5,4,4,4.2,4.1,3.6,3.4,3.7,3.9,3.8,3.9,4,4.1,4.2,3.7,3,5.1,6.1,5.8,6.2,6.3,5.5,5,3.8]                      # Load_VPP2
        
    P_Wmax_3=[9.3,10.1,7.2,7.5,7.9,6.4,7.1,6.9,5.6,5.4,5.2,4,3.8,3,2.8,3.2,2.5,1.1,2.1,2.9,2.7,3,4.6,5.5]             # Wind_VPP3
    load_3=[4,2.1,1.1,1.1,0.7,1,1.9,3.6,3.8,4.2,5.8,5.6,5.8,5.6,5.7,6.1,8,10,9.4,8.2,6.2,5.5,4.8,2.2]                 # Load_VPP3
        
    T=length(load_1)
        
    a_MT=[0.08, 0.1,  0.15]         # MT cost coefficients for nonlinear part
    b_MT=[0.9,  0.6,  0.5]          # MT cost coefficients for linear part
    c_MT=[1.2, 1, 0.8]              # MT cost coefficients for no-load cost (Different from PowerTech)
        
    P_MT_min=[1,2,1]                # Lower bounds of MT                    (Different from PowerTech)
    P_MT_max=[6,5,4]                # Upper bounds of MT
    P_MT_dn=[3.5, 3.0, 2]           # Downwards rate of MT
    P_MT_up=[3.5, 3.0, 2]           # Upwards rate of MT
        
    λ_BS=[0.05,0.05,0.05]           # BS degradation cost coefficients
    P_BS_max=[0.6,0.6,1.2]          # Upper bounds of BS
    SOC_min=[0.2,0.2,0.2]           # Lower bounds of SOC 
    SOC_max=[0.9,0.9,0.9]           # Upper bounds of SOC 
    SOC_initial=[0.4,0.4,0.4]       # Initial SOC
    BS_capacity=[1,1,2]             # Capacity of BS
        
    trade_max=[10,10,10]            # Set the maximum amount of electricity purchased and sold betwwen VPP and DSO to 10
    trade_min=[0,0,0]               # Set the maximum amount of electricity purchased and sold betwwen VPP and DSO to 10
        
    M₂=30
    M₁=30
    
    #----------------------define model----------------
    model_SP2= Model()
    #-------------------------body-------------
         
    # Primal variables for VPP¹
    @variable(model_SP2, Pᵇ¹[1:T])
    @variable(model_SP2, Pˢ¹[1:T])
    @variable(model_SP2, Pᵂ¹[1:T])
    @variable(model_SP2, Pᴹ¹[1:T])
    @variable(model_SP2, Pᵇˢ¹[1:T])
    @variable(model_SP2, SoC¹[1:T])
    @variable(model_SP2, Pⱽᴾᴾ¹[1:T]) 
    @variable(model_SP2, u¹[1:T],Bin) 
    
    # Primal variables for VPP²
    @variable(model_SP2, Pᵇ²[1:T])
    @variable(model_SP2, Pˢ²[1:T])
    @variable(model_SP2, Pᵂ²[1:T])
    @variable(model_SP2, Pᴹ²[1:T])
    @variable(model_SP2, Pᵇˢ²[1:T])
    @variable(model_SP2, SoC²[1:T])
    @variable(model_SP2, Pⱽᴾᴾ²[1:T]) 
    @variable(model_SP2, u²[1:T],Bin) 
        
    # Primal variables for VPP³
    @variable(model_SP2, Pᵇ³[1:T])
    @variable(model_SP2, Pˢ³[1:T])
    @variable(model_SP2, Pᵂ³[1:T])
    @variable(model_SP2, Pᴹ³[1:T])
    @variable(model_SP2, Pᵇˢ³[1:T])
    @variable(model_SP2, SoC³[1:T])
    @variable(model_SP2, Pⱽᴾᴾ³[1:T]) 
    @variable(model_SP2, u³[1:T],Bin) 
                   
    # Primal variables for DSO
    @variable(model_SP2, Pᴰˢᴼᵇ[1:T]) 
    @variable(model_SP2, Pᴰˢᴼˢ[1:T]) 
    @variable(model_SP2, N₁[1:T],Bin) 
    @variable(model_SP2, N₂[1:T],Bin) 
    
    # This program is the original problem, which is considered to be the constraint for the SP2
    # Constraints for LL/VPPs
    @constraint(model_SP2, Pⱽᴾᴾ¹==Pᵇ¹-Pˢ¹) 
    @constraint(model_SP2, Pⱽᴾᴾ²==Pᵇ²-Pˢ²) 
    @constraint(model_SP2, Pⱽᴾᴾ³==Pᵇ³-Pˢ³) 
    @constraint(model_SP2, Pⱽᴾᴾ¹+Pᵂ¹+Pᴹ¹+Pᵇˢ¹==load_1)               # Power balance for VPP1
    @constraint(model_SP2, Pⱽᴾᴾ²+Pᵂ²+Pᴹ²+Pᵇˢ²==load_2)               # Power balance for VPP2              
    @constraint(model_SP2, Pⱽᴾᴾ³+Pᵂ³+Pᴹ³+Pᵇˢ³==load_3)               # Power balance for VPP3               
        
    for i in 1:T                                                      
        @constraint(model_SP2, trade_max[1]-Pᵇ¹[i]>=0)               # Trading volumes constraints (Buy)
        @constraint(model_SP2, Pᵇ¹[i]-trade_min[1]>=0)                   
        @constraint(model_SP2, trade_max[2]-Pᵇ²[i]>=0)       
        @constraint(model_SP2, Pᵇ²[i]-trade_min[2]>=0)
        @constraint(model_SP2, trade_max[3]-Pᵇ³[i]>=0)       
        @constraint(model_SP2, Pᵇ³[i]-trade_min[3]>=0)
        
        @constraint(model_SP2, trade_max[1]-Pˢ¹[i]>=0)               # Trading volumes constraints (Sell)      
        @constraint(model_SP2, Pˢ¹[i]-trade_min[1]>=0)                   
        @constraint(model_SP2, trade_max[2]-Pˢ²[i]>=0)       
        @constraint(model_SP2, Pˢ²[i]-trade_min[2]>=0)
        @constraint(model_SP2, trade_max[3]-Pˢ³[i]>=0)       
        @constraint(model_SP2, Pˢ³[i]-trade_min[3]>=0)                 
        
        @constraint(model_SP2, P_Wmax_1[i]-Pᵂ¹[i]>=0)               # WT output constraints                    
        @constraint(model_SP2, Pᵂ¹[i]>=0 )
        @constraint(model_SP2, P_Wmax_2[i]-Pᵂ²[i]>=0)                    
        @constraint(model_SP2, Pᵂ²[i]>=0 )
        @constraint(model_SP2, P_Wmax_3[i]-Pᵂ³[i]>=0)                    
        @constraint(model_SP2, Pᵂ³[i]>=0 )
         
        @constraint(model_SP2, P_MT_max[1]*u¹[i]-Pᴹ¹[i]>=0)         # MT output constraints with UC           
        @constraint(model_SP2, Pᴹ¹[i]-P_MT_min[1]*u¹[i]>=0)
        @constraint(model_SP2, P_MT_max[2]*u²[i]-Pᴹ²[i]>=0)             
        @constraint(model_SP2, Pᴹ²[i]-P_MT_min[2]*u²[i]>=0)
        @constraint(model_SP2, P_MT_max[3]*u³[i]-Pᴹ³[i]>=0)             
        @constraint(model_SP2, Pᴹ³[i]-P_MT_min[3]*u³[i]>=0)
        
        @constraint(model_SP2, Pᵇˢ¹[i]>=-P_BS_max[1])               # BS output constraints               
        @constraint(model_SP2, Pᵇˢ¹[i]<=P_BS_max[1])
        @constraint(model_SP2, Pᵇˢ²[i]>=-P_BS_max[2])                     
        @constraint(model_SP2, Pᵇˢ²[i]<=P_BS_max[2])
        @constraint(model_SP2, Pᵇˢ³[i]>=-P_BS_max[3])                     
        @constraint(model_SP2, Pᵇˢ³[i]<=P_BS_max[3])
        
        @constraint(model_SP2, SOC_min[1]<=SoC¹[i])                 # SOC output constraints              
        @constraint(model_SP2, SoC¹[i]<=SOC_max[1]) 
        @constraint(model_SP2, SOC_min[2]<=SoC²[i])                       
        @constraint(model_SP2, SoC²[i]<=SOC_max[2]) 
        @constraint(model_SP2, SOC_min[3]<=SoC³[i])                       
        @constraint(model_SP2, SoC³[i]<=SOC_max[3])                        
        end
        
    @constraint(model_SP2, SoC¹[1]==SOC_initial[1]-Pᵇˢ¹[1]/BS_capacity[1])         # SOC initial capacity constraints
    @constraint(model_SP2, SoC²[1]==SOC_initial[2]-Pᵇˢ²[1]/BS_capacity[2])   
    @constraint(model_SP2, SoC³[1]==SOC_initial[3]-Pᵇˢ³[1]/BS_capacity[3])   
        
    for t in 2:T
        @constraint(model_SP2, SoC¹[t]==SoC¹[t-1]-Pᵇˢ¹[t]/BS_capacity[1])          # Relationship betwwen SOC and Pᵇˢ
        @constraint(model_SP2, SoC²[t]==SoC²[t-1]-Pᵇˢ²[t]/BS_capacity[2])   
        @constraint(model_SP2, SoC³[t]==SoC³[t-1]-Pᵇˢ³[t]/BS_capacity[3]) 
        
        @constraint(model_SP2, Pᴹ¹[t]-Pᴹ¹[t-1]<=P_MT_up[1])                        # MT downwards/upwards rates constraints 
        @constraint(model_SP2, Pᴹ¹[t-1]-Pᴹ¹[t]<=P_MT_dn[1])
        @constraint(model_SP2, Pᴹ²[t]-Pᴹ²[t-1]<=P_MT_up[2])
        @constraint(model_SP2, Pᴹ²[t-1]-Pᴹ²[t]<=P_MT_dn[2])
        @constraint(model_SP2, Pᴹ³[t]-Pᴹ³[t-1]<=P_MT_up[3])
        @constraint(model_SP2, Pᴹ³[t-1]-Pᴹ³[t]<=P_MT_dn[3])
    end
        
        
    LL_obj_VPP1=sum(λᵇ.*Pᵇ¹ -λˢ.*Pˢ¹ +a_MT[1]*Pᴹ¹.*Pᴹ¹+b_MT[1]*Pᴹ¹ +c_MT[1]*u¹ +λ_BS[1]*Pᵇˢ¹.*Pᵇˢ¹)       # Obj for VPP1    
    LL_obj_VPP2=sum(λᵇ.*Pᵇ² -λˢ.*Pˢ² +a_MT[2]*Pᴹ².*Pᴹ²+b_MT[2]*Pᴹ² +c_MT[2]*u² +λ_BS[2]*Pᵇˢ².*Pᵇˢ²)       # Obj for VPP2 
    LL_obj_VPP3=sum(λᵇ.*Pᵇ³ -λˢ.*Pˢ³ +a_MT[3]*Pᴹ³.*Pᴹ³+b_MT[3]*Pᴹ³ +c_MT[3]*u³ +λ_BS[3]*Pᵇˢ³.*Pᵇˢ³)       # Obj for VPP3
        
    @constraint(model_SP2,LL_obj_VPP1+LL_obj_VPP2+LL_obj_VPP3<=ψ_for_SP2) 
    
    # Constraints for UL/DSO
    @constraint(model_SP2,N₁-N₂==0)                                    # N₂ & N₁ must be the same
    
    # big-M formulation of judgement 2
    @constraint(model_SP2, Pⱽᴾᴾ¹+Pⱽᴾᴾ²+Pⱽᴾᴾ³<=M₂*N₂)
    @constraint(model_SP2, -M₂*(ones(1,T)'-N₂).<=Pⱽᴾᴾ¹+Pⱽᴾᴾ²+Pⱽᴾᴾ³)
    @constraint(model_SP2, -M₂*N₂.<=Pᴰˢᴼˢ+(Pⱽᴾᴾ¹+Pⱽᴾᴾ²+Pⱽᴾᴾ³))
    @constraint(model_SP2, Pᴰˢᴼˢ+(Pⱽᴾᴾ¹+Pⱽᴾᴾ²+Pⱽᴾᴾ³).<=M₂*N₂)
    @constraint(model_SP2, -M₂*(ones(1,T)'-N₂).<=Pᴰˢᴼˢ)
    @constraint(model_SP2, Pᴰˢᴼˢ.<=M₂*(ones(1,T)'-N₂))
    
    # big-M formulation of judgement 1  
    @constraint(model_SP2, Pⱽᴾᴾ¹+Pⱽᴾᴾ²+Pⱽᴾᴾ³<=M₁*N₁)
    @constraint(model_SP2, -M₁*(ones(1,T)'-N₁).<=Pⱽᴾᴾ¹+Pⱽᴾᴾ²+Pⱽᴾᴾ³)
    @constraint(model_SP2, -M₁*(ones(1,T)'-N₁).<=Pᴰˢᴼᵇ-(Pⱽᴾᴾ¹+Pⱽᴾᴾ²+Pⱽᴾᴾ³))
    @constraint(model_SP2, Pᴰˢᴼᵇ-(Pⱽᴾᴾ¹+Pⱽᴾᴾ²+Pⱽᴾᴾ³).<=M₁*(ones(1,T)'-N₁))
    @constraint(model_SP2, -M₁*N₁.<=Pᴰˢᴼᵇ)
    @constraint(model_SP2, Pᴰˢᴼᵇ.<=M₁*N₁)
    
    @objective(model_SP2, Max,  sum(λ_PMs.*Pᴰˢᴼˢ)-sum(λ_PMb.*Pᴰˢᴼᵇ) + sum(λᵇ.*(Pᵇ¹+Pᵇ²+Pᵇ³))-sum(λˢ.*(Pˢ¹+Pˢ²+Pˢ³)))  
    
    set_optimizer(model_SP2, Gurobi.Optimizer)
    optimize!(model_SP2)
    
    u¹=JuMP.value.(u¹)
    u²=JuMP.value.(u²)
    u³=JuMP.value.(u³)
    LB=max(LB, JuMP.value.( sum(λ_PMs.*Pᴰˢᴼˢ)-sum(λ_PMb.*Pᴰˢᴼᵇ) + sum(λᵇ.*(Pᵇ¹+Pᵇ²+Pᵇ³))-sum(λˢ.*(Pˢ¹+Pˢ²+Pˢ³)) ))
    
    return u¹, u², u³, LB
    end
    