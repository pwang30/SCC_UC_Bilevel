function MP(   x₂₁,x₂₂,x₃₁,x₃₂,x₄₁,x₄₂,x₅₁,x₅₂,x₂₇₁,x₂₇₂,x₃₀₁,x₃₀₂, LB )
    # MP, the most complex part, including primal and dual variables, also the complmentarity constraints
    
    #----------------------related parameters input---------------
    
    P_Wmax_1=[2,1.5,1.6,1.8,1.3,0.6,2.8,3.3,3.9,4,3.3,2.9,2.7,2,0.2,3.2,5.1,3.1,1.8,2,1.3,1,2,3.8]                    # Wind_VPP1
    load_1=[2.2,1.8,3,6,5.8,5.2,5.6,3.8,2.5,2.7,3,2.6,2.2,2.1,4.2,5.8,6.2,6.3,6.5,6.6,6.3,6.2,6,5.7]                  # Load_VPP1
    
    P_Wmax_2=[4.7,5.1,4.3,4.1,3.8,3.9,4,5,5,4.8,3.9,4.3,5,5.2,5.8,5.6,1.6,0.9,5.8,4.1,3.6,3.5,3.1,3.8]                # Wind_VPP2
    load_2=[5,4,4,4.2,4.1,3.6,3.4,3.7,3.9,3.8,3.9,4,4.1,4.2,3.7,3,5.1,6.1,5.8,6.2,6.3,5.5,5,3.8]                      # Load_VPP2
    
    P_Wmax_3=[9.3,10.1,7.2,7.5,7.9,6.4,7.1,6.9,5.6,5.4,5.2,4,3.8,3,2.8,3.2,2.5,1.1,2.1,2.9,2.7,3,4.6,5.5]             # Wind_VPP3
    load_3=[4,2.1,1.1,1.1,0.7,1,1.9,3.6,3.8,4.2,5.8,5.6,5.8,5.6,5.7,6.1,8,10,9.4,8.2,6.2,5.5,4.8,2.2]                 # Load_VPP3
    
    λ_PMb= [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.75,0.75,0.75,0.75,1.2,1.2,1.2,0.75,0.75,0.75,0.75,1.2,1.2,1.2,1.2,0.4,0.4] 
    λ_PMs= [0,0,0,0,0,0,0,0.35,0.35,0.35,0.35,0.5,0.5,0.5,0.35,0.35,0.35,0.35,0.5,0.5,0.5,0.5,0,0]
    
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
    
    M₁=30
    M₂=30
    
    if length(u¹)==0
        
    #----------------------define model----------------
        model_MP= Model()
    
    #-------------------------body-------------
    @variable(model_MP, λᵇ[1:T])          # Prices set by DSO
    @variable(model_MP, λˢ[1:T])
    # Dual variables in lower level-------------------------------
        @variable(model_MP, μᵇ¹ₘₐₓ[1:T]>=0)    # dual variables for buying power from DSO
        @variable(model_MP, μᵇ¹ₘᵢₙ[1:T]>=0)  
        @variable(model_MP, μᵇ²ₘₐₓ[1:T]>=0)
        @variable(model_MP, μᵇ²ₘᵢₙ[1:T]>=0)  
        @variable(model_MP, μᵇ³ₘₐₓ[1:T]>=0)
        @variable(model_MP, μᵇ³ₘᵢₙ[1:T]>=0)  
    
        @variable(model_MP, μˢ¹ₘₐₓ[1:T]>=0)   # dual variables for selling power to DSO
        @variable(model_MP, μˢ¹ₘᵢₙ[1:T]>=0) 
        @variable(model_MP, μˢ²ₘₐₓ[1:T]>=0) 
        @variable(model_MP, μˢ²ₘᵢₙ[1:T]>=0)
        @variable(model_MP, μˢ³ₘₐₓ[1:T]>=0) 
        @variable(model_MP, μˢ³ₘᵢₙ[1:T]>=0)
    
        @variable(model_MP, μᵂ¹ₘₐₓ[1:T]>=0)  # dual variables for output bounds of WT
        @variable(model_MP, μᵂ¹ₘᵢₙ[1:T]>=0) 
        @variable(model_MP, μᵂ²ₘₐₓ[1:T]>=0) 
        @variable(model_MP, μᵂ²ₘᵢₙ[1:T]>=0)
        @variable(model_MP, μᵂ³ₘₐₓ[1:T]>=0) 
        @variable(model_MP, μᵂ³ₘᵢₙ[1:T]>=0)
    
        @variable(model_MP, μᵇˢ¹ₘₐₓ[1:T]>=0)    # dual variables for output bounds of BS
        @variable(model_MP, μᵇˢ¹ₘᵢₙ[1:T]>=0)
        @variable(model_MP, μᵇˢ²ₘₐₓ[1:T]>=0) 
        @variable(model_MP, μᵇˢ²ₘᵢₙ[1:T]>=0)
        @variable(model_MP, μᵇˢ³ₘₐₓ[1:T]>=0) 
        @variable(model_MP, μᵇˢ³ₘᵢₙ[1:T]>=0)
    
        @variable(model_MP, μˢᵒᶜ¹ₘₐₓ[1:T]>=0)   # dual variables for bounds of SoC
        @variable(model_MP, μˢᵒᶜ¹ₘᵢₙ[1:T]>=0)
        @variable(model_MP, μˢᵒᶜ²ₘₐₓ[1:T]>=0) 
        @variable(model_MP, μˢᵒᶜ²ₘᵢₙ[1:T]>=0)
        @variable(model_MP, μˢᵒᶜ³ₘₐₓ[1:T]>=0) 
        @variable(model_MP, μˢᵒᶜ³ₘᵢₙ[1:T]>=0)
    
        @variable(model_MP, μᴹ¹ₘₐₓ[1:T]>=0)     # dual variables for output bounds of MT
        @variable(model_MP, μᴹ¹ₘᵢₙ[1:T]>=0) 
        @variable(model_MP, μᴹ²ₘₐₓ[1:T]>=0) 
        @variable(model_MP, μᴹ²ₘᵢₙ[1:T]>=0)
        @variable(model_MP, μᴹ³ₘₐₓ[1:T]>=0) 
        @variable(model_MP, μᴹ³ₘᵢₙ[1:T]>=0)
    
        @variable(model_MP, μᵁ¹ᵣ[1:T]>=0)      # dual variables for upwards/downwards rates of MT
        @variable(model_MP, μᴰ¹ᵣ[1:T]>=0)       
        @variable(model_MP, μᵁ²ᵣ[1:T]>=0)       
        @variable(model_MP, μᴰ²ᵣ[1:T]>=0)       
        @variable(model_MP, μᵁ³ᵣ[1:T]>=0)       
        @variable(model_MP, μᴰ³ᵣ[1:T]>=0)        
    
        @variable(model_MP, v¹[1:T])        # for power balance equation
        @variable(model_MP, v²[1:T])
        @variable(model_MP, v³[1:T])
    
        @variable(model_MP, w¹[1:T])        # variables for relationship between BS output and SoC
        @variable(model_MP, w²[1:T])         
        @variable(model_MP, w³[1:T])       
    
     # Primal variables in lower level, existed as a bound for real solution-------------------------------
     # Primal variables for VPP¹
        @variable(model_MP, Pᵇ¹[1:T])
        @variable(model_MP, Pˢ¹[1:T])
        @variable(model_MP, Pᵂ¹[1:T])
        @variable(model_MP, Pᴹ¹[1:T])
        @variable(model_MP, Pᵇˢ¹[1:T])
        @variable(model_MP, SoC¹[1:T])
        @variable(model_MP, u¹[1:T],Bin) 
    
    # Primal variables for VPP²
        @variable(model_MP, Pᵇ²[1:T])
        @variable(model_MP, Pˢ²[1:T])
        @variable(model_MP, Pᵂ²[1:T])
        @variable(model_MP, Pᴹ²[1:T])
        @variable(model_MP, Pᵇˢ²[1:T])
        @variable(model_MP, SoC²[1:T])
        @variable(model_MP, u²[1:T],Bin) 
        
    # Primal variables for VPP³
        @variable(model_MP, Pᵇ³[1:T])
        @variable(model_MP, Pˢ³[1:T])
        @variable(model_MP, Pᵂ³[1:T])
        @variable(model_MP, Pᴹ³[1:T])
        @variable(model_MP, Pᵇˢ³[1:T])
        @variable(model_MP, SoC³[1:T])
        @variable(model_MP, u³[1:T],Bin) 
    
    # Primal variables in lower level, existed as real solution-------------------------------
        # for VPP¹
        @variable(model_MP, Pᵇ¹⁰[1:T])
        @variable(model_MP, Pˢ¹⁰[1:T])
        @variable(model_MP, Pᵂ¹⁰[1:T])
        @variable(model_MP, Pᴹ¹⁰[1:T])
        @variable(model_MP, Pᵇˢ¹⁰[1:T])
        @variable(model_MP, SoC¹⁰[1:T])
        @variable(model_MP, Pⱽᴾᴾ¹⁰[1:T]) 
        @variable(model_MP, u¹⁰[1:T],Bin) 
    
        # for VPP²
        @variable(model_MP, Pᵇ²⁰[1:T])
        @variable(model_MP, Pˢ²⁰[1:T])
        @variable(model_MP, Pᵂ²⁰[1:T])
        @variable(model_MP, Pᴹ²⁰[1:T])
        @variable(model_MP, Pᵇˢ²⁰[1:T])
        @variable(model_MP, SoC²⁰[1:T])
        @variable(model_MP, Pⱽᴾᴾ²⁰[1:T]) 
        @variable(model_MP, u²⁰[1:T],Bin) 
    
        # for VPP³
        @variable(model_MP, Pᵇ³⁰[1:T])
        @variable(model_MP, Pˢ³⁰[1:T])
        @variable(model_MP, Pᵂ³⁰[1:T])
        @variable(model_MP, Pᴹ³⁰[1:T])
        @variable(model_MP, Pᵇˢ³⁰[1:T])
        @variable(model_MP, SoC³⁰[1:T])
        @variable(model_MP, Pⱽᴾᴾ³⁰[1:T]) 
        @variable(model_MP, u³⁰[1:T],Bin) 
    
    
    # Write constraints for real solution in LL program, in this part, there is no dual variables, can be seen as nomal optimization problem
    @constraint(model_MP, Pᵇ¹⁰-Pˢ¹⁰==Pⱽᴾᴾ¹⁰)                     
    @constraint(model_MP, Pᵇ²⁰-Pˢ²⁰==Pⱽᴾᴾ²⁰)                  
    @constraint(model_MP, Pᵇ³⁰-Pˢ³⁰==Pⱽᴾᴾ³⁰)           
    @constraint(model_MP, Pⱽᴾᴾ¹⁰+Pᵂ¹⁰+Pᴹ¹⁰+Pᵇˢ¹⁰==load_1)          # Power balance for VPP¹,   dual variable: v¹
    @constraint(model_MP, Pⱽᴾᴾ²⁰+Pᵂ²⁰+Pᴹ²⁰+Pᵇˢ²⁰==load_2)          # Power balance for VPP²,   dual variable: v² 
    @constraint(model_MP, Pⱽᴾᴾ³⁰+Pᵂ³⁰+Pᴹ³⁰+Pᵇˢ³⁰==load_3)          # Power balance for VPP³,   dual variable: v³
    
    for i in 1:T                                                   
        @constraint(model_MP, trade_max[1]-Pᵇ¹⁰[i]>=0)             # Lower and Upper bounds for trading volumes (buy) of VPPs
        @constraint(model_MP, Pᵇ¹⁰[i]-trade_min[1]>=0 )            
        @constraint(model_MP, trade_max[1]-Pᵇ²⁰[i]>=0)       
        @constraint(model_MP, Pᵇ²⁰[i]-trade_min[1]>=0 )                   
        @constraint(model_MP, trade_max[1]-Pᵇ³⁰[i]>=0)       
        @constraint(model_MP, Pᵇ³⁰[i]-trade_min[1]>=0 )                   
    
        @constraint(model_MP, trade_max[1]-Pˢ¹⁰[i]>=0)             # Lower and Upper bounds for trading volumes (sell) of VPPs
        @constraint(model_MP, Pˢ¹⁰[i]-trade_min[1]>=0 )            
        @constraint(model_MP, trade_max[1]-Pˢ²⁰[i]>=0)       
        @constraint(model_MP, Pˢ²⁰[i]-trade_min[1]>=0 )                   
        @constraint(model_MP, trade_max[1]-Pˢ³⁰[i]>=0)       
        @constraint(model_MP, Pˢ³⁰[i]-trade_min[1]>=0 )
    
        @constraint(model_MP, P_Wmax_1[i]-Pᵂ¹⁰[i]>=0 )             # Lower and Upper bounds for output of WT
        @constraint(model_MP, Pᵂ¹⁰[i]>=0 )
        @constraint(model_MP, P_Wmax_2[i]-Pᵂ²⁰[i]>=0 )              
        @constraint(model_MP, Pᵂ²⁰[i]>=0 )
        @constraint(model_MP, P_Wmax_3[i]-Pᵂ³⁰[i]>=0 )              
        @constraint(model_MP, Pᵂ³⁰[i]>=0 )
        
        @constraint(model_MP, P_MT_max[1]*u¹⁰[i]-Pᴹ¹⁰[i]>=0 )      # Lower and Upper bounds for output of MT with UC
        @constraint(model_MP, Pᴹ¹⁰[i]-P_MT_min[1]*u¹⁰[i]>=0 )
        @constraint(model_MP, P_MT_max[2]*u²⁰[i]-Pᴹ²⁰[i]>=0 )          
        @constraint(model_MP, Pᴹ²⁰[i]-P_MT_min[2]*u²⁰[i]>=0 )
        @constraint(model_MP, P_MT_max[3]*u³⁰[i]-Pᴹ³⁰[i]>=0 )              
        @constraint(model_MP, Pᴹ³⁰[i]-P_MT_min[3]*u³⁰[i]>=0 )
    
        @constraint(model_MP, Pᵇˢ¹⁰[i]>=-P_BS_max[1])              # BS output constraints               
        @constraint(model_MP, Pᵇˢ¹⁰[i]<=P_BS_max[1])
        @constraint(model_MP, Pᵇˢ²⁰[i]>=-P_BS_max[2])                     
        @constraint(model_MP, Pᵇˢ²⁰[i]<=P_BS_max[2])
        @constraint(model_MP, Pᵇˢ³⁰[i]>=-P_BS_max[3])                     
        @constraint(model_MP, Pᵇˢ³⁰[i]<=P_BS_max[3])
    
        @constraint(model_MP, SOC_min[1]<=SoC¹⁰[i])                # Lower and Upper bounds for SoC 
        @constraint(model_MP, SoC¹⁰[i]<=SOC_max[1])   
        @constraint(model_MP, SOC_min[2]<=SoC²⁰[i])                 
        @constraint(model_MP, SoC²⁰[i]<=SOC_max[2]) 
        @constraint(model_MP, SOC_min[3]<=SoC³⁰[i])                 
        @constraint(model_MP, SoC³⁰[i]<=SOC_max[3])                      
    
    end
    
    @constraint(model_MP, SoC¹⁰[1]==SOC_initial[1]-Pᵇˢ¹⁰[1]/BS_capacity[1])         # SOC initial capacity constraints
    @constraint(model_MP, SoC²⁰[1]==SOC_initial[2]-Pᵇˢ²⁰[1]/BS_capacity[2])   
    @constraint(model_MP, SoC³⁰[1]==SOC_initial[3]-Pᵇˢ³⁰[1]/BS_capacity[3])   
        
    for t in 2:T
        @constraint(model_MP, SoC¹⁰[t]==SoC¹⁰[t-1]-Pᵇˢ¹⁰[t]/BS_capacity[1])          # Relationship betwwen SOC and Pᵇˢ⁰
        @constraint(model_MP, SoC²⁰[t]==SoC²⁰[t-1]-Pᵇˢ²⁰[t]/BS_capacity[2])   
        @constraint(model_MP, SoC³⁰[t]==SoC³⁰[t-1]-Pᵇˢ³⁰[t]/BS_capacity[3]) 
        
        @constraint(model_MP, Pᴹ¹⁰[t]-Pᴹ¹⁰[t-1]<=P_MT_up[1])                         # MT downwards/upwards rates constraints 
        @constraint(model_MP, Pᴹ¹⁰[t-1]-Pᴹ¹⁰[t]<=P_MT_dn[1])
        @constraint(model_MP, Pᴹ²⁰[t]-Pᴹ²⁰[t-1]<=P_MT_up[2])
        @constraint(model_MP, Pᴹ²⁰[t-1]-Pᴹ²⁰[t]<=P_MT_dn[2])
        @constraint(model_MP, Pᴹ³⁰[t]-Pᴹ³⁰[t-1]<=P_MT_up[3])
        @constraint(model_MP, Pᴹ³⁰[t-1]-Pᴹ³⁰[t]<=P_MT_dn[3])
    end
    
    
    # Complmentarity Constraints (CC) in lower level-------------------------------
    # 0<=yʲ.(Pᵗπʲ-wᵗ)>=0           primal variables time dual constraints
    # 0<=πʲ.(R-Kx-Nzʲ-Pyʲ)>=0      dual variables time primal constraints
             
    @constraint(model_MP, Pᵇ¹-Pˢ¹+Pᵂ¹+Pᴹ¹+Pᵇˢ¹==load_1)          # Power balance for VPP¹,   dual variable: v¹
    @constraint(model_MP, Pᵇ²-Pˢ²+Pᵂ²+Pᴹ²+Pᵇˢ²==load_2)          # Power balance for VPP²,   dual variable: v² 
    @constraint(model_MP, Pᵇ³-Pˢ³+Pᵂ³+Pᴹ³+Pᵇˢ³==load_3)          # Power balance for VPP³,   dual variable: v³
    
    # Firstly write     0<=πʲ.(R-Kx-Nzʲ-Pyʲ)>=0     dual variables time primal constraints
    for i in 1:T
        @constraint(model_MP, μᵇ¹ₘₐₓ[i]*(trade_max[1]-Pᵇ¹[i])==0)  # Bounds (buy) for trading volume of VPP1: Pᵇ¹,  dual variable: μᵇ¹ₘₐₓ and μᵇ¹ₘᵢₙ
        @constraint(model_MP, trade_max[1]-Pᵇ¹[i]>=0 )
        @constraint(model_MP, μᵇ¹ₘᵢₙ[i]* (Pᵇ¹[i]-trade_min[1])==0 )    
        @constraint(model_MP, Pᵇ¹[i]-trade_min[1]>=0 )
        @constraint(model_MP, μᵇ²ₘₐₓ[i]*(trade_max[2]-Pᵇ²[i])==0)  # Bounds (buy) for trading volume of VPP2: Pᵇ²,  dual variable: μᵇ²ₘₐₓ and μᵇ²ₘᵢₙ
        @constraint(model_MP, trade_max[2]-Pᵇ²[i]>=0 )
        @constraint(model_MP, μᵇ²ₘᵢₙ[i]* (Pᵇ²[i]-trade_min[2])==0 )    
        @constraint(model_MP, Pᵇ²[i]-trade_min[2]>=0 )
        @constraint(model_MP, μᵇ³ₘₐₓ[i]*(trade_max[3]-Pᵇ³[i])==0)  # Bounds (buy) for trading volume of VPP3: Pᵇ³,  dual variable: μᵇ³ₘₐₓ and μᵇ³ₘᵢₙ
        @constraint(model_MP, trade_max[3]-Pᵇ³[i]>=0 )
        @constraint(model_MP, μᵇ³ₘᵢₙ[i]* (Pᵇ³[i]-trade_min[3])==0 )    
        @constraint(model_MP, Pᵇ³[i]-trade_min[3]>=0 )
        
        @constraint(model_MP, μˢ¹ₘₐₓ[i]*(trade_max[1]-Pˢ¹[i])==0)  # Bounds (sell) for trading volume of VPP1: Pˢ¹,  dual variable: μˢ¹ₘₐₓ and μˢ¹ₘᵢₙ
        @constraint(model_MP, trade_max[1]-Pˢ¹[i]>=0)
        @constraint(model_MP, μˢ¹ₘᵢₙ[i]* (Pˢ¹[i]-trade_min[1])==0)    
        @constraint(model_MP, Pˢ¹[i]-trade_min[1]>=0)
        @constraint(model_MP, μˢ²ₘₐₓ[i]*(trade_max[2]-Pˢ²[i])==0)  # Bounds (sell) for trading volume of VPP2: Pˢ²,  dual variable: μˢ²ₘₐₓ and μˢ²ₘᵢₙ
        @constraint(model_MP, trade_max[2]-Pˢ²[i]>=0)
        @constraint(model_MP, μˢ²ₘᵢₙ[i]* (Pˢ²[i]-trade_min[2])==0)    
        @constraint(model_MP, Pˢ²[i]-trade_min[2]>=0)
        @constraint(model_MP, μˢ³ₘₐₓ[i]*(trade_max[3]-Pˢ³[i])==0)  # Bounds (sell) for trading volume of VPP3: Pˢ³,  dual variable: μˢ³ₘₐₓ and μˢ³ₘᵢₙ
        @constraint(model_MP, trade_max[3]-Pˢ³[i]>=0)
        @constraint(model_MP, μˢ³ₘᵢₙ[i]* (Pˢ³[i]-trade_min[3])==0)    
        @constraint(model_MP, Pˢ³[i]-trade_min[3]>=0)
    
        @constraint(model_MP, μᵂ¹ₘₐₓ[i]* (P_Wmax_1[i]-Pᵂ¹[i])==0)    # Bounds for output of WT in VPP1: Pᵂ¹, dual variable: μᵂ¹ₘₐₓ and μᵂ¹ₘᵢₙ
        @constraint(model_MP, P_Wmax_1[i]-Pᵂ¹[i]>=0)
        @constraint(model_MP, μᵂ¹ₘᵢₙ[i]* (Pᵂ¹[i])==0)                        
        @constraint(model_MP, Pᵂ¹[i]>=0)
        @constraint(model_MP, μᵂ²ₘₐₓ[i]* (P_Wmax_2[i]-Pᵂ²[i])==0)    # Bounds for output of WT in VPP2: Pᵂ², dual variable: μᵂ²ₘₐₓ and μᵂ²ₘᵢₙ
        @constraint(model_MP, P_Wmax_2[i]-Pᵂ²[i]>=0)
        @constraint(model_MP, μᵂ²ₘᵢₙ[i]* (Pᵂ²[i])==0)                        
        @constraint(model_MP, Pᵂ²[i]>=0)
        @constraint(model_MP, μᵂ³ₘₐₓ[i]* (P_Wmax_3[i]-Pᵂ³[i])==0)    # Bounds for output of WT in VPP2: Pᵂ³, dual variable: μᵂ³ₘₐₓ and μᵂ³ₘᵢₙ
        @constraint(model_MP, P_Wmax_3[i]-Pᵂ³[i]>=0)
        @constraint(model_MP, μᵂ³ₘᵢₙ[i]* (Pᵂ³[i])==0)                        
        @constraint(model_MP, Pᵂ³[i]>=0)
    
        @constraint(model_MP, μᴹ¹ₘₐₓ[i]* (P_MT_max[1]*u¹[i]-Pᴹ¹[i])==0 )  # Bounds for output of MT in VPP1, with UC: Pᴹ¹, dual variable: μᴹ¹ₘₐₓ and μᴹ¹ₘᵢₙ
        @constraint(model_MP, P_MT_max[1]*u¹[i]-Pᴹ¹[i]>=0 )
        @constraint(model_MP, μᴹ¹ₘᵢₙ[i]* (Pᴹ¹[i]-P_MT_min[1]*u¹[i])==0 )       
        @constraint(model_MP, Pᴹ¹[i]-P_MT_min[1]*u¹[i]>=0 )
        @constraint(model_MP, μᴹ²ₘₐₓ[i]* (P_MT_max[2]*u²[i]-Pᴹ²[i])==0 )  # Bounds for output of MT in VPP2, with UC: Pᴹ², dual variable: μᴹ²ₘₐₓ and μᴹ²ₘᵢₙ
        @constraint(model_MP, P_MT_max[2]*u²[i]-Pᴹ²[i]>=0 )
        @constraint(model_MP, μᴹ²ₘᵢₙ[i]* (Pᴹ²[i]-P_MT_min[2]*u²[i])==0 )       
        @constraint(model_MP, Pᴹ²[i]-P_MT_min[2]*u²[i]>=0 )
        @constraint(model_MP, μᴹ³ₘₐₓ[i]* (P_MT_max[3]*u³[i]-Pᴹ³[i])==0 )  # Bounds for output of MT in VPP3, with UC: Pᴹ³, dual variable: μᴹ³ₘₐₓ and μᴹ³ₘᵢₙ
        @constraint(model_MP, P_MT_max[3]*u³[i]-Pᴹ³[i]>=0 )
        @constraint(model_MP, μᴹ³ₘᵢₙ[i]* (Pᴹ³[i]-P_MT_min[3]*u³[i])==0 )       
        @constraint(model_MP, Pᴹ³[i]-P_MT_min[3]*u³[i]>=0 )
    
        @constraint(model_MP, μᵇˢ¹ₘₐₓ[i]* (P_BS_max[1]-Pᵇˢ¹[i])==0 )       # Bounds for output of BS in VPP1: Pᵇˢ¹, dual variable: μᵇˢ¹ₘₐₓ and μᵇˢ¹ₘᵢₙ
        @constraint(model_MP, P_BS_max[1]-Pᵇˢ¹[i]>=0 )
        @constraint(model_MP, μᵇˢ¹ₘᵢₙ[i]* (Pᵇˢ¹[i]+P_BS_max[1])==0 )  
        @constraint(model_MP, Pᵇˢ¹[i]+P_BS_max[1]>=0 )
        @constraint(model_MP, μᵇˢ²ₘₐₓ[i]* (P_BS_max[2]-Pᵇˢ²[i])==0 )       # Bounds for output of BS in VPP2: Pᵇˢ², dual variable: μᵇˢ²ₘₐₓ and μᵇˢ²ₘᵢₙ
        @constraint(model_MP, P_BS_max[2]-Pᵇˢ²[i]>=0 )
        @constraint(model_MP, μᵇˢ²ₘᵢₙ[i]* (Pᵇˢ²[i]+P_BS_max[2])==0 )  
        @constraint(model_MP, Pᵇˢ²[i]+P_BS_max[2]>=0 )
        @constraint(model_MP, μᵇˢ³ₘₐₓ[i]* (P_BS_max[3]-Pᵇˢ³[i])==0 )       # Bounds for output of BS in VPP3: Pᵇˢ³, dual variable: μᵇˢ³ₘₐₓ and μᵇˢ³ₘᵢₙ
        @constraint(model_MP, P_BS_max[3]-Pᵇˢ³[i]>=0 )
        @constraint(model_MP, μᵇˢ³ₘᵢₙ[i]* (Pᵇˢ³[i]+P_BS_max[3])==0 )  
        @constraint(model_MP, Pᵇˢ³[i]+P_BS_max[3]>=0 )
    
        @constraint(model_MP, SoC¹[i]<=SOC_max[1])                         # Bounds for SoC in VPP1: SoC¹, dual variable:  μˢᵒᶜ¹ₘₐₓ and  μˢᵒᶜ¹ₘᵢₙ
        @constraint(model_MP, μˢᵒᶜ¹ₘₐₓ[i]*(SOC_max[1]-SoC¹[i])==0)
        @constraint(model_MP, SOC_min[1]<=SoC¹[i])   
        @constraint(model_MP, μˢᵒᶜ¹ₘᵢₙ[i]*(SoC¹[i]-SOC_min[1])==0)                       
        @constraint(model_MP, SoC²[i]<=SOC_max[2])                         # Bounds for SoC in VPP2: SoC², dual variable:  μˢᵒᶜ²ₘₐₓ and  μˢᵒᶜ²ₘᵢₙ
        @constraint(model_MP, μˢᵒᶜ²ₘₐₓ[i]*(SOC_max[2]-SoC²[i])==0)
        @constraint(model_MP, SOC_min[2]<=SoC²[i])   
        @constraint(model_MP, μˢᵒᶜ²ₘᵢₙ[i]*(SoC²[i]-SOC_min[2])==0)
        @constraint(model_MP, SoC³[i]<=SOC_max[3])                         # Bounds for SoC in VPP3: SoC³, dual variable:  μˢᵒᶜ³ₘₐₓ and  μˢᵒᶜ³ₘᵢₙ
        @constraint(model_MP, μˢᵒᶜ³ₘₐₓ[i]*(SOC_max[3]-SoC³[i])==0)
        @constraint(model_MP, SOC_min[3]<=SoC³[i])   
        @constraint(model_MP, μˢᵒᶜ³ₘᵢₙ[i]*(SoC³[i]-SOC_min[3])==0)     
    
    end
    
    @constraint(model_MP, SoC¹[1]==SOC_initial[1]-Pᵇˢ¹[1]/BS_capacity[1])    # SOC initial capacity constraints in VPP1, dual variable:  w¹ 
    @constraint(model_MP, SoC²[1]==SOC_initial[2]-Pᵇˢ²[1]/BS_capacity[2])    # SOC initial capacity constraints in VPP2, dual variable:  w² 
    @constraint(model_MP, SoC³[1]==SOC_initial[3]-Pᵇˢ³[1]/BS_capacity[3])    # SOC initial capacity constraints in VPP3, dual variable:  w³ 
        
    for t in 2:T
        @constraint(model_MP, SoC¹[t]==SoC¹[t-1]-Pᵇˢ¹[t]/BS_capacity[1])     # Relationship betwwen SOC and Pᵇˢ⁰, dual variable:  w¹
        @constraint(model_MP, SoC²[t]==SoC²[t-1]-Pᵇˢ²[t]/BS_capacity[2])     # Relationship betwwen SOC and Pᵇˢ⁰, dual variable:  w²
        @constraint(model_MP, SoC³[t]==SoC³[t-1]-Pᵇˢ³[t]/BS_capacity[3])     # Relationship betwwen SOC and Pᵇˢ⁰, dual variable:  w³
        
        @constraint(model_MP, Pᴹ¹[t]-Pᴹ¹[t-1]<=P_MT_up[1])                 # MT downwards/upwards rates constraints in VPP1, dual variable:  μᵁ¹ᵣ and μᴰ¹ᵣ
        @constraint(model_MP, μᵁ¹ᵣ[t]*(P_MT_up[1]-Pᴹ¹[t]+Pᴹ¹[t-1])==0) 
        @constraint(model_MP, Pᴹ¹[t-1]-Pᴹ¹[t]<=P_MT_dn[1])
        @constraint(model_MP, μᴰ¹ᵣ[t]*(P_MT_dn[1]-Pᴹ¹[t-1]+Pᴹ¹[t])==0) 
        @constraint(model_MP, Pᴹ²[t]-Pᴹ²[t-1]<=P_MT_up[2])                 # MT downwards/upwards rates constraints in VPP2, dual variable:  μᵁ²ᵣ and μᴰ²ᵣ
        @constraint(model_MP, μᵁ²ᵣ[t]*(P_MT_up[2]-Pᴹ²[t]+Pᴹ²[t-1])==0) 
        @constraint(model_MP, Pᴹ²[t-1]-Pᴹ²[t]<=P_MT_dn[2])
        @constraint(model_MP, μᴰ²ᵣ[t]*(P_MT_dn[2]-Pᴹ²[t-1]+Pᴹ²[t])==0) 
        @constraint(model_MP, Pᴹ³[t]-Pᴹ³[t-1]<=P_MT_up[3])                 # MT downwards/upwards rates constraints in VPP3, dual variable:  μᵁ³ᵣ and μᴰ³ᵣ
        @constraint(model_MP, μᵁ³ᵣ[t]*(P_MT_up[3]-Pᴹ³[t]+Pᴹ³[t-1])==0) 
        @constraint(model_MP, Pᴹ³[t-1]-Pᴹ³[t]<=P_MT_dn[3])
        @constraint(model_MP, μᴰ³ᵣ[t]*(P_MT_dn[3]-Pᴹ³[t-1]+Pᴹ³[t])==0) 
    end
    
    
    # Secondly, write    0<=yʲ.(Pᵗπʲ+Eᵗvʲ+wᵗ)>=0   primal variables time dual constraints
    # please consider the products of dual variables and coffecients of equations
        #  for Pᵇ¹, Pᵇ², Pᵇ³ in VPPs
        @constraint(model_MP,  λᵇ +v¹ -μᵇ¹ₘₐₓ + μᵇ¹ₘᵢₙ >= 0)        # for Pᵇ¹
        @constraint(model_MP, Pᵇ¹.* ( λᵇ +v¹ -μᵇ¹ₘₐₓ + μᵇ¹ₘᵢₙ )==0)
        @constraint(model_MP,  λᵇ +v² -μᵇ²ₘₐₓ + μᵇ²ₘᵢₙ >= 0)        # for Pᵇ²
        @constraint(model_MP, Pᵇ².* ( λᵇ +v² -μᵇ²ₘₐₓ + μᵇ²ₘᵢₙ )==0)
        @constraint(model_MP,  λᵇ +v³ -μᵇ³ₘₐₓ + μᵇ³ₘᵢₙ >= 0)        # for Pᵇ³
        @constraint(model_MP, Pᵇ³.* ( λᵇ +v³ -μᵇ³ₘₐₓ + μᵇ³ₘᵢₙ )==0)
    
        #  for Pˢ¹, Pˢ², Pˢ³ in VPPs
        @constraint(model_MP,  -λˢ -v¹ -μˢ¹ₘₐₓ + μˢ¹ₘᵢₙ >= 0)        # for Pˢ¹
        @constraint(model_MP, Pˢ¹.* ( -λˢ -v¹ -μˢ¹ₘₐₓ + μˢ¹ₘᵢₙ )==0)
        @constraint(model_MP,  -λˢ -v² -μˢ²ₘₐₓ + μˢ²ₘᵢₙ >= 0)        # for Pˢ²
        @constraint(model_MP, Pˢ².* ( -λˢ -v² -μˢ²ₘₐₓ + μˢ²ₘᵢₙ )==0)
        @constraint(model_MP,  -λˢ -v³ -μˢ³ₘₐₓ + μˢ³ₘᵢₙ >= 0)        # for Pˢ³
        @constraint(model_MP, Pˢ³.* ( -λˢ -v³ -μˢ³ₘₐₓ + μˢ³ₘᵢₙ )==0)
    
        # for Pᵂ¹, Pᵂ², Pᵂ³ in VPPs
        @constraint(model_MP,   v¹ -μᵂ¹ₘₐₓ + μᵂ¹ₘᵢₙ >= 0)           # for Pᵂ¹
        @constraint(model_MP, Pᵂ¹.* (v¹ -μᵂ¹ₘₐₓ + μᵂ¹ₘᵢₙ)==0)
        @constraint(model_MP,   v² -μᵂ²ₘₐₓ + μᵂ²ₘᵢₙ >= 0)           # for Pᵂ²
        @constraint(model_MP, Pᵂ².* (v² -μᵂ²ₘₐₓ + μᵂ²ₘᵢₙ)==0)
        @constraint(model_MP,   v³ -μᵂ³ₘₐₓ + μᵂ³ₘᵢₙ >= 0)           # for Pᵂ³
        @constraint(model_MP, Pᵂ³.* (v³ -μᵂ³ₘₐₓ + μᵂ³ₘᵢₙ)==0)
    
        # for Pᴹ¹, Pᴹ², Pᴹ³ in VPPs
        @constraint(model_MP, b_MT[1] +2*a_MT[1]*Pᴹ¹[1] +v¹[1] -μᴹ¹ₘₐₓ[1] +μᴹ¹ₘᵢₙ[1] >=0)    # for Pᴹ¹
        @constraint(model_MP, Pᴹ¹[1]*(b_MT[1] +2*a_MT[1]*Pᴹ¹[1] +v¹[1] -μᴹ¹ₘₐₓ[1] +μᴹ¹ₘᵢₙ[1])==0)
        @constraint(model_MP, b_MT[2] +2*a_MT[2]*Pᴹ²[1] +v²[1] -μᴹ²ₘₐₓ[1] +μᴹ²ₘᵢₙ[1] >=0)    # for Pᴹ²
        @constraint(model_MP, Pᴹ²[1]*(b_MT[2] +2*a_MT[2]*Pᴹ²[1] +v²[1] -μᴹ²ₘₐₓ[1] +μᴹ²ₘᵢₙ[1])==0)
        @constraint(model_MP, b_MT[3] +2*a_MT[3]*Pᴹ³[1] +v³[1] -μᴹ³ₘₐₓ[1] +μᴹ³ₘᵢₙ[1] >=0)    # for Pᴹ³
        @constraint(model_MP, Pᴹ³[1]*(b_MT[3] +2*a_MT[3]*Pᴹ³[1] +v³[1] -μᴹ³ₘₐₓ[1] +μᴹ³ₘᵢₙ[1])==0)
    for t in 2: T
        @constraint(model_MP, b_MT[1] +2*a_MT[1]*Pᴹ¹[t] +v¹[t] -μᴹ¹ₘₐₓ[t] +μᴹ¹ₘᵢₙ[t]   -μᵁ¹ᵣ[t] +μᵁ¹ᵣ[t-1] +μᴰ¹ᵣ[t] -μᴰ¹ᵣ[t-1]>=0)    # for Pᴹ¹
        @constraint(model_MP, Pᴹ¹[t]*(  b_MT[1] +2*a_MT[1]*Pᴹ¹[t] +v¹[t] -μᴹ¹ₘₐₓ[t] +μᴹ¹ₘᵢₙ[t]   -μᵁ¹ᵣ[t] +μᵁ¹ᵣ[t-1] +μᴰ¹ᵣ[t] -μᴰ¹ᵣ[t-1]  )==0)
        @constraint(model_MP, b_MT[2] +2*a_MT[2]*Pᴹ²[t] +v²[t] -μᴹ²ₘₐₓ[t] +μᴹ²ₘᵢₙ[t]   -μᵁ²ᵣ[t] +μᵁ²ᵣ[t-1] +μᴰ²ᵣ[t] -μᴰ²ᵣ[t-1]>=0)    # for Pᴹ²
        @constraint(model_MP, Pᴹ²[t]*(  b_MT[2] +2*a_MT[2]*Pᴹ²[t] +v²[t] -μᴹ²ₘₐₓ[t] +μᴹ²ₘᵢₙ[t]   -μᵁ²ᵣ[t] +μᵁ²ᵣ[t-1] +μᴰ²ᵣ[t] -μᴰ²ᵣ[t-1]  )==0)
        @constraint(model_MP, b_MT[3] +2*a_MT[3]*Pᴹ³[t] +v³[t] -μᴹ³ₘₐₓ[t] +μᴹ³ₘᵢₙ[t]   -μᵁ³ᵣ[t] +μᵁ³ᵣ[t-1] +μᴰ³ᵣ[t] -μᴰ³ᵣ[t-1]>=0)    # for Pᴹ³
        @constraint(model_MP, Pᴹ³[t]*(  b_MT[3] +2*a_MT[3]*Pᴹ³[t] +v³[t] -μᴹ³ₘₐₓ[t] +μᴹ³ₘᵢₙ[t]   -μᵁ³ᵣ[t] +μᵁ³ᵣ[t-1] +μᴰ³ᵣ[t] -μᴰ³ᵣ[t-1]  )==0)
    end
        
        # for Pᵇˢ¹, Pᵇˢ², Pᵇˢ³ in VPPs
        @constraint(model_MP,  2*λ_BS[1]*Pᵇˢ¹ +v¹ -w¹/BS_capacity[1] +μᵇˢ¹ₘᵢₙ -μᵇˢ¹ₘₐₓ   >= 0)   # for Pᵇˢ¹
        @constraint(model_MP, Pᵇˢ¹.* ( 2*λ_BS[1]*Pᵇˢ¹ +v¹ -w¹/BS_capacity[1] +μᵇˢ¹ₘᵢₙ -μᵇˢ¹ₘₐₓ )==0)
        @constraint(model_MP,  2*λ_BS[2]*Pᵇˢ² +v² -w²/BS_capacity[2] +μᵇˢ²ₘᵢₙ -μᵇˢ²ₘₐₓ   >= 0)   # for Pᵇˢ²
        @constraint(model_MP, Pᵇˢ².* ( 2*λ_BS[2]*Pᵇˢ² +v² -w²/BS_capacity[2] +μᵇˢ²ₘᵢₙ -μᵇˢ²ₘₐₓ )==0)
        @constraint(model_MP,  2*λ_BS[3]*Pᵇˢ³ +v³ -w³/BS_capacity[3] +μᵇˢ³ₘᵢₙ -μᵇˢ³ₘₐₓ   >= 0)   # for Pᵇˢ³
        @constraint(model_MP, Pᵇˢ³.* ( 2*λ_BS[3]*Pᵇˢ³ +v³ -w³/BS_capacity[3] +μᵇˢ³ₘᵢₙ -μᵇˢ³ₘₐₓ )==0)
    
        # for SoC¹, SoC², SoC³ in VPPs
        @constraint(model_MP, μˢᵒᶜ¹ₘᵢₙ[1] -μˢᵒᶜ¹ₘₐₓ[1] -w¹[1] >=0)     # for SoC¹
        @constraint(model_MP, SoC¹[1]*( μˢᵒᶜ¹ₘᵢₙ[1] -μˢᵒᶜ¹ₘₐₓ[1] -w¹[1] )==0)
        @constraint(model_MP, μˢᵒᶜ²ₘᵢₙ[1] -μˢᵒᶜ²ₘₐₓ[1] -w²[1] >=0)     # for SoC²
        @constraint(model_MP, SoC²[1]*( μˢᵒᶜ²ₘᵢₙ[1] -μˢᵒᶜ²ₘₐₓ[1] -w²[1] )==0)
        @constraint(model_MP, μˢᵒᶜ³ₘᵢₙ[1] -μˢᵒᶜ³ₘₐₓ[1] -w³[1] >=0)     # for SoC³
        @constraint(model_MP, SoC³[1]*( μˢᵒᶜ³ₘᵢₙ[1] -μˢᵒᶜ³ₘₐₓ[1] -w³[1] )==0)
    for t in 2: T
        @constraint(model_MP, μˢᵒᶜ¹ₘᵢₙ[t] -μˢᵒᶜ¹ₘₐₓ[t] -w¹[t] +w¹[t-1] >=0)     # for SoC¹
        @constraint(model_MP, SoC¹[t]*( μˢᵒᶜ¹ₘᵢₙ[t] -μˢᵒᶜ¹ₘₐₓ[t] -w¹[t] +w¹[t-1])==0)
        @constraint(model_MP, μˢᵒᶜ²ₘᵢₙ[t] -μˢᵒᶜ²ₘₐₓ[t] -w²[t] +w²[t-1] >=0)     # for SoC²
        @constraint(model_MP, SoC²[t]*( μˢᵒᶜ²ₘᵢₙ[t] -μˢᵒᶜ²ₘₐₓ[t] -w²[t] +w²[t-1])==0)
        @constraint(model_MP, μˢᵒᶜ³ₘᵢₙ[t] -μˢᵒᶜ³ₘₐₓ[t] -w³[t] +w³[t-1] >=0)     # for SoC³
        @constraint(model_MP, SoC³[t]*( μˢᵒᶜ³ₘᵢₙ[t] -μˢᵒᶜ³ₘₐₓ[t] -w³[t] +w³[t-1] )==0)
    end
        
       
    # Primal variables and constraints for upper level, also in the MP optimization (no dual variables)
        @variable(model_MP, Pᴰˢᴼᵇ[1:T])       # Quantities DSO trade with Wholesale market
        @variable(model_MP, Pᴰˢᴼˢ[1:T]) 
        @variable(model_MP, N₁[1:T],Bin)      # Binary variables for logical judgement
        @variable(model_MP, N₂[1:T],Bin) 
     
        @constraint(model_MP, λᵇ <= λ_PMb)    # Upper and Lower bounds for prices
        @constraint(model_MP, λˢ <= λ_PMb)
        @constraint(model_MP, λ_PMs <= λˢ)
        @constraint(model_MP, λ_PMs <= λᵇ)
    
        @constraint(model_MP, Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰<=M₂*N₂)      # big-M formulation of logical judgement 2
        @constraint(model_MP, -M₂*(ones(1,T)'-N₂).<=Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰)
        @constraint(model_MP, -M₂*N₂.<=Pᴰˢᴼˢ+(Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰))
        @constraint(model_MP, Pᴰˢᴼˢ+(Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰).<=M₂*N₂)
        @constraint(model_MP, -M₂*(ones(1,T)'-N₂).<=Pᴰˢᴼˢ)
        @constraint(model_MP, Pᴰˢᴼˢ.<=M₂*(ones(1,T)'-N₂))
     
        @constraint(model_MP, Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰<=M₁*N₁)      # big-M formulation of logical judgement 1 
        @constraint(model_MP, -M₁*(ones(1,T)'-N₁).<=Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰)
        @constraint(model_MP, -M₁*(ones(1,T)'-N₁).<=Pᴰˢᴼᵇ-(Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰))
        @constraint(model_MP, Pᴰˢᴼᵇ-(Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰).<=M₁*(ones(1,T)'-N₁))
        @constraint(model_MP, -M₁*N₁.<=Pᴰˢᴼᵇ)
        @constraint(model_MP, Pᴰˢᴼᵇ.<=M₁*N₁)
    
    # Constrain the upper bound of LL obj with real solution 
    LL_obj_VPP1_real=sum(λᵇ.*Pᵇ¹⁰ -λˢ.*Pˢ¹⁰ +a_MT[1]*Pᴹ¹⁰.*Pᴹ¹⁰+b_MT[1]*Pᴹ¹⁰ +c_MT[1]*u¹⁰ +λ_BS[1]*Pᵇˢ¹⁰.*Pᵇˢ¹⁰)       # Obj for VPP1    
    LL_obj_VPP2_real=sum(λᵇ.*Pᵇ²⁰ -λˢ.*Pˢ²⁰ +a_MT[2]*Pᴹ²⁰.*Pᴹ²⁰+b_MT[2]*Pᴹ²⁰ +c_MT[2]*u²⁰ +λ_BS[2]*Pᵇˢ²⁰.*Pᵇˢ²⁰)       # Obj for VPP2 
    LL_obj_VPP3_real=sum(λᵇ.*Pᵇ³⁰ -λˢ.*Pˢ³⁰ +a_MT[3]*Pᴹ³⁰.*Pᴹ³⁰+b_MT[3]*Pᴹ³⁰ +c_MT[3]*u³⁰ +λ_BS[3]*Pᵇˢ³⁰.*Pᵇˢ³⁰)       # Obj for VPP3
    LL_obj_real=LL_obj_VPP1_real+LL_obj_VPP2_real+LL_obj_VPP3_real              # Obj for LL program
    
    LL_obj_VPP1_cons=sum(λᵇ.*Pᵇ¹ -λˢ.*Pˢ¹ +a_MT[1]*Pᴹ¹.*Pᴹ¹+b_MT[1]*Pᴹ¹ +c_MT[1]*u¹ +λ_BS[1]*Pᵇˢ¹.*Pᵇˢ¹)       
    LL_obj_VPP2_cons=sum(λᵇ.*Pᵇ² -λˢ.*Pˢ² +a_MT[2]*Pᴹ².*Pᴹ²+b_MT[2]*Pᴹ² +c_MT[2]*u² +λ_BS[2]*Pᵇˢ².*Pᵇˢ²)       
    LL_obj_VPP3_cons=sum(λᵇ.*Pᵇ³ -λˢ.*Pˢ³ +a_MT[3]*Pᴹ³.*Pᴹ³+b_MT[3]*Pᴹ³ +c_MT[3]*u³ +λ_BS[3]*Pᵇˢ³.*Pᵇˢ³)       
    LL_obj_cons=LL_obj_VPP1_cons+LL_obj_VPP2_cons+LL_obj_VPP3_cons             
             
    @constraint(model_MP, LL_obj_real <= LL_obj_cons)        # The upper bound for LL program
    
    # Obj of upper-level program
        @objective(model_MP, Max,  sum(λ_PMs.*Pᴰˢᴼˢ)-sum(λ_PMb.*Pᴰˢᴼᵇ) + sum(λᵇ.*(Pᵇ¹⁰+Pᵇ²⁰+Pᵇ³⁰))-sum(λˢ.*(Pˢ¹⁰+Pˢ²⁰+Pˢ³⁰)))  
        set_optimizer(model_MP, Gurobi.Optimizer)
        # set_optimizer_attribute(model_MP, "NonConvex", -1)
        optimize!(model_MP)
    
        UB=min(UB, JuMP.value.( sum(λ_PMs.*Pᴰˢᴼˢ)-sum(λ_PMb.*Pᴰˢᴼᵇ) + sum(λᵇ.*(Pᵇ¹⁰+Pᵇ²⁰+Pᵇ³⁰))-sum(λˢ.*(Pˢ¹⁰+Pˢ²⁰+Pˢ³⁰))  ) )
        λᵇ=JuMP.value.(λᵇ)
        λˢ=JuMP.value.(λˢ)
    
      
    return  UB,  λᵇ, λˢ
    
    
    
    
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------------
    
    
    else # elseif u¹(binary variables are not empty)
    
        #----------------------define model----------------
        model_MP= Model()
    
        #-------------------------body-------------
        @variable(model_MP, λᵇ[1:T])          # Prices set by DSO
        @variable(model_MP, λˢ[1:T])
        # Dual variables in lower level-------------------------------
            @variable(model_MP, μᵇ¹ₘₐₓ[1:T]>=0)    # dual variables for buying power from DSO
            @variable(model_MP, μᵇ¹ₘᵢₙ[1:T]>=0)  
            @variable(model_MP, μᵇ²ₘₐₓ[1:T]>=0)
            @variable(model_MP, μᵇ²ₘᵢₙ[1:T]>=0)  
            @variable(model_MP, μᵇ³ₘₐₓ[1:T]>=0)
            @variable(model_MP, μᵇ³ₘᵢₙ[1:T]>=0)  
        
            @variable(model_MP, μˢ¹ₘₐₓ[1:T]>=0)   # dual variables for selling power to DSO
            @variable(model_MP, μˢ¹ₘᵢₙ[1:T]>=0) 
            @variable(model_MP, μˢ²ₘₐₓ[1:T]>=0) 
            @variable(model_MP, μˢ²ₘᵢₙ[1:T]>=0)
            @variable(model_MP, μˢ³ₘₐₓ[1:T]>=0) 
            @variable(model_MP, μˢ³ₘᵢₙ[1:T]>=0)
        
            @variable(model_MP, μᵂ¹ₘₐₓ[1:T]>=0)  # dual variables for output bounds of WT
            @variable(model_MP, μᵂ¹ₘᵢₙ[1:T]>=0) 
            @variable(model_MP, μᵂ²ₘₐₓ[1:T]>=0) 
            @variable(model_MP, μᵂ²ₘᵢₙ[1:T]>=0)
            @variable(model_MP, μᵂ³ₘₐₓ[1:T]>=0) 
            @variable(model_MP, μᵂ³ₘᵢₙ[1:T]>=0)
        
            @variable(model_MP, μᵇˢ¹ₘₐₓ[1:T]>=0)    # dual variables for output bounds of BS
            @variable(model_MP, μᵇˢ¹ₘᵢₙ[1:T]>=0)
            @variable(model_MP, μᵇˢ²ₘₐₓ[1:T]>=0) 
            @variable(model_MP, μᵇˢ²ₘᵢₙ[1:T]>=0)
            @variable(model_MP, μᵇˢ³ₘₐₓ[1:T]>=0) 
            @variable(model_MP, μᵇˢ³ₘᵢₙ[1:T]>=0)
        
            @variable(model_MP, μˢᵒᶜ¹ₘₐₓ[1:T]>=0)   # dual variables for bounds of SoC
            @variable(model_MP, μˢᵒᶜ¹ₘᵢₙ[1:T]>=0)
            @variable(model_MP, μˢᵒᶜ²ₘₐₓ[1:T]>=0) 
            @variable(model_MP, μˢᵒᶜ²ₘᵢₙ[1:T]>=0)
            @variable(model_MP, μˢᵒᶜ³ₘₐₓ[1:T]>=0) 
            @variable(model_MP, μˢᵒᶜ³ₘᵢₙ[1:T]>=0)
        
            @variable(model_MP, μᴹ¹ₘₐₓ[1:T]>=0)     # dual variables for output bounds of MT
            @variable(model_MP, μᴹ¹ₘᵢₙ[1:T]>=0) 
            @variable(model_MP, μᴹ²ₘₐₓ[1:T]>=0) 
            @variable(model_MP, μᴹ²ₘᵢₙ[1:T]>=0)
            @variable(model_MP, μᴹ³ₘₐₓ[1:T]>=0) 
            @variable(model_MP, μᴹ³ₘᵢₙ[1:T]>=0)
        
            @variable(model_MP, μᵁ¹ᵣ[1:T]>=0)      # dual variables for upwards/downwards rates of MT
            @variable(model_MP, μᴰ¹ᵣ[1:T]>=0)       
            @variable(model_MP, μᵁ²ᵣ[1:T]>=0)       
            @variable(model_MP, μᴰ²ᵣ[1:T]>=0)       
            @variable(model_MP, μᵁ³ᵣ[1:T]>=0)       
            @variable(model_MP, μᴰ³ᵣ[1:T]>=0)        
        
            @variable(model_MP, v¹[1:T])        # for power balance equation
            @variable(model_MP, v²[1:T])
            @variable(model_MP, v³[1:T])
        
            @variable(model_MP, w¹[1:T])        # variables for relationship between BS output and SoC
            @variable(model_MP, w²[1:T])         
            @variable(model_MP, w³[1:T])       
        
         # Primal variables in lower level, existed as a bound for real solution-------------------------------
         # Primal variables for VPP¹
            @variable(model_MP, Pᵇ¹[1:T])
            @variable(model_MP, Pˢ¹[1:T])
            @variable(model_MP, Pᵂ¹[1:T])
            @variable(model_MP, Pᴹ¹[1:T])
            @variable(model_MP, Pᵇˢ¹[1:T])
            @variable(model_MP, SoC¹[1:T])
            @variable(model_MP, Pⱽᴾᴾ¹[1:T]) 
        
        # Primal variables for VPP²
            @variable(model_MP, Pᵇ²[1:T])
            @variable(model_MP, Pˢ²[1:T])
            @variable(model_MP, Pᵂ²[1:T])
            @variable(model_MP, Pᴹ²[1:T])
            @variable(model_MP, Pᵇˢ²[1:T])
            @variable(model_MP, SoC²[1:T])
            @variable(model_MP, Pⱽᴾᴾ²[1:T]) 
            
        # Primal variables for VPP³
            @variable(model_MP, Pᵇ³[1:T])
            @variable(model_MP, Pˢ³[1:T])
            @variable(model_MP, Pᵂ³[1:T])
            @variable(model_MP, Pᴹ³[1:T])
            @variable(model_MP, Pᵇˢ³[1:T])
            @variable(model_MP, SoC³[1:T])
            @variable(model_MP, Pⱽᴾᴾ³[1:T]) 
        
        # Primal variables in lower level, existed as real solution-------------------------------
            # for VPP¹
            @variable(model_MP, Pᵇ¹⁰[1:T])
            @variable(model_MP, Pˢ¹⁰[1:T])
            @variable(model_MP, Pᵂ¹⁰[1:T])
            @variable(model_MP, Pᴹ¹⁰[1:T])
            @variable(model_MP, Pᵇˢ¹⁰[1:T])
            @variable(model_MP, SoC¹⁰[1:T])
            @variable(model_MP, Pⱽᴾᴾ¹⁰[1:T]) 
            @variable(model_MP, u¹⁰[1:T],Bin) 
        
            # for VPP²
            @variable(model_MP, Pᵇ²⁰[1:T])
            @variable(model_MP, Pˢ²⁰[1:T])
            @variable(model_MP, Pᵂ²⁰[1:T])
            @variable(model_MP, Pᴹ²⁰[1:T])
            @variable(model_MP, Pᵇˢ²⁰[1:T])
            @variable(model_MP, SoC²⁰[1:T])
            @variable(model_MP, Pⱽᴾᴾ²⁰[1:T]) 
            @variable(model_MP, u²⁰[1:T],Bin) 
        
            # for VPP³
            @variable(model_MP, Pᵇ³⁰[1:T])
            @variable(model_MP, Pˢ³⁰[1:T])
            @variable(model_MP, Pᵂ³⁰[1:T])
            @variable(model_MP, Pᴹ³⁰[1:T])
            @variable(model_MP, Pᵇˢ³⁰[1:T])
            @variable(model_MP, SoC³⁰[1:T])
            @variable(model_MP, Pⱽᴾᴾ³⁰[1:T]) 
            @variable(model_MP, u³⁰[1:T],Bin) 
        
        
        # Write constraints for real solution in LL program, in this part, there is no dual variables, can be seen as nomal optimization problem
        @constraint(model_MP, Pᵇ¹⁰-Pˢ¹⁰==Pⱽᴾᴾ¹⁰)                     
        @constraint(model_MP, Pᵇ²⁰-Pˢ²⁰==Pⱽᴾᴾ²⁰)                  
        @constraint(model_MP, Pᵇ³⁰-Pˢ³⁰==Pⱽᴾᴾ³⁰)           
        @constraint(model_MP, Pⱽᴾᴾ¹⁰+Pᵂ¹⁰+Pᴹ¹⁰+Pᵇˢ¹⁰==load_1)          # Power balance for VPP¹,   dual variable: v¹
        @constraint(model_MP, Pⱽᴾᴾ²⁰+Pᵂ²⁰+Pᴹ²⁰+Pᵇˢ²⁰==load_2)          # Power balance for VPP²,   dual variable: v² 
        @constraint(model_MP, Pⱽᴾᴾ³⁰+Pᵂ³⁰+Pᴹ³⁰+Pᵇˢ³⁰==load_3)          # Power balance for VPP³,   dual variable: v³
        
        for i in 1:T                                                   
            @constraint(model_MP, trade_max[1]-Pᵇ¹⁰[i]>=0)             # Lower and Upper bounds for trading volumes (buy) of VPPs
            @constraint(model_MP, Pᵇ¹⁰[i]-trade_min[1]>=0 )            
            @constraint(model_MP, trade_max[1]-Pᵇ²⁰[i]>=0)       
            @constraint(model_MP, Pᵇ²⁰[i]-trade_min[1]>=0 )                   
            @constraint(model_MP, trade_max[1]-Pᵇ³⁰[i]>=0)       
            @constraint(model_MP, Pᵇ³⁰[i]-trade_min[1]>=0 )                   
        
            @constraint(model_MP, trade_max[1]-Pˢ¹⁰[i]>=0)             # Lower and Upper bounds for trading volumes (sell) of VPPs
            @constraint(model_MP, Pˢ¹⁰[i]-trade_min[1]>=0 )            
            @constraint(model_MP, trade_max[1]-Pˢ²⁰[i]>=0)       
            @constraint(model_MP, Pˢ²⁰[i]-trade_min[1]>=0 )                   
            @constraint(model_MP, trade_max[1]-Pˢ³⁰[i]>=0)       
            @constraint(model_MP, Pˢ³⁰[i]-trade_min[1]>=0 )
        
            @constraint(model_MP, P_Wmax_1[i]-Pᵂ¹⁰[i]>=0 )             # Lower and Upper bounds for output of WT
            @constraint(model_MP, Pᵂ¹⁰[i]>=0 )
            @constraint(model_MP, P_Wmax_2[i]-Pᵂ²⁰[i]>=0 )              
            @constraint(model_MP, Pᵂ²⁰[i]>=0 )
            @constraint(model_MP, P_Wmax_3[i]-Pᵂ³⁰[i]>=0 )              
            @constraint(model_MP, Pᵂ³⁰[i]>=0 )
            
            @constraint(model_MP, P_MT_max[1]*u¹⁰[i]-Pᴹ¹⁰[i]>=0 )      # Lower and Upper bounds for output of MT with UC
            @constraint(model_MP, Pᴹ¹⁰[i]-P_MT_min[1]*u¹⁰[i]>=0 )
            @constraint(model_MP, P_MT_max[2]*u²⁰[i]-Pᴹ²⁰[i]>=0 )          
            @constraint(model_MP, Pᴹ²⁰[i]-P_MT_min[2]*u²⁰[i]>=0 )
            @constraint(model_MP, P_MT_max[3]*u³⁰[i]-Pᴹ³⁰[i]>=0 )              
            @constraint(model_MP, Pᴹ³⁰[i]-P_MT_min[3]*u³⁰[i]>=0 )
        
            @constraint(model_MP, Pᵇˢ¹⁰[i]>=-P_BS_max[1])              # BS output constraints               
            @constraint(model_MP, Pᵇˢ¹⁰[i]<=P_BS_max[1])
            @constraint(model_MP, Pᵇˢ²⁰[i]>=-P_BS_max[2])                     
            @constraint(model_MP, Pᵇˢ²⁰[i]<=P_BS_max[2])
            @constraint(model_MP, Pᵇˢ³⁰[i]>=-P_BS_max[3])                     
            @constraint(model_MP, Pᵇˢ³⁰[i]<=P_BS_max[3])
        
            @constraint(model_MP, SOC_min[1]<=SoC¹⁰[i])                # Lower and Upper bounds for SoC 
            @constraint(model_MP, SoC¹⁰[i]<=SOC_max[1])   
            @constraint(model_MP, SOC_min[2]<=SoC²⁰[i])                 
            @constraint(model_MP, SoC²⁰[i]<=SOC_max[2]) 
            @constraint(model_MP, SOC_min[3]<=SoC³⁰[i])                 
            @constraint(model_MP, SoC³⁰[i]<=SOC_max[3])                      
        
        end
        
        @constraint(model_MP, SoC¹⁰[1]==SOC_initial[1]-Pᵇˢ¹⁰[1]/BS_capacity[1])         # SOC initial capacity constraints
        @constraint(model_MP, SoC²⁰[1]==SOC_initial[2]-Pᵇˢ²⁰[1]/BS_capacity[2])   
        @constraint(model_MP, SoC³⁰[1]==SOC_initial[3]-Pᵇˢ³⁰[1]/BS_capacity[3])   
            
        for t in 2:T
            @constraint(model_MP, SoC¹⁰[t]==SoC¹⁰[t-1]-Pᵇˢ¹⁰[t]/BS_capacity[1])          # Relationship betwwen SOC and Pᵇˢ⁰
            @constraint(model_MP, SoC²⁰[t]==SoC²⁰[t-1]-Pᵇˢ²⁰[t]/BS_capacity[2])   
            @constraint(model_MP, SoC³⁰[t]==SoC³⁰[t-1]-Pᵇˢ³⁰[t]/BS_capacity[3]) 
            
            @constraint(model_MP, Pᴹ¹⁰[t]-Pᴹ¹⁰[t-1]<=P_MT_up[1])                         # MT downwards/upwards rates constraints 
            @constraint(model_MP, Pᴹ¹⁰[t-1]-Pᴹ¹⁰[t]<=P_MT_dn[1])
            @constraint(model_MP, Pᴹ²⁰[t]-Pᴹ²⁰[t-1]<=P_MT_up[2])
            @constraint(model_MP, Pᴹ²⁰[t-1]-Pᴹ²⁰[t]<=P_MT_dn[2])
            @constraint(model_MP, Pᴹ³⁰[t]-Pᴹ³⁰[t-1]<=P_MT_up[3])
            @constraint(model_MP, Pᴹ³⁰[t-1]-Pᴹ³⁰[t]<=P_MT_dn[3])
        end
        
        
        # Complmentarity Constraints (CC) in lower level-------------------------------
        # 0<=yʲ.(Pᵗπʲ-wᵗ)>=0           primal variables time dual constraints
        # 0<=πʲ.(R-Kx-Nzʲ-Pyʲ)>=0      dual variables time primal constraints
                 
        @constraint(model_MP, Pᵇ¹-Pˢ¹+Pᵂ¹+Pᴹ¹+Pᵇˢ¹==load_1)          # Power balance for VPP¹,   dual variable: v¹
        @constraint(model_MP, Pᵇ²-Pˢ²+Pᵂ²+Pᴹ²+Pᵇˢ²==load_2)          # Power balance for VPP²,   dual variable: v² 
        @constraint(model_MP, Pᵇ³-Pˢ³+Pᵂ³+Pᴹ³+Pᵇˢ³==load_3)          # Power balance for VPP³,   dual variable: v³
        
        # Firstly write     0<=πʲ.(R-Kx-Nzʲ-Pyʲ)>=0     dual variables time primal constraints
        for i in 1:T
            @constraint(model_MP, μᵇ¹ₘₐₓ[i]*(trade_max[1]-Pᵇ¹[i])==0)  # Bounds (buy) for trading volume of VPP1: Pᵇ¹,  dual variable: μᵇ¹ₘₐₓ and μᵇ¹ₘᵢₙ
            @constraint(model_MP, trade_max[1]-Pᵇ¹[i]>=0 )
            @constraint(model_MP, μᵇ¹ₘᵢₙ[i]* (Pᵇ¹[i]-trade_min[1])==0 )    
            @constraint(model_MP, Pᵇ¹[i]-trade_min[1]>=0 )
            @constraint(model_MP, μᵇ²ₘₐₓ[i]*(trade_max[2]-Pᵇ²[i])==0)  # Bounds (buy) for trading volume of VPP2: Pᵇ²,  dual variable: μᵇ²ₘₐₓ and μᵇ²ₘᵢₙ
            @constraint(model_MP, trade_max[2]-Pᵇ²[i]>=0 )
            @constraint(model_MP, μᵇ²ₘᵢₙ[i]* (Pᵇ²[i]-trade_min[2])==0 )    
            @constraint(model_MP, Pᵇ²[i]-trade_min[2]>=0 )
            @constraint(model_MP, μᵇ³ₘₐₓ[i]*(trade_max[3]-Pᵇ³[i])==0)  # Bounds (buy) for trading volume of VPP3: Pᵇ³,  dual variable: μᵇ³ₘₐₓ and μᵇ³ₘᵢₙ
            @constraint(model_MP, trade_max[3]-Pᵇ³[i]>=0 )
            @constraint(model_MP, μᵇ³ₘᵢₙ[i]* (Pᵇ³[i]-trade_min[3])==0 )    
            @constraint(model_MP, Pᵇ³[i]-trade_min[3]>=0 )
            
            @constraint(model_MP, μˢ¹ₘₐₓ[i]*(trade_max[1]-Pˢ¹[i])==0)  # Bounds (sell) for trading volume of VPP1: Pˢ¹,  dual variable: μˢ¹ₘₐₓ and μˢ¹ₘᵢₙ
            @constraint(model_MP, trade_max[1]-Pˢ¹[i]>=0)
            @constraint(model_MP, μˢ¹ₘᵢₙ[i]* (Pˢ¹[i]-trade_min[1])==0)    
            @constraint(model_MP, Pˢ¹[i]-trade_min[1]>=0)
            @constraint(model_MP, μˢ²ₘₐₓ[i]*(trade_max[2]-Pˢ²[i])==0)  # Bounds (sell) for trading volume of VPP2: Pˢ²,  dual variable: μˢ²ₘₐₓ and μˢ²ₘᵢₙ
            @constraint(model_MP, trade_max[2]-Pˢ²[i]>=0)
            @constraint(model_MP, μˢ²ₘᵢₙ[i]* (Pˢ²[i]-trade_min[2])==0)    
            @constraint(model_MP, Pˢ²[i]-trade_min[2]>=0)
            @constraint(model_MP, μˢ³ₘₐₓ[i]*(trade_max[3]-Pˢ³[i])==0)  # Bounds (sell) for trading volume of VPP3: Pˢ³,  dual variable: μˢ³ₘₐₓ and μˢ³ₘᵢₙ
            @constraint(model_MP, trade_max[3]-Pˢ³[i]>=0)
            @constraint(model_MP, μˢ³ₘᵢₙ[i]* (Pˢ³[i]-trade_min[3])==0)    
            @constraint(model_MP, Pˢ³[i]-trade_min[3]>=0)
        
            @constraint(model_MP, μᵂ¹ₘₐₓ[i]* (P_Wmax_1[i]-Pᵂ¹[i])==0)    # Bounds for output of WT in VPP1: Pᵂ¹, dual variable: μᵂ¹ₘₐₓ and μᵂ¹ₘᵢₙ
            @constraint(model_MP, P_Wmax_1[i]-Pᵂ¹[i]>=0)
            @constraint(model_MP, μᵂ¹ₘᵢₙ[i]* (Pᵂ¹[i])==0)                        
            @constraint(model_MP, Pᵂ¹[i]>=0)
            @constraint(model_MP, μᵂ²ₘₐₓ[i]* (P_Wmax_2[i]-Pᵂ²[i])==0)    # Bounds for output of WT in VPP2: Pᵂ², dual variable: μᵂ²ₘₐₓ and μᵂ²ₘᵢₙ
            @constraint(model_MP, P_Wmax_2[i]-Pᵂ²[i]>=0)
            @constraint(model_MP, μᵂ²ₘᵢₙ[i]* (Pᵂ²[i])==0)                        
            @constraint(model_MP, Pᵂ²[i]>=0)
            @constraint(model_MP, μᵂ³ₘₐₓ[i]* (P_Wmax_3[i]-Pᵂ³[i])==0)    # Bounds for output of WT in VPP2: Pᵂ³, dual variable: μᵂ³ₘₐₓ and μᵂ³ₘᵢₙ
            @constraint(model_MP, P_Wmax_3[i]-Pᵂ³[i]>=0)
            @constraint(model_MP, μᵂ³ₘᵢₙ[i]* (Pᵂ³[i])==0)                        
            @constraint(model_MP, Pᵂ³[i]>=0)
        
            @constraint(model_MP, μᴹ¹ₘₐₓ[i]* (P_MT_max[1]*u¹[i]-Pᴹ¹[i])==0 )  # Bounds for output of MT in VPP1, with UC: Pᴹ¹, dual variable: μᴹ¹ₘₐₓ and μᴹ¹ₘᵢₙ
            @constraint(model_MP, P_MT_max[1]*u¹[i]-Pᴹ¹[i]>=0 )
            @constraint(model_MP, μᴹ¹ₘᵢₙ[i]* (Pᴹ¹[i]-P_MT_min[1]*u¹[i])==0 )       
            @constraint(model_MP, Pᴹ¹[i]-P_MT_min[1]*u¹[i]>=0 )
            @constraint(model_MP, μᴹ²ₘₐₓ[i]* (P_MT_max[2]*u²[i]-Pᴹ²[i])==0 )  # Bounds for output of MT in VPP2, with UC: Pᴹ², dual variable: μᴹ²ₘₐₓ and μᴹ²ₘᵢₙ
            @constraint(model_MP, P_MT_max[2]*u²[i]-Pᴹ²[i]>=0 )
            @constraint(model_MP, μᴹ²ₘᵢₙ[i]* (Pᴹ²[i]-P_MT_min[2]*u²[i])==0 )       
            @constraint(model_MP, Pᴹ²[i]-P_MT_min[2]*u²[i]>=0 )
            @constraint(model_MP, μᴹ³ₘₐₓ[i]* (P_MT_max[3]*u³[i]-Pᴹ³[i])==0 )  # Bounds for output of MT in VPP3, with UC: Pᴹ³, dual variable: μᴹ³ₘₐₓ and μᴹ³ₘᵢₙ
            @constraint(model_MP, P_MT_max[3]*u³[i]-Pᴹ³[i]>=0 )
            @constraint(model_MP, μᴹ³ₘᵢₙ[i]* (Pᴹ³[i]-P_MT_min[3]*u³[i])==0 )       
            @constraint(model_MP, Pᴹ³[i]-P_MT_min[3]*u³[i]>=0 )
        
            @constraint(model_MP, μᵇˢ¹ₘₐₓ[i]* (P_BS_max[1]-Pᵇˢ¹[i])==0 )       # Bounds for output of BS in VPP1: Pᵇˢ¹, dual variable: μᵇˢ¹ₘₐₓ and μᵇˢ¹ₘᵢₙ
            @constraint(model_MP, P_BS_max[1]-Pᵇˢ¹[i]>=0 )
            @constraint(model_MP, μᵇˢ¹ₘᵢₙ[i]* (Pᵇˢ¹[i]+P_BS_max[1])==0 )  
            @constraint(model_MP, Pᵇˢ¹[i]+P_BS_max[1]>=0 )
            @constraint(model_MP, μᵇˢ²ₘₐₓ[i]* (P_BS_max[2]-Pᵇˢ²[i])==0 )       # Bounds for output of BS in VPP2: Pᵇˢ², dual variable: μᵇˢ²ₘₐₓ and μᵇˢ²ₘᵢₙ
            @constraint(model_MP, P_BS_max[2]-Pᵇˢ²[i]>=0 )
            @constraint(model_MP, μᵇˢ²ₘᵢₙ[i]* (Pᵇˢ²[i]+P_BS_max[2])==0 )  
            @constraint(model_MP, Pᵇˢ²[i]+P_BS_max[2]>=0 )
            @constraint(model_MP, μᵇˢ³ₘₐₓ[i]* (P_BS_max[3]-Pᵇˢ³[i])==0 )       # Bounds for output of BS in VPP3: Pᵇˢ³, dual variable: μᵇˢ³ₘₐₓ and μᵇˢ³ₘᵢₙ
            @constraint(model_MP, P_BS_max[3]-Pᵇˢ³[i]>=0 )
            @constraint(model_MP, μᵇˢ³ₘᵢₙ[i]* (Pᵇˢ³[i]+P_BS_max[3])==0 )  
            @constraint(model_MP, Pᵇˢ³[i]+P_BS_max[3]>=0 )
        
            @constraint(model_MP, SoC¹[i]<=SOC_max[1])                         # Bounds for SoC in VPP1: SoC¹, dual variable:  μˢᵒᶜ¹ₘₐₓ and  μˢᵒᶜ¹ₘᵢₙ
            @constraint(model_MP, μˢᵒᶜ¹ₘₐₓ[i]*(SOC_max[1]-SoC¹[i])==0)
            @constraint(model_MP, SOC_min[1]<=SoC¹[i])   
            @constraint(model_MP, μˢᵒᶜ¹ₘᵢₙ[i]*(SoC¹[i]-SOC_min[1])==0)                       
            @constraint(model_MP, SoC²[i]<=SOC_max[2])                         # Bounds for SoC in VPP2: SoC², dual variable:  μˢᵒᶜ²ₘₐₓ and  μˢᵒᶜ²ₘᵢₙ
            @constraint(model_MP, μˢᵒᶜ²ₘₐₓ[i]*(SOC_max[2]-SoC²[i])==0)
            @constraint(model_MP, SOC_min[2]<=SoC²[i])   
            @constraint(model_MP, μˢᵒᶜ²ₘᵢₙ[i]*(SoC²[i]-SOC_min[2])==0)
            @constraint(model_MP, SoC³[i]<=SOC_max[3])                         # Bounds for SoC in VPP3: SoC³, dual variable:  μˢᵒᶜ³ₘₐₓ and  μˢᵒᶜ³ₘᵢₙ
            @constraint(model_MP, μˢᵒᶜ³ₘₐₓ[i]*(SOC_max[3]-SoC³[i])==0)
            @constraint(model_MP, SOC_min[3]<=SoC³[i])   
            @constraint(model_MP, μˢᵒᶜ³ₘᵢₙ[i]*(SoC³[i]-SOC_min[3])==0)     
        
        end
        
        @constraint(model_MP, SoC¹[1]==SOC_initial[1]-Pᵇˢ¹[1]/BS_capacity[1])    # SOC initial capacity constraints in VPP1, dual variable:  w¹ 
        @constraint(model_MP, SoC²[1]==SOC_initial[2]-Pᵇˢ²[1]/BS_capacity[2])    # SOC initial capacity constraints in VPP2, dual variable:  w² 
        @constraint(model_MP, SoC³[1]==SOC_initial[3]-Pᵇˢ³[1]/BS_capacity[3])    # SOC initial capacity constraints in VPP3, dual variable:  w³ 
            
        for t in 2:T
            @constraint(model_MP, SoC¹[t]==SoC¹[t-1]-Pᵇˢ¹[t]/BS_capacity[1])     # Relationship betwwen SOC and Pᵇˢ⁰, dual variable:  w¹
            @constraint(model_MP, SoC²[t]==SoC²[t-1]-Pᵇˢ²[t]/BS_capacity[2])     # Relationship betwwen SOC and Pᵇˢ⁰, dual variable:  w²
            @constraint(model_MP, SoC³[t]==SoC³[t-1]-Pᵇˢ³[t]/BS_capacity[3])     # Relationship betwwen SOC and Pᵇˢ⁰, dual variable:  w³
            
            @constraint(model_MP, Pᴹ¹[t]-Pᴹ¹[t-1]<=P_MT_up[1])                 # MT downwards/upwards rates constraints in VPP1, dual variable:  μᵁ¹ᵣ and μᴰ¹ᵣ
            @constraint(model_MP, μᵁ¹ᵣ[t]*(P_MT_up[1]-Pᴹ¹[t]+Pᴹ¹[t-1])==0) 
            @constraint(model_MP, Pᴹ¹[t-1]-Pᴹ¹[t]<=P_MT_dn[1])
            @constraint(model_MP, μᴰ¹ᵣ[t]*(P_MT_dn[1]-Pᴹ¹[t-1]+Pᴹ¹[t])==0) 
            @constraint(model_MP, Pᴹ²[t]-Pᴹ²[t-1]<=P_MT_up[2])                 # MT downwards/upwards rates constraints in VPP2, dual variable:  μᵁ²ᵣ and μᴰ²ᵣ
            @constraint(model_MP, μᵁ²ᵣ[t]*(P_MT_up[2]-Pᴹ²[t]+Pᴹ²[t-1])==0) 
            @constraint(model_MP, Pᴹ²[t-1]-Pᴹ²[t]<=P_MT_dn[2])
            @constraint(model_MP, μᴰ²ᵣ[t]*(P_MT_dn[2]-Pᴹ²[t-1]+Pᴹ²[t])==0) 
            @constraint(model_MP, Pᴹ³[t]-Pᴹ³[t-1]<=P_MT_up[3])                 # MT downwards/upwards rates constraints in VPP3, dual variable:  μᵁ³ᵣ and μᴰ³ᵣ
            @constraint(model_MP, μᵁ³ᵣ[t]*(P_MT_up[3]-Pᴹ³[t]+Pᴹ³[t-1])==0) 
            @constraint(model_MP, Pᴹ³[t-1]-Pᴹ³[t]<=P_MT_dn[3])
            @constraint(model_MP, μᴰ³ᵣ[t]*(P_MT_dn[3]-Pᴹ³[t-1]+Pᴹ³[t])==0) 
        end
        
        
        # Secondly, write    0<=yʲ.(Pᵗπʲ+Eᵗvʲ+wᵗ)>=0   primal variables time dual constraints
        # please consider the products of dual variables and coffecients of equations
            #  for Pᵇ¹, Pᵇ², Pᵇ³ in VPPs
            @constraint(model_MP,  λᵇ +v¹ -μᵇ¹ₘₐₓ + μᵇ¹ₘᵢₙ >= 0)        # for Pᵇ¹
            @constraint(model_MP, Pᵇ¹.* ( λᵇ +v¹ -μᵇ¹ₘₐₓ + μᵇ¹ₘᵢₙ )==0)
            @constraint(model_MP,  λᵇ +v² -μᵇ²ₘₐₓ + μᵇ²ₘᵢₙ >= 0)        # for Pᵇ²
            @constraint(model_MP, Pᵇ².* ( λᵇ +v² -μᵇ²ₘₐₓ + μᵇ²ₘᵢₙ )==0)
            @constraint(model_MP,  λᵇ +v³ -μᵇ³ₘₐₓ + μᵇ³ₘᵢₙ >= 0)        # for Pᵇ³
            @constraint(model_MP, Pᵇ³.* ( λᵇ +v³ -μᵇ³ₘₐₓ + μᵇ³ₘᵢₙ )==0)
        
            #  for Pˢ¹, Pˢ², Pˢ³ in VPPs
            @constraint(model_MP,  -λˢ -v¹ -μˢ¹ₘₐₓ + μˢ¹ₘᵢₙ >= 0)        # for Pˢ¹
            @constraint(model_MP, Pˢ¹.* ( -λˢ -v¹ -μˢ¹ₘₐₓ + μˢ¹ₘᵢₙ )==0)
            @constraint(model_MP,  -λˢ -v² -μˢ²ₘₐₓ + μˢ²ₘᵢₙ >= 0)        # for Pˢ²
            @constraint(model_MP, Pˢ².* ( -λˢ -v² -μˢ²ₘₐₓ + μˢ²ₘᵢₙ )==0)
            @constraint(model_MP,  -λˢ -v³ -μˢ³ₘₐₓ + μˢ³ₘᵢₙ >= 0)        # for Pˢ³
            @constraint(model_MP, Pˢ³.* ( -λˢ -v³ -μˢ³ₘₐₓ + μˢ³ₘᵢₙ )==0)
        
            # for Pᵂ¹, Pᵂ², Pᵂ³ in VPPs
            @constraint(model_MP,   v¹ -μᵂ¹ₘₐₓ + μᵂ¹ₘᵢₙ >= 0)           # for Pᵂ¹
            @constraint(model_MP, Pᵂ¹.* (v¹ -μᵂ¹ₘₐₓ + μᵂ¹ₘᵢₙ)==0)
            @constraint(model_MP,   v² -μᵂ²ₘₐₓ + μᵂ²ₘᵢₙ >= 0)           # for Pᵂ²
            @constraint(model_MP, Pᵂ².* (v² -μᵂ²ₘₐₓ + μᵂ²ₘᵢₙ)==0)
            @constraint(model_MP,   v³ -μᵂ³ₘₐₓ + μᵂ³ₘᵢₙ >= 0)           # for Pᵂ³
            @constraint(model_MP, Pᵂ³.* (v³ -μᵂ³ₘₐₓ + μᵂ³ₘᵢₙ)==0)
        
            # for Pᴹ¹, Pᴹ², Pᴹ³ in VPPs
            @constraint(model_MP, b_MT[1] +2*a_MT[1]*Pᴹ¹[1] +v¹[1] -μᴹ¹ₘₐₓ[1] +μᴹ¹ₘᵢₙ[1] >=0)    # for Pᴹ¹
            @constraint(model_MP, Pᴹ¹[1]*(b_MT[1] +2*a_MT[1]*Pᴹ¹[1] +v¹[1] -μᴹ¹ₘₐₓ[1] +μᴹ¹ₘᵢₙ[1])==0)
            @constraint(model_MP, b_MT[2] +2*a_MT[2]*Pᴹ²[1] +v²[1] -μᴹ²ₘₐₓ[1] +μᴹ²ₘᵢₙ[1] >=0)    # for Pᴹ²
            @constraint(model_MP, Pᴹ²[1]*(b_MT[2] +2*a_MT[2]*Pᴹ²[1] +v²[1] -μᴹ²ₘₐₓ[1] +μᴹ²ₘᵢₙ[1])==0)
            @constraint(model_MP, b_MT[3] +2*a_MT[3]*Pᴹ³[1] +v³[1] -μᴹ³ₘₐₓ[1] +μᴹ³ₘᵢₙ[1] >=0)    # for Pᴹ³
            @constraint(model_MP, Pᴹ³[1]*(b_MT[3] +2*a_MT[3]*Pᴹ³[1] +v³[1] -μᴹ³ₘₐₓ[1] +μᴹ³ₘᵢₙ[1])==0)
        for t in 2: T
            @constraint(model_MP, b_MT[1] +2*a_MT[1]*Pᴹ¹[t] +v¹[t] -μᴹ¹ₘₐₓ[t] +μᴹ¹ₘᵢₙ[t]   -μᵁ¹ᵣ[t] +μᵁ¹ᵣ[t-1] +μᴰ¹ᵣ[t] -μᴰ¹ᵣ[t-1]>=0)    # for Pᴹ¹
            @constraint(model_MP, Pᴹ¹[t]*(  b_MT[1] +2*a_MT[1]*Pᴹ¹[t] +v¹[t] -μᴹ¹ₘₐₓ[t] +μᴹ¹ₘᵢₙ[t]   -μᵁ¹ᵣ[t] +μᵁ¹ᵣ[t-1] +μᴰ¹ᵣ[t] -μᴰ¹ᵣ[t-1]  )==0)
            @constraint(model_MP, b_MT[2] +2*a_MT[2]*Pᴹ²[t] +v²[t] -μᴹ²ₘₐₓ[t] +μᴹ²ₘᵢₙ[t]   -μᵁ²ᵣ[t] +μᵁ²ᵣ[t-1] +μᴰ²ᵣ[t] -μᴰ²ᵣ[t-1]>=0)    # for Pᴹ²
            @constraint(model_MP, Pᴹ²[t]*(  b_MT[2] +2*a_MT[2]*Pᴹ²[t] +v²[t] -μᴹ²ₘₐₓ[t] +μᴹ²ₘᵢₙ[t]   -μᵁ²ᵣ[t] +μᵁ²ᵣ[t-1] +μᴰ²ᵣ[t] -μᴰ²ᵣ[t-1]  )==0)
            @constraint(model_MP, b_MT[3] +2*a_MT[3]*Pᴹ³[t] +v³[t] -μᴹ³ₘₐₓ[t] +μᴹ³ₘᵢₙ[t]   -μᵁ³ᵣ[t] +μᵁ³ᵣ[t-1] +μᴰ³ᵣ[t] -μᴰ³ᵣ[t-1]>=0)    # for Pᴹ³
            @constraint(model_MP, Pᴹ³[t]*(  b_MT[3] +2*a_MT[3]*Pᴹ³[t] +v³[t] -μᴹ³ₘₐₓ[t] +μᴹ³ₘᵢₙ[t]   -μᵁ³ᵣ[t] +μᵁ³ᵣ[t-1] +μᴰ³ᵣ[t] -μᴰ³ᵣ[t-1]  )==0)
        end
            
            # for Pᵇˢ¹, Pᵇˢ², Pᵇˢ³ in VPPs
            @constraint(model_MP,  2*λ_BS[1]*Pᵇˢ¹ +v¹ -w¹/BS_capacity[1] +μᵇˢ¹ₘᵢₙ -μᵇˢ¹ₘₐₓ   >= 0)   # for Pᵇˢ¹
            @constraint(model_MP, Pᵇˢ¹.* ( 2*λ_BS[1]*Pᵇˢ¹ +v¹ -w¹/BS_capacity[1] +μᵇˢ¹ₘᵢₙ -μᵇˢ¹ₘₐₓ )==0)
            @constraint(model_MP,  2*λ_BS[2]*Pᵇˢ² +v² -w²/BS_capacity[2] +μᵇˢ²ₘᵢₙ -μᵇˢ²ₘₐₓ   >= 0)   # for Pᵇˢ²
            @constraint(model_MP, Pᵇˢ².* ( 2*λ_BS[2]*Pᵇˢ² +v² -w²/BS_capacity[2] +μᵇˢ²ₘᵢₙ -μᵇˢ²ₘₐₓ )==0)
            @constraint(model_MP,  2*λ_BS[3]*Pᵇˢ³ +v³ -w³/BS_capacity[3] +μᵇˢ³ₘᵢₙ -μᵇˢ³ₘₐₓ   >= 0)   # for Pᵇˢ³
            @constraint(model_MP, Pᵇˢ³.* ( 2*λ_BS[3]*Pᵇˢ³ +v³ -w³/BS_capacity[3] +μᵇˢ³ₘᵢₙ -μᵇˢ³ₘₐₓ )==0)
        
            # for SoC¹, SoC², SoC³ in VPPs
            @constraint(model_MP, μˢᵒᶜ¹ₘᵢₙ[1] -μˢᵒᶜ¹ₘₐₓ[1] -w¹[1] >=0)     # for SoC¹
            @constraint(model_MP, SoC¹[1]*( μˢᵒᶜ¹ₘᵢₙ[1] -μˢᵒᶜ¹ₘₐₓ[1] -w¹[1] )==0)
            @constraint(model_MP, μˢᵒᶜ²ₘᵢₙ[1] -μˢᵒᶜ²ₘₐₓ[1] -w²[1] >=0)     # for SoC²
            @constraint(model_MP, SoC²[1]*( μˢᵒᶜ²ₘᵢₙ[1] -μˢᵒᶜ²ₘₐₓ[1] -w²[1] )==0)
            @constraint(model_MP, μˢᵒᶜ³ₘᵢₙ[1] -μˢᵒᶜ³ₘₐₓ[1] -w³[1] >=0)     # for SoC³
            @constraint(model_MP, SoC³[1]*( μˢᵒᶜ³ₘᵢₙ[1] -μˢᵒᶜ³ₘₐₓ[1] -w³[1] )==0)
        for t in 2: T
            @constraint(model_MP, μˢᵒᶜ¹ₘᵢₙ[t] -μˢᵒᶜ¹ₘₐₓ[t] -w¹[t] +w¹[t-1] >=0)     # for SoC¹
            @constraint(model_MP, SoC¹[t]*( μˢᵒᶜ¹ₘᵢₙ[t] -μˢᵒᶜ¹ₘₐₓ[t] -w¹[t] +w¹[t-1])==0)
            @constraint(model_MP, μˢᵒᶜ²ₘᵢₙ[t] -μˢᵒᶜ²ₘₐₓ[t] -w²[t] +w²[t-1] >=0)     # for SoC²
            @constraint(model_MP, SoC²[t]*( μˢᵒᶜ²ₘᵢₙ[t] -μˢᵒᶜ²ₘₐₓ[t] -w²[t] +w²[t-1])==0)
            @constraint(model_MP, μˢᵒᶜ³ₘᵢₙ[t] -μˢᵒᶜ³ₘₐₓ[t] -w³[t] +w³[t-1] >=0)     # for SoC³
            @constraint(model_MP, SoC³[t]*( μˢᵒᶜ³ₘᵢₙ[t] -μˢᵒᶜ³ₘₐₓ[t] -w³[t] +w³[t-1] )==0)
        end
            
           
        # Primal variables and constraints for upper level, also in the MP optimization (no dual variables)
            @variable(model_MP, Pᴰˢᴼᵇ[1:T])       # Quantities DSO trade with Wholesale market
            @variable(model_MP, Pᴰˢᴼˢ[1:T]) 
            @variable(model_MP, N₁[1:T],Bin)      # Binary variables for logical judgement
            @variable(model_MP, N₂[1:T],Bin) 
         
            @constraint(model_MP, λᵇ <= λ_PMb)    # Upper and Lower bounds for prices
            @constraint(model_MP, λˢ <= λ_PMb)
            @constraint(model_MP, λ_PMs <= λˢ)
            @constraint(model_MP, λ_PMs <= λᵇ)
        
            @constraint(model_MP, Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰<=M₂*N₂)      # big-M formulation of logical judgement 2
            @constraint(model_MP, -M₂*(ones(1,T)'-N₂).<=Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰)
            @constraint(model_MP, -M₂*N₂.<=Pᴰˢᴼˢ+(Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰))
            @constraint(model_MP, Pᴰˢᴼˢ+(Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰).<=M₂*N₂)
            @constraint(model_MP, -M₂*(ones(1,T)'-N₂).<=Pᴰˢᴼˢ)
            @constraint(model_MP, Pᴰˢᴼˢ.<=M₂*(ones(1,T)'-N₂))
         
            @constraint(model_MP, Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰<=M₁*N₁)      # big-M formulation of logical judgement 1 
            @constraint(model_MP, -M₁*(ones(1,T)'-N₁).<=Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰)
            @constraint(model_MP, -M₁*(ones(1,T)'-N₁).<=Pᴰˢᴼᵇ-(Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰))
            @constraint(model_MP, Pᴰˢᴼᵇ-(Pⱽᴾᴾ¹⁰+Pⱽᴾᴾ²⁰+Pⱽᴾᴾ³⁰).<=M₁*(ones(1,T)'-N₁))
            @constraint(model_MP, -M₁*N₁.<=Pᴰˢᴼᵇ)
            @constraint(model_MP, Pᴰˢᴼᵇ.<=M₁*N₁)
        
        # Constrain the upper bound of LL obj with real solution 
        LL_obj_VPP1_real=sum(λᵇ.*Pᵇ¹⁰ -λˢ.*Pˢ¹⁰ +a_MT[1]*Pᴹ¹⁰.*Pᴹ¹⁰+b_MT[1]*Pᴹ¹⁰ +c_MT[1]*u¹⁰ +λ_BS[1]*Pᵇˢ¹⁰.*Pᵇˢ¹⁰)       # Obj for VPP1    
        LL_obj_VPP2_real=sum(λᵇ.*Pᵇ²⁰ -λˢ.*Pˢ²⁰ +a_MT[2]*Pᴹ²⁰.*Pᴹ²⁰+b_MT[2]*Pᴹ²⁰ +c_MT[2]*u²⁰ +λ_BS[2]*Pᵇˢ²⁰.*Pᵇˢ²⁰)       # Obj for VPP2 
        LL_obj_VPP3_real=sum(λᵇ.*Pᵇ³⁰ -λˢ.*Pˢ³⁰ +a_MT[3]*Pᴹ³⁰.*Pᴹ³⁰+b_MT[3]*Pᴹ³⁰ +c_MT[3]*u³⁰ +λ_BS[3]*Pᵇˢ³⁰.*Pᵇˢ³⁰)       # Obj for VPP3
        LL_obj_real=LL_obj_VPP1_real+LL_obj_VPP2_real+LL_obj_VPP3_real              # Obj for LL program
        
        LL_obj_VPP1_cons=sum(λᵇ.*Pᵇ¹ -λˢ.*Pˢ¹ +a_MT[1]*Pᴹ¹.*Pᴹ¹+b_MT[1]*Pᴹ¹ +c_MT[1]*u¹ +λ_BS[1]*Pᵇˢ¹.*Pᵇˢ¹)       
        LL_obj_VPP2_cons=sum(λᵇ.*Pᵇ² -λˢ.*Pˢ² +a_MT[2]*Pᴹ².*Pᴹ²+b_MT[2]*Pᴹ² +c_MT[2]*u² +λ_BS[2]*Pᵇˢ².*Pᵇˢ²)       
        LL_obj_VPP3_cons=sum(λᵇ.*Pᵇ³ -λˢ.*Pˢ³ +a_MT[3]*Pᴹ³.*Pᴹ³+b_MT[3]*Pᴹ³ +c_MT[3]*u³ +λ_BS[3]*Pᵇˢ³.*Pᵇˢ³)       
        LL_obj_cons=LL_obj_VPP1_cons+LL_obj_VPP2_cons+LL_obj_VPP3_cons             
                 
        @constraint(model_MP, LL_obj_real <= LL_obj_cons)        # The upper bound for LL program
        
        # Obj of upper-level program
            @objective(model_MP, Max,  sum(λ_PMs.*Pᴰˢᴼˢ)-sum(λ_PMb.*Pᴰˢᴼᵇ) + sum(λᵇ.*(Pᵇ¹⁰+Pᵇ²⁰+Pᵇ³⁰))-sum(λˢ.*(Pˢ¹⁰+Pˢ²⁰+Pˢ³⁰)))  
            set_optimizer(model_MP, Gurobi.Optimizer)
            # set_optimizer_attribute(model_MP, "NonConvex", -1)
            optimize!(model_MP)
        
            UB=min(UB, JuMP.value.( sum(λ_PMs.*Pᴰˢᴼˢ)-sum(λ_PMb.*Pᴰˢᴼᵇ) + sum(λᵇ.*(Pᵇ¹⁰+Pᵇ²⁰+Pᵇ³⁰))-sum(λˢ.*(Pˢ¹⁰+Pˢ²⁰+Pˢ³⁰))  ) )
            λᵇ=JuMP.value.(λᵇ)
            λˢ=JuMP.value.(λˢ)
        
          
        return  UB,  λᵇ, λˢ
    
    
    
    
    
    
    
    
    
    
    
    end
    
    end