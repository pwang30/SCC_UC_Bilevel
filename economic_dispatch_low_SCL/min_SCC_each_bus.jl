function min_SCC_each_bus(commitment_SGs,I_IBG,β,v) 

    numnodes=30                         # number of nodes
    #            Bus Number	     x1	      x2
        SGpara=[
                   2	       0.0697	0.0712	
                   3	       0.0531	0.0672	
                   4	       0.0573	0.0503	
                   5	       0.0846	0.0658	
                   27	       0.0719	0.0657	
                   30	       0.0683	0.0749 ]  # buses where SGs are located

    Y_SGs = zeros(size(SGpara,1), size(SGpara,2)-1)    # define ADMITTANCE MATRIX of the SGs, buses:2,3,4,5,27,30
    I_SGs = zeros(size(SGpara,1), size(SGpara,2)-1)    # define I_SGs of the SGs, buses:2,3,4,5,27,30
    
    Yₗᵢₙₑ=admittance_matrix_calculation(numnodes)   # calculate the ADMITTANCE MATRIX of the network

    for k in 1:size(SGpara,1)                      # calculate the ADMITTANCE MATRIX of the SGs
        for j in 2:size(SGpara,2)
            Y_SGs[k, j-1] = 1/SGpara[k, j]      # coffecient here can be changed           
        end           
    end 
    

    E_sg=v                   # nominal voltage of the SGs
    E_sg=β*E_sg
    I_SGs=Y_SGs.*E_sg        # calculate the I_SGs of the SGs, buses:2,3,4,5,27,30



    SCC_min=zeros(1,numnodes)  # define the minimum SCC for each bus
    I_SCC_all_buses=zeros(24,numnodes)  # define the SCC for each bus for all scenarios


    for j in 1:numnodes
        index = 1
        Y_SGs_with_status = zeros(numnodes, numnodes)     # define the ADMITTANCE MATRIX for SGs status
        Y_total = zeros(numnodes, numnodes)        # define the total ADMITTANCE MATRIX
        Z = zeros(numnodes, numnodes)              # define the impedance matrix
        status_SGs=zeros(1,12)                      # define the status of SGs

        for t in 1:24
            status_SGs=commitment_SGs[t,:]  
            status_IBG=[1 1 1]  # status of IBG
            for i in 1:length(SGpara[:,2:end])
                Y_SGs_with_status[Int(SGpara[Int(ceil(i/2)), 1]), Int(SGpara[Int(ceil(i/2)), 1])] = Y_SGs_with_status[Int(SGpara[Int(ceil(i/2)), 1]), Int(SGpara[Int(ceil(i/2)), 1])]+Y_SGs[Int(ceil(i/2)),index] * status_SGs[i] 
                index = index+1
                if index > size(Y_SGs, 2)
                    index = 1
                end
            end
            
            Y_total .= Yₗᵢₙₑ + Y_SGs_with_status      # calculate the total ADMITTANCE MATRIX
            Z .= inv(Y_total)                        # calculate the IMPEDANCE MATRIX
        
            I_SCC_all_buses[t,j] = (
                    Z[j,2]*(I_SGs[1, 1]*status_SGs[1]+ I_SGs[1, 2]*status_SGs[2])+
                    Z[j,3]*(I_SGs[2, 1]*status_SGs[3]+ I_SGs[2, 2]*status_SGs[4])+
                    Z[j,4]*(I_SGs[3, 1]*status_SGs[5]+ I_SGs[3, 2]*status_SGs[6])+
                    Z[j,5]*(I_SGs[4, 1]*status_SGs[7]+ I_SGs[4, 2]*status_SGs[8])+
                    Z[j,27]*(I_SGs[5, 1]*status_SGs[9]+ I_SGs[5, 2]*status_SGs[10])+
                    Z[j,30]*(I_SGs[6, 1]*status_SGs[11]+ I_SGs[6, 2]*status_SGs[12])+
                    Z[j,1]*I_IBG*status_IBG[1]+ Z[j,23]*I_IBG*status_IBG[2]+ Z[j,26]*I_IBG*status_IBG[3])/Z[j,j]   # calculate the SCC for SGs, buses:2,3,4,5,27,30
            
        end
            SCC_min[j]=minimum(I_SCC_all_buses[:,j])  # calculate the minimum SCC for each bus
    end

    return SCC_min
        
        
 end
    