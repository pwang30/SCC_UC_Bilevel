function offline_trainning(I_scc_all, matrix_ω, Iₗᵢₘ, v)

#-----------------------------------Dataset Classification
K_g=[]
K_c=[]
K_m=[]
Obj_total=[]
for k in 1:size(I_scc_all,2)

I_scc_Ω1 = []     # Initialize the classification matrix
I_scc_Ω2 = []
I_scc_Ω3 = []
matrix_ω_Ω1 = []
matrix_ω_Ω2 = []
matrix_ω_Ω3 = []

for i in 1:size(I_scc_all,1)  # Traverse I_scc and matrix_ω for classification
    if I_scc_all[i,k] < Iₗᵢₘ
        push!(I_scc_Ω1, I_scc_all[i,k])
        push!(matrix_ω_Ω1, matrix_ω[i,:])
    elseif Iₗᵢₘ <= I_scc_all[i,k] < Iₗᵢₘ + v
        push!(I_scc_Ω2, I_scc_all[i,k])
        push!(matrix_ω_Ω2, matrix_ω[i,:])
    else
        push!(I_scc_Ω3, I_scc_all[i,k])
        push!(matrix_ω_Ω3, matrix_ω[i,:])
    end
end

I_scc_Ω1 = hcat(I_scc_Ω1...) # Convert the classification results into a matrix
I_scc_Ω2 = hcat(I_scc_Ω2...)
I_scc_Ω3 = hcat(I_scc_Ω3...)
matrix_ω_Ω1 = hcat(matrix_ω_Ω1...)'
matrix_ω_Ω2 = hcat(matrix_ω_Ω2...)'
matrix_ω_Ω3 = hcat(matrix_ω_Ω3...)'


#-----------------------------------Define Offline-Training Model
model_ot= Model()

# @variable(model_ot, k_fg[1:6]>=0)  # buses of SGs are 2,3,4,5,27,30. F=4
# @variable(model_ot, k_fc[1:3]>=0)  # buses of IBG are 1,23,26. F=4
# @variable(model_ot, k_fm[1:15]>=0)  # pairs of SGs

@variable(model_ot, k_fg[1:6])  # buses of SGs are 2,3,4,5,27,30. F=4
@variable(model_ot, k_fc[1:3])  # buses of IBG are 1,23,26. F=4
@variable(model_ot, k_fm[1:15])  # pairs of SGs

for i in 1:size(matrix_ω_Ω1,1)
    @constraint(model_ot, sum(k_fg.* matrix_ω_Ω1[i,1:6])+ sum(k_fc.* matrix_ω_Ω1[i,7:9])+ sum(k_fm.* matrix_ω_Ω1[i,10:24]) <=Iₗᵢₘ ) 
end

for i in 1:size(matrix_ω_Ω3,1)
    @constraint(model_ot, sum(k_fg.* matrix_ω_Ω3[i,1:6])+ sum(k_fc.* matrix_ω_Ω3[i,7:9])+ sum(k_fm.* matrix_ω_Ω3[i,10:24]) >=Iₗᵢₘ ) 
end


@variable(model_ot, penalty_one_time_item[1:size(matrix_ω_Ω2,1)])
for i in 1:size(matrix_ω_Ω2,1)
    @constraint(model_ot, penalty_one_time_item[i]==I_scc_Ω2[i]- (sum(k_fg.* matrix_ω_Ω2[i,1:6])+ sum(k_fc.* matrix_ω_Ω2[i,7:9])+ sum(k_fm.* matrix_ω_Ω2[i,10:24])))
end


@objective(model_ot, Min, sum(penalty_one_time_item.^2))  
#-----------------------------------Solve and Output Results
set_optimizer(model_ot , Gurobi.Optimizer)
optimize!(model_ot)

k_fg=JuMP.value.(k_fg)
k_fc=JuMP.value.(k_fc)
k_fm=JuMP.value.(k_fm)
obj=JuMP.value.(sum(penalty_one_time_item.^2))

push!(K_g, k_fg)
push!(K_c, k_fc)
push!(K_m, k_fm)
push!(Obj_total, obj)

end

K_g = hcat(K_g...) 
K_c = hcat(K_c...)
K_m = hcat(K_m...)
Obj_total = hcat(Obj_total...)

return  K_g, K_c, K_m, Obj_total

end
