#Define block multiplications, inversions etc with the fast elimination algorithm:

using LinearAlgebra
using Einsum

function block_tri_lower_inverse_old(mat)
    """
    Calculates the inverse of a lower triangular matrix where each elemet itself is a block matrix. Used for mixed SBR series!
    mat should have the first two dimensions coressponding to the block of n, the last two to the time. Assumes there are 1's on the diagonal!
    
    """                    
    len_rows = size(mat, 3)
    len_n = size(mat, 1)
    invM = similar(mat)

    # Do the diagonals first:
    for diag in 1:len_rows
        invM[:, :, diag, diag] .= inv(mat[:, :, diag, diag])
    end

    for row in 2:len_rows
        for col in 1:row - 1
            temp = zeros(size(mat)[1:2])
            for col2 in col:row - 1
                temp .+= mat[:, :, row, col2] * invM[:, :, col2, col]
            end
            invM[:, :, row, col] .= -invM[:, :, row, row] * temp
        end
    end

    return invM
    
end

function block_tri_lower_inverse(mat)
    """
    Calculates the inverse of a lower triangular matrix where each elemet itself is a block matrix. Used for mixed SBR series!
    mat should have the first two dimensions coressponding to the block of n, the last two to the time. Assumes there are 1's on the diagonal!
    
    """
                        
    len_rows = size(mat, 3)
    len_n    = size(mat, 1)
    invM     = zeros(size(mat))
    
    invM[:, :, 1, 1] .= inv(mat[:, :, 1, 1])

    for row in 2:len_rows
        D_inv = inv(mat[:, :, row, row])
        mat_copy = deepcopy(mat[:,:,row,1:row])
        inv_copy = deepcopy(invM[:,:,1:row,1:row])
        
        @einsum temp_n[i,m,o] := mat_copy[i,j,l] * inv_copy[j,m,l,o]
                temp_n2 = block_identity(len_n,row)[:,:,row,1:row] .- temp_n[:,:,1:row]
        @einsum inv_temp[i,m,k] := D_inv[i,j] * temp_n2[j,m,k]
        
        invM[:,:,row,1:row] .= inv_temp[:,:,:]
            
    end

                        
    return invM
                        
end
    
function block_mat_mul(mat1,mat2)
    """
    Calculates the matrix product between two compatible matrices etc which have the block structure! The first two indices corresspond to the list index.
    """
                                    
    len_rows = size(mat1, 3)
    mul = similar(mat1)

    for i in 1:len_rows, j in 1:len_rows, k in 1:len_rows
        mul[:, :, i, j] .+= mat1[:, :, i, k] * mat2[:, :, k, j]
    end

    @einsum mul[i,n,k,o] := mat1[i,j,k,l] * mat2[j,n,l,o]
    
    return mul

end

function block_lower_shift(mat)
                                    
    len_rows = size(mat, 3)
    L = diagm(-1 => ones(len_rows - 1))

    @einsum shifted[i,j,k,m] :=  mat[i,j,k,l] * L[l,m]
    
    return shifted
                                    
end
    

function block_mat_mix_mul(mat1,mat2)
    """
    Calculates the matrix product between a matrix which has the block structure with another matrix which has a structure with only one n but two time indices! The first two indices corresspond to the list index.
    """
    
    #len_rows = np.shape(mat1)[2]
    #mul      = np.zeros([np.shape(mat1)[0],np.shape(mat1)[2],np.shape(mat1)[3]])
    
    #for i in range(len_rows):
        #for j in range(len_rows):
            #for k in range(len_rows):
                #mul[:,i,j] += np.matmul(mat1[:,:,i,k],mat2[:,k,j])
    
    #mul = np.einsum('ijkl,jlm->ikm',mat1,mat2,optimize='optimal')
                                    
    @einsum mul[i,k,m] := mat1[i,j,k,l] * mat2[j,l,m]
    
    return mul
                                    
end
    
function block_mat_vec_mul(mat,vec)
    """
    Calculates the matrix product between a compatible matrix and a vector etc which have the block structure! The first two indices corresspond to the list index.
    """
                                            
    len_rows = size(mat, 3)
    mul = zeros(size(mat)[1], len_rows)

    for i in 1:len_rows, k in 1:len_rows
        mul[:, i] .+= mat[:, :, i, k] * vec[:, k]
    end

    return mul
                                            
end

function block_vec_mat_mul(vec,mat)
    """
    Calculates the matrix product between a compatible vector and a matrix which have the block structure! The first two indices corresspond to the list index.
    """
                                                    
    len_rows = size(mat, 3)
    mul = zeros(size(mat)[1], len_rows)

    for i in 1:len_rows, k in 1:len_rows
        mul[:, i] .+= vec[:, k] * mat[:, :, k, i]
    end

    return mul
                                                    
end
    
function block_vec_mat_mul_single_sp(vec,mat)
    """
    Calculates the matrix product between a compatible vector and a matrix which have the block structure! 
    Only the first index corresspond to the list index and the vec has no time index
    """
    
    len_rows = size(mat, 2)
    mul = zeros(len_rows, len_rows)

    for i in 1:size(mat, 1)
        mul[:,:] .+= vec[i] * mat[i,:,:]
    end

    return mul
    
end
    
function block_identity(dim_list,dim_time)
    """
    Creates the identity for the block matrix system
    """
    
    identity = zeros(dim_list,dim_list,dim_time,dim_time)
    
    for i in 1:dim_time
        for j in 1:dim_list
            identity[j,j,i,i] = 1.
        end
    end
        
    return identity
                                                            
end

function response_combinations(n1,n2,R)
    
    """
    Creates appropriate responde combinations in the bubbles or the "irreducible memory functions" when cross response functions are allowed!

    This works for systems with at most binary reactions! [To be updated]
    
    """
    
    x1 = findall(n1 .!= 0)
    x2 = findall(n2 .!= 0)
    
    if length(x1) == 2 && length(x2) == 2
    
        resp_comb = R[x1[1],x2[1]].*R[x1[2],x2[2]] .+ R[x1[1],x2[2]].*R[x1[2],x2[1]]
        
    elseif length(x1) == 1 && length(x2) == 2
        
        resp_comb = 2. .*R[x1[1],x2[1]].*R[x1[1],x2[2]]
    
    elseif length(x1) == 2 && length(x2) == 1
        
        resp_comb = 2. .*R[x1[1],x2[1]].*R[x1[2],x2[1]]
        
    elseif length(x1) == 1 && length(x2) == 1
        
        resp_comb = 2. .*R[x1[1],x2[1]].*R[x1[1],x2[1]]
        
    end
        
    return resp_comb
        
end