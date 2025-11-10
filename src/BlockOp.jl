#Define block multiplications, inversions etc with the fast elimination algorithm:

using LinearAlgebra
using Einsum

"""
    block_tri_lower_inverse_old(mat)

**Old implementation** of block lower-triangular matrix inversion.

⚠️ **Deprecated**: Use [`block_tri_lower_inverse`](@ref) instead for better performance.

This function is kept for reference and testing purposes only.

# See Also

- [`block_tri_lower_inverse`](@ref): Optimized version using einsum
"""
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

"""
    block_tri_lower_inverse(mat)

Compute inverse of **block lower-triangular** matrix using fast elimination.

This is the core linear algebra operation for gSBR self-energies, implementing the
resummation of bubble diagrams. The matrix has a two-level structure:
- **Block level**: Each element is itself a matrix (dimension `len_n × len_n`)
- **Time level**: Blocks arranged in lower-triangular form (dimension `len_time × len_time`)

# Arguments

- `mat::Array{Float64, 4}`: Input matrix with shape `(len_n, len_n, len_time, len_time)`
  - Dimensions 1-2: Block matrix elements (n-indices)
  - Dimensions 3-4: Time indices (lower-triangular structure)

# Returns

- `Array{Float64, 4}`: Inverse matrix with same shape

# Algorithm

Uses optimized forward elimination with `@einsum` for tensor contractions:

1. Invert diagonal blocks: \$M_{ii}^{-1}\$
2. For each row below diagonal:
   - Compute \$\\Xi_{ij} = -M_{ii}^{-1} \\sum_{k=j}^{i-1} M_{ik} \\Xi_{kj}\$

This is **much faster** than naive matrix inversion for large time grids.

# Usage in gSBR

Appears in self-energy calculations as:

```julia
χ = block_lower_shift(Γ_bubble * c_mn)  # Build lower-triangular matrix
Ξ = block_tri_lower_inverse(I - χ)      # Invert to get geometric series
```

This inverts the operator \$[I - \\chi]\$ to resum all bubble contributions.

# Example

```julia
# 3 time steps, 2 n-indices
mat = zeros(2, 2, 3, 3)
mat[:,:,1,1] = I(2)
mat[:,:,2,2] = I(2)
mat[:,:,3,3] = I(2)
mat[:,:,2,1] = [0.1 0.0; 0.0 0.1]  # Lower-triangular coupling

inv_mat = block_tri_lower_inverse(mat)
# Result satisfies: sum(mat[:,:,i,k] * inv_mat[:,:,k,j]) ≈ I(2) * δ_{ij}
```

# See Also

- [`block_tri_lower_inverse_old`](@ref): Slower reference implementation
- [`block_identity`](@ref): Creates identity for this structure
- [`self_energy_SBR_mixed!`](@ref), [`self_energy_SBR_mixed_cross_noC!`](@ref): Main users
"""
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

"""
    block_mat_mul(mat1, mat2)

Multiply two **block-structured** matrices with matching dimensions.

Each matrix element is itself a block (n-indices), and the matrices have time structure.
Uses optimized einsum contraction for efficiency.

# Arguments

- `mat1::Array{Float64, 4}`: First matrix `(len_n, len_n, len_time, len_time)`
- `mat2::Array{Float64, 4}`: Second matrix (same shape)

# Returns

- `Array{Float64, 4}`: Product `mat1 * mat2` with same shape

# Mathematical Operation

Performs block matrix multiplication:

```math
(M_1 M_2)_{ij} = \\sum_k M_1_{ik} M_2_{kj}
```

where each \$M_{ij}\$ is itself an `len_n × len_n` matrix.

# Example

```julia
mat1 = rand(2, 2, 3, 3)  # Random block matrix
mat2 = block_identity(2, 3)  # Block identity
result = block_mat_mul(mat1, mat2)  # Should equal mat1
```

# See Also

- [`block_mat_mix_mul`](@ref): Multiply with mixed structure
- [`block_mat_vec_mul`](@ref): Multiply with vector
"""    
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

"""
    block_lower_shift(mat)

Shift block matrix to create **strict lower-triangular** structure (zero diagonal).

Multiplies the time-dimension by a shift matrix to move all blocks down by one time step.
This implements the causality constraint in memory integrals.

# Arguments

- `mat::Array{Float64, 4}`: Input matrix `(len_n, len_n, len_time, len_time)`

# Returns

- `Array{Float64, 4}`: Shifted matrix with diagonal blocks zeroed

# Operation

Applies shift operator in time dimension:

```math
\\text{shifted}[i,j,k,m] = \\sum_\\ell \\text{mat}[i,j,k,\\ell] \\cdot L[\\ell,m]
```

where \$L\$ is the lower-shift matrix: `L = diagm(-1 => ones(len_time - 1))`

# Example

```julia
mat = ones(2, 2, 3, 3)  # All blocks = ones(2,2)
shifted = block_lower_shift(mat)
# shifted[:,:,1,:] = 0 (first row cleared)
# shifted[:,:,2,1] = mat[:,:,2,2] (moved down)
# shifted[:,:,3,2] = mat[:,:,3,3] (moved down)
```

# Usage in gSBR

Creates the strict lower-triangular structure needed for geometric series:

```julia
χ = block_lower_shift(Γ .* c_mn)  # Memory kernel (no diagonal)
Ξ = block_tri_lower_inverse(I - χ)  # Resum: I + χ + χ² + ...
```

# See Also

- [`block_tri_lower_inverse`](@ref): Inverts the shifted result
"""
function block_lower_shift(mat)
                                    
    len_rows = size(mat, 3)
    L = diagm(-1 => ones(len_rows - 1))

    @einsum shifted[i,j,k,m] :=  mat[i,j,k,l] * L[l,m]
    
    return shifted
                                    
end

"""
    block_mat_mix_mul(mat1, mat2)

Multiply **block matrix** with **mixed-structure matrix** (one n-index, two time indices).

This specialized operation appears in gSBR when contracting the full block structure
with single-species contributions.

# Arguments

- `mat1::Array{Float64, 4}`: Block matrix `(len_n1, len_n2, len_time, len_time)`
- `mat2::Array{Float64, 3}`: Mixed matrix `(len_n2, len_time, len_time)` (only first index is block)

# Returns

- `Array{Float64, 3}`: Result `(len_n1, len_time, len_time)`

# Mathematical Operation

```math
C_{i,k,m} = \\sum_{j,\\ell} M1_{i,j,k,\\ell} \\cdot M2_{j,\\ell,m}
```

The result has the mixed structure (one n-index, two time indices).

# Usage Example

In gSBR self-energies:

```julia
Γ = ...  # Block bubble matrix (4D)
cN0 = ...  # Coefficient-density products (3D)
Ξ_μ = block_mat_mix_mul(Γ, cN0)  # Contract to get single-species contribution
```

# See Also

- [`block_mat_mul`](@ref): Full block-block multiplication
- [`block_vec_mat_mul_single_sp`](@ref): Vector-matrix multiplication
"""    
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

"""
    block_mat_vec_mul(mat, vec)

Calculates the matrix product between a compatible matrix and a vector etc which have the block structure! The first two indices corresspond to the list index.
""" 
function block_mat_vec_mul(mat,vec)

                                            
    len_rows = size(mat, 3)
    mul = zeros(size(mat)[1], len_rows)

    for i in 1:len_rows, k in 1:len_rows
        mul[:, i] .+= mat[:, :, i, k] * vec[:, k]
    end

    return mul
                                            
end

"""
    block_vec_mat_mul(vec, mat)
    
Calculates the matrix product between a compatible vector and a matrix which have the block structure! The first two indices corresspond to the list index.
"""
function block_vec_mat_mul(vec,mat)
                                                    
    len_rows = size(mat, 3)
    mul = zeros(size(mat)[1], len_rows)

    for i in 1:len_rows, k in 1:len_rows
        mul[:, i] .+= vec[:, k] * mat[:, :, k, i]
    end

    return mul
                                                    
end
    
"""
    block_vec_mat_mul_single_sp(vec, mat)

Calculates the matrix product between a compatible vector and a matrix which have the block structure! 
Only the first index corresspond to the list index and the vec has no time index
"""
function block_vec_mat_mul_single_sp(vec,mat)

    
    len_rows = size(mat, 2)
    mul = zeros(len_rows, len_rows)

    for i in 1:size(mat, 1)
        mul[:,:] .+= vec[i] * mat[i,:,:]
    end

    return mul
    
end

"""
    block_identity(dim_list, dim_time)

Create **block identity matrix** for gSBR linear algebra.

Generates a 4D array representing the identity operator in the block-time structure:
- Block dimension: `dim_list × dim_list` identity matrices
- Time dimension: `dim_time × dim_time` diagonal structure

# Arguments

- `dim_list::Int`: Size of block matrices (number of n-indices)
- `dim_time::Int`: Number of time steps

# Returns

- `Array{Float64, 4}`: Identity with shape `(dim_list, dim_list, dim_time, dim_time)`

# Structure

```math
I[i,j,k,\\ell] = \\delta_{ij} \\delta_{k\\ell}
```

Only elements with `i == j` and `k == l` are 1.0, all others are 0.0.

# Example

```julia
I_block = block_identity(3, 5)  # 3 n-indices, 5 time steps
# size(I_block) = (3, 3, 5, 5)
# I_block[1,1,2,2] = 1.0
# I_block[1,2,2,2] = 0.0 (off-diagonal in block)
# I_block[1,1,2,3] = 0.0 (off-diagonal in time)
```

# Usage

Essential for constructing operators in gSBR:

```julia
Ξ = block_tri_lower_inverse(block_identity(len_n, t) - χ)
# Inverts (I - χ) to get geometric series
```

# See Also

- [`block_tri_lower_inverse`](@ref): Inverts operators built with this identity
- [`block_mat_mul`](@ref): Multiplies block matrices
"""
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

"""
    response_combinations(n1, n2, R)

Compute all permutations of cross-species response products.

When computing the bubble corrections with cross-species responses, products like
\$R_{i,j}(t,t') R_{k,\\ell}(t,t')\$ must account for all ways to pair species indices.
This function handles the combinatorics.

# Arguments

- `n1::Vector{Int}`: First multi-index (indicates which species appear)
- `n2::Vector{Int}`: Second multi-index
- `R::Array{Float64}`: Cross-response matrix \$R_{ij}(t,t')\$ with shape `(num_species, num_species, t, t)`

# Returns

- `Float64` or `Array{Float64}`: Appropriate combination of response products

# Cases

**Both n1 and n2 have 2 non-zero entries** (e.g., n1=[1,1,0], n2=[1,0,1]):
```math
R_{i_1,j_1} R_{i_2,j_2} + R_{i_1,j_2} R_{i_2,j_1}
```

**n1 has 1 entry, n2 has 2** (e.g., n1=[2,0,0], n2=[1,1,0]):
```math
2 R_{i,j_1} R_{i,j_2}
```

**n1 has 2 entries, n2 has 1** (symmetric to above):
```math
2 R_{i_1,j} R_{i_2,j}
```

**Both have 1 entry** (e.g., n1=[2,0,0], n2=[1,0,0]):
```math
2 R_{i,j}^2
```

# Limitations

⚠️ **Currently only supports binary reactions** (max 2 reactants per reaction).
Higher-order reactions would require additional combinatorics.

# Example

```julia
# S + E reaction: n1=[1,1,0], n2=[1,0,0]
R = rand(3, 3, 10, 10)  # 3 species, 10 time points
combo = response_combinations([1,1,0], [1,0,0], R)
# Returns: 2 * R[1,1,:,:] .* R[2,1,:,:]
```

# Usage in Self-Energies

Appears in MCA cross-species calculations:

```julia
Γ = collect(
    ==(sum(n′), sum(n′′)) .* response_combinations(n′, n′′, variables.R[:,:,tt,ttt])
    for n′ in n_listNEW_R, n′′ in n_listNEW_R, tt in 1:t, ttt in 1:t
)
```

# See Also

- [`self_energy_alpha2_cross!`](@ref): Main user of this function
- [`self_energy_SBR_mixed_cross_noC!`](@ref): Also uses cross-species products
"""
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