# Achedmy -- Adaptive CHEmical Dynamics using MemorY

This module implements the **memory corrections** (SBR - self-consistent bubble resummation) for the dynamics of Chemical Reaction Networks (CRNs), calculating the one point and two point quantities accurately.

The CRNs are defined using `Catalyst.jl` and the module uses `KB.jl` to solve the resulting two time equations using adaptive time step given some tolerance values.

Author: Moshir Harsh

Email : btemoshir@gmail.com

Dependencies:
```
Catalyst.jl
KadanoffBaym.jl
LinearAlgebra.jl
....
```

TODO: Implement C calculation!
