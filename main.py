from dmt import SimplicialComplex, DiscreteMorseFunction, PersistentHomology, TopologicalInvariants

# Step 1: Construct a simplicial complex
complex = SimplicialComplex()
complex.add_simplex((1,))       # vertex A
complex.add_simplex((2,))       # vertex B
complex.add_simplex((3,))       # vertex C
complex.add_simplex((1, 2))   # edge AB
complex.add_simplex((2, 3))   # edge BC
complex.add_simplex((1, 3))   # edge AC
complex.add_simplex((1, 2, 3))  # face ABC

# Step 2: Assign a discrete Morse function and compute gradient vector field
dmf = DiscreteMorseFunction(complex)
dmf.assign_random(low=0.0, high=1.0)  # Random assignments to simplices
gvf = dmf.compute_gradient_vector_field()

# Step 3: Identify critical cells
critical_cells = dmf.identify_critical_cells()
print("Critical cells:", critical_cells)

# Step 4: Perform collapses to simplify the complex
dmf.perform_collapse()
print("After collapse, f-vector:", dmf.complex.compute_f_vector())

# Step 5: Compute topological invariants and persistent homology
betti_numbers = TopologicalInvariants.compute_betti_numbers(dmf.complex)
print("Betti numbers after simplification:", betti_numbers)

persistence = PersistentHomology.compute_persistence_diagram(dmf.complex)
print("Persistence diagram:", persistence)
