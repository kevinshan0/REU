using QuantumSavory

reg = Register(2)

# Create Bell State
initialize!(reg[1], Z1)
initialize!(reg[2], Z1)
apply!(reg[1], H)
apply!((reg[1], reg[2]), CNOT)

### observables show expectation value
xobs = observable((reg[1], reg[2]), X ⊗ X)
zobs = observable((reg[1], reg[2]), Z ⊗ Z)

println(xobs)
println(zobs)
