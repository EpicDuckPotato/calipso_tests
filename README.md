## Repository Overview

Discrete-time dynamics, as well as things like state and control limits, are defined in structs that extend the DynamicalSystem struct in dynamical_system.jl. Here, the two systems are Acrobot (acrobot.jl) and Snake (snake.jl). 

The dynamics constraint associated with a dynamical system can be computed by the functions in dynamics_constraint.jl.

Cost functions extend the StageCost struct in stage_cost.jl.

The Transcription struct in transcription.jl takes the components of the above files and formulates the NLP associated with the trajectory optimization problem.

To run an acrobot swingup with no control limits, run include("./calipso_acrobot_no_cones.jl"). To run a snake swingup with no control limits, run include("./calipso_snake_no_cones.jl").

To run an acrobot swingup with control limits, run include("./calipso_acrobot.jl"). To run a snake swingup with control limits, run include("./calipso_snake.jl")

The snake swingup with control limits (i.e. with cones) crashes my laptop if I don't kill it within a few minutes. Without control limits (i.e. without cones), it runs fine. The acrobot with cones does finish, but there's still a very large delay before it does, as I describe below.

## Problem Explanation

The problem that I'm having is that when cone! is called on line 113 of https://github.com/thowell/CALIPSO.jl/blob/main/src/solver/solver.jl, there is an enormous delay between the call to the function and the first line in the function getting executed. In other words, if I add a print statement to line 112, of solver.jl, and add a print statement to line 78 of https://github.com/thowell/CALIPSO.jl/blob/main/src/solver/cones/cone.jl, there is a large delay between these prints. When I use an acrobot, the delay is finite, and the second print eventually gets executed. For the underwater snake robot I'm considering, the second print never executes, my computer starts slowing down, and I have to kill Julia externally.

In summary, do you know why there would be a (potentially computer-crashing) delay between line 113 of solver.jl and line 79 of cone.jl? I'm able to solve problems without cones without any issue.
