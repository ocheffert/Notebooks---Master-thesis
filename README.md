# Notebooks of our Master thesis

## Description

The aim of the thesis is to develop and analyze solvers for Differential Algebraic Equations (DAEs) systems of equations.
Through those notebooks, we show how to solve classical physicial, electrical and chemical problems with 4 types of solvers thta we have implemebted or modified.

## Usage

Determining the index of your problem is crucial in order to solve it. All solvers are able to solve index 1 systems however RadauIIa can solve high index problems (index 2 and 3) under a particular form.
Radau will achieve a high accuracy but may be solwer in some cases whereas Rosenbrock solvers are efficient for index 1 problem with a low tolerance.

## Current examples

- The simple pendulum problem (`pendulum.ipynb`) shows how to deal with a high index DAEs system. Many constrained mechanical problems
are naturally described by a system of index 3 DAEs. A robotic manipulator of 2 linked arms is also a good example of a high index DAEs system and is described in `robot.ipynb`.
- `transistor.ipynb` shows how to model a transistor amplifier with a index 1 DAEs system. We show it suffices to write the basic law of electricity to solve the problem unlike the ODEs form which must be derived by mathematical developments.
- `robertson.ipynb` shows how to solve the Robertson problem with a index 1 DAEs system. This problem is a stiff one and the solvers can be compared based on their ability to solve stiff problems.
- `cstr.ipynb` shows how to solve a CSTR problem with a index 1 DAEs system. The CSTR is controlled by a PID controller.

## Structure

You can find our solvers in the `solvers` directory. They follow the `scipy.integrate.solve_ivp` fashion such that they have to be called through this scipy function.
Each notebook shows how to solve a particular problem with each of the 4 problems.
