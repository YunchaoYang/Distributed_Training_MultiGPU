from sympy import Symbol, Eq, Abs

import modulus
from modulus.hydra import to_absolute_path, to_yaml, instantiate_arch
from modulus.hydra.config import ModulusConfig
from modulus.csv_utils.csv_rw import csv_to_dict
from modulus.continuous.solvers.solver import Solver
from modulus.continuous.domain.domain import Domain
from modulus.geometry.csg.csg_2d import Rectangle

from modulus.continuous.constraints.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.continuous.validator.validator import PointwiseValidator
from modulus.continuous.inferencer.inferencer import PointwiseInferencer
from modulus.key import Key
from modulus.PDES.navier_stokes import NavierStokes
from modulus.tensorboard_utils.plotter import ValidatorPlotter, InferencerPlotter
from modulus.architecture import layers

import numpy as np

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))

    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network", jit=cfg.jit)]

    # add constraints to solver
    # make geometry
    height = 0.1
    width = 0.1
    x, y = Symbol("x"), Symbol("y")
    rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

    # make ldc domain
    ldc_domain = Domain()

    # top wall
    top_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 1.0, "v": 0},
        batch_size=cfg.batch_size.TopWall,
        lambda_weighting={"u": 1.0 - 20 * Abs(x), "v": 1.0},  # weight edges to be zero
        criteria=Eq(y, height / 2),
    )
    ldc_domain.add_constraint(top_wall, "top_wall")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.NoSlip,
        criteria=y < height / 2,
    )
    ldc_domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.Interior,
        bounds={x: (-width / 2, width / 2), y: (-height / 2, height / 2)},
        lambda_weighting={
            "continuity": rec.sdf,
            "momentum_x": rec.sdf,
            "momentum_y": rec.sdf,
        },
    )
    ldc_domain.add_constraint(interior, "interior")

    # add validator
    # remove the validator section, Yunchao 
    # mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
    # openfoam_var = csv_to_dict(
    #     to_absolute_path("openfoam/cavity_uniformVel0.csv"), mapping
    # )
    # openfoam_var["x"] += -width / 2  # center OpenFoam data
    # openfoam_var["y"] += -height / 2  # center OpenFoam data
    # openfoam_invar_numpy = {
    #     key: value for key, value in openfoam_var.items() if key in ["x", "y"]
    # }
    # openfoam_outvar_numpy = {
    #     key: value for key, value in openfoam_var.items() if key in ["u", "v"]
    # }
    # openfoam_validator = PointwiseValidator(
    #     openfoam_invar_numpy,
    #     openfoam_outvar_numpy,
    #     nodes,
    #     batch_size=1024,
    #     plotter=ValidatorPlotter(),
    # )
    # ldc_domain.add_validator(openfoam_validator)

    # add inferencer data
    openfoam_invar_numpy = {'x':np.array([[-0.1,-0.1, 0.1, 0.1]]).T, 'y':np.array([[-0.1,0.1,0.1,-0.1]]).T}
    grid_inference = PointwiseInferencer(
        openfoam_invar_numpy,
        ["u", "v", "p"],
        nodes,
        batch_size=1024,
        plotter=InferencerPlotter(),
    )
    ldc_domain.add_inferencer(grid_inference, "inf_data")

    # make solver
    slv = Solver(cfg, ldc_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
