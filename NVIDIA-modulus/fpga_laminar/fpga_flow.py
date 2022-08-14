from fpga_geometry import *

import sys
import torch
from sympy import Symbol, Eq, Abs, tanh
import numpy as np

import modulus
from modulus.hydra import to_absolute_path, to_yaml, instantiate_arch
from modulus.hydra.config import ModulusConfig
from modulus.csv_utils.csv_rw import csv_to_dict
from modulus.continuous.solvers.solver import Solver
from modulus.continuous.domain.domain import Domain
from modulus.geometry.csg.csg_3d import Box, Channel, Plane
from modulus.continuous.constraints.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.continuous.validator.validator import PointwiseValidator
from modulus.continuous.inferencer.inferencer import PointwiseInferencer
from modulus.continuous.monitor.monitor import PointwiseMonitor
from modulus.key import Key
from modulus.node import Node
from modulus.PDES.navier_stokes import NavierStokes, Curl
from modulus.PDES.basic import NormalDotVec, GradNormal
from modulus.architecture.fully_connected import FullyConnectedArch
from modulus.architecture.fourier_net import FourierNetArch
from modulus.architecture.siren import SirenArch
from modulus.architecture.modified_fourier_net import ModifiedFourierNetArch
from modulus.architecture.dgm import DGMArch

from modulus.distributed.manager import DistributedManager

import os

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))

    # params for simulation
    # fluid params
    nu = 0.02
    rho = 1
    inlet_vel = 1.0
    volumetric_flow = 1.125


    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nu, rho=rho, dim=3, time=False)
    normal_dot_vel = NormalDotVec()
    equation_nodes = ns.make_nodes() + normal_dot_vel.make_nodes()


    # determine inputs outputs of the network
    input_keys = [Key("x"), Key("y"), Key("z")]
    if cfg.custom.exact_continuity:
        c = Curl(("a", "b", "c"), ("u", "v", "w"))
        equation_nodes += c.make_nodes()
        output_keys = [Key("a"), Key("b"), Key("c"), Key("p")]
    else:
        output_keys = [Key("u"), Key("v"), Key("w"), Key("p")]


    # select the network and the specific configs
    if cfg.custom.arch == "FullyConnectedArch":
        flow_net = FullyConnectedArch(
            input_keys=input_keys,
            output_keys=output_keys,
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    elif cfg.custom.arch == "FourierNetArch":
        flow_net = FourierNetArch(
            input_keys=input_keys,
            output_keys=output_keys,
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    elif cfg.custom.arch == "SirenArch":
        flow_net = SirenArch(
            input_keys=input_keys,
            output_keys=output_keys,
            normalization={"x": (-2.5, 2.5), "y": (-2.5, 2.5), "z": (-2.5, 2.5)},
        )
    elif cfg.custom.arch == "ModifiedFourierNetArch":
        flow_net = ModifiedFourierNetArch(
            input_keys=input_keys,
            output_keys=output_keys,
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    elif cfg.custom.arch == "DGMArch":
        flow_net = DGMArch(
            input_keys=input_keys,
            output_keys=output_keys,
            layer_size=128,
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    else:
        sys.exit(
            "Network not configured for this script. Please include the network in the script"
        )


    flow_nodes = equation_nodes + [flow_net.make_node(name="flow_network", jit=cfg.jit)]


    # make flow domain
    flow_domain = Domain()


    # inlet
    constraint_inlet = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=inlet,
        outvar={"u": inlet_vel, "v": 0, "w": 0},
        batch_size=cfg.batch_size.inlet,
        criteria=Eq(x, channel_origin[0]),
        lambda_weighting={"u": channel.sdf, "v": 1.0, "w": 1.0},  # weight zero on edges
        quasirandom=cfg.custom.quasirandom,
    )
    flow_domain.add_constraint(constraint_inlet, "inlet")

    # outlet
    constraint_outlet = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_origin[0] + channel_dim[0]),
        quasirandom=cfg.custom.quasirandom,
    )
    flow_domain.add_constraint(constraint_outlet, "outlet")

    # no slip for channel walls
    no_slip = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
        quasirandom=cfg.custom.quasirandom,
    )
    flow_domain.add_constraint(no_slip, "no_slip")



    # flow interior low res away from fpga
    lr_interior = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=lr_geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        bounds=lr_bounds,
        batch_size=cfg.batch_size.lr_interior,
        lambda_weighting={
            "continuity": geo.sdf,
            "momentum_x": geo.sdf,
            "momentum_y": geo.sdf,
            "momentum_z": geo.sdf,
        },
        quasirandom=cfg.custom.quasirandom,
    )
    flow_domain.add_constraint(lr_interior, "lr_interior")



    # flow interiror high res near fpga
    hr_interior = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=hr_geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_z": 0, "momentum_y": 0},
        bounds=hr_bounds,
        batch_size=cfg.batch_size.hr_interior,
        lambda_weighting={
            "continuity": geo.sdf,
            "momentum_x": geo.sdf,
            "momentum_y": geo.sdf,
            "momentum_z": geo.sdf,
        },
        quasirandom=cfg.custom.quasirandom,
    )
    flow_domain.add_constraint(hr_interior, "hr_interior")



    # integral continuity
    integral_continuity = IntegralBoundaryConstraint(
        nodes=flow_nodes,
        geometry=integral_plane,
        outvar={"normal_dot_vel": volumetric_flow},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        criteria=geo.sdf > 0,
        lambda_weighting={"normal_dot_vel": 1.0},
        param_ranges=x_pos_range,
        quasirandom=cfg.custom.quasirandom,
    )
    flow_domain.add_constraint(integral_continuity, "integral_continuity")



    # flow data
    mapping = {
        "Points:0": "x",
        "Points:1": "y",
        "Points:2": "z",
        "U:0": "u",
        "U:1": "v",
        "U:2": "w",
        "p_rgh": "p",
    }

    openfoam_var = csv_to_dict(
        to_absolute_path("../openfoam/fpga_heat_fluid0.csv"), mapping
    )
    openfoam_var["x"] = openfoam_var["x"] + channel_origin[0]
    openfoam_var["y"] = openfoam_var["y"] + channel_origin[1]
    openfoam_var["z"] = openfoam_var["z"] + channel_origin[2]
    openfoam_invar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["x", "y", "z"]
    }
    openfoam_outvar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["u", "v", "w", "p"]
    }


    # remove validator by Yunchao
    #openfoam_validator = PointwiseValidator(
    #    openfoam_invar_numpy, openfoam_outvar_numpy, flow_nodes
    #)
    #flow_domain.add_validator(openfoam_validator)


    # add pressure monitor
    invar_front_pressure = integral_plane.sample_boundary(
        1024,
        param_ranges={
            x_pos: heat_sink_base_origin[0] - heat_sink_base_dim[0],
        },
    )

    pressure_monitor = PointwiseMonitor(
        invar_front_pressure,
        output_names=["p"],
        metrics={"front_pressure": lambda var: torch.mean(var["p"])},
        nodes=flow_nodes,
    )

    flow_domain.add_monitor(pressure_monitor)
    invar_back_pressure = integral_plane.sample_boundary(
        1024,
        param_ranges={
            x_pos: heat_sink_base_origin[0] + 2 * heat_sink_base_dim[0],
        },
    )
    pressure_monitor = PointwiseMonitor(
        invar_back_pressure,
        output_names=["p"],
        metrics={"back_pressure": lambda var: torch.mean(var["p"])},
        nodes=flow_nodes,
    )
    flow_domain.add_monitor(pressure_monitor)

    # add inferencer data
    inference = PointwiseInferencer(
        geo.sample_interior(5000, bounds=lr_bounds), ["u", "v", "p"], flow_nodes
    )
    flow_domain.add_inferencer(inference, "inf_data")

    # make solver
    flow_slv = Solver(cfg, flow_domain)

    # start flow solver
    flow_slv.solve()


if __name__ == "__main__":

    #ngpus_per_node = torch.cuda.device_count()

    #if "RANK" in os.environ:
    #    rank = int(os.environ.get("RANK"))
    #    if "LOCAL_RANK" in os.environ:
    #        local_rank = int(os.environ.get("LOCAL_RANK"))
    #    else:
    #        local_rank = rank % torch.cuda.device_count()
    #        
    #    print("ENV local_rank= ", local_rank)

    #print("SLURM_LOCALID=", int(os.environ.get("SLURM_LOCALID")))

    # Initialize the singleton
    #DistributedManager.initialize()

    
    # Get a manager object
    #manager = DistributedManager()    

    # Parallel attributes
    #print("rank", manager.rank)
    #print("local_rank=", manager.local_rank)
    #print("world_size=", manager.world_size)
    #print("device=", manager.device)

    run()
