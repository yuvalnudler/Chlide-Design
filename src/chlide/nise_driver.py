import io
import os
import re
import sys
import time
import json
import shutil
import warnings
import subprocess
from typing import *
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings("ignore")

# NOTE: If you move the run_nise*.py script, adjust the NISE_DIRECTORY_PATH
# This script lives in /home/data/yuvaln/nise_bdfp44_v8/ so that the run writes here;
# NISE and its LASErMPNN submodule stay read-only in /home/data/programs/NISE.
NISE_DIRECTORY_PATH = '/home/data/programs/NISE'
# This campaign's own directory -- tools/ and input/ live here, not in the NISE install.
# Two DIFFERENT roots, which the per-campaign copies conflated because the driver lived
# inside the run directory:
#   NISE_RUN_DIR  the campaign's own tree -- input/, relaxed/, logs/. Set per run.
#   TOOLS_DIR     this repo's tools/. One copy, versioned, never per-campaign.
NISE_RUN_DIR = os.environ.get('CHLIDE_RUN_DIR') or str(Path(os.path.abspath(__file__)).parent)
TOOLS_DIR = Path(__file__).resolve().parents[2] / 'tools'
LASER_PATH = str(Path(NISE_DIRECTORY_PATH) / 'LASErMPNN')
sys.path.append(NISE_DIRECTORY_PATH)

import wandb
import torch
import plotly
import numpy as np
import prody as pr
import pandas as pd
from rdkit import Chem
import plotly.express as px
from rdkit.Chem import AllChem
from rdkit_to_params import Params

from LASErMPNN.run_inference import load_model_from_parameter_dict # type: ignore
from LASErMPNN.run_batch_inference import _run_inference, output_protein_structure, output_ligand_structure # type: ignore
from utility_scripts.burial_calc import compute_fast_ligand_burial_mask
from utility_scripts.calc_symmetry_aware_rmsd import _main as calc_rmsd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from ligand_chirality import ligand_config   # noqa: E402  (needs the path insert above)
try:    # only needed for FOLD_ENGINE == "af3"; a Boltz-led run must not require it
    from af3_adapter import af3_json, fold_shards, to_boltz_layout  # noqa: E402
except ImportError:  # pragma: no cover
    af3_json = fold_shards = to_boltz_layout = None


def compute_objective_function(confidence_metrics_dict: dict, objective_function: str) -> float:
    """
    Compute the objective function which will be MAXIMIZED by NISE.

    The input to this function is a dictionary of metrics extracted from Boltz-2x
    including design_ligand_plddt, iptm, and affinity metrics if running with
    boltz2_predict_affinity = True.

    The default behavior is to return design_ligand_plddt, but other metrics 
    and combinations of metrics could be used here.
    """

    # NOTE: add your own objective function in another elif block here!

    if objective_function == 'ligand_plddt':
        return confidence_metrics_dict['design_ligand_plddt']

    elif objective_function == 'iptm':
        return confidence_metrics_dict['iptm']

    elif objective_function == 'pbind':
        return confidence_metrics_dict['affinity_probability_binary']

    elif objective_function == 'ligand_plddt_and_pbind':
        return confidence_metrics_dict['design_ligand_plddt'] + confidence_metrics_dict['affinity_probability_binary']

    elif objective_function == 'iptm_and_pbind':
        return confidence_metrics_dict['iptm'] + confidence_metrics_dict['affinity_probability_binary']

    else:
        raise ValueError(f'Objective_function strategy {objective_function} is not implemented.')


# Boltz ships NO CCD entry for any chlorophyll (CLA/CHL/BCL are all absent from its complete
# 45228-component snapshot, though HEM and B12 are present) because RDKit ETKDG cannot embed an
# Mg-chlorin. Passing the ligand as SMILES therefore forces a disconnected [Mg+2], and Boltz then
# parks the metal ~27 A from the macrocycle, which also breaks NISE's symmetry-aware RMSD (the
# LASErMPNN pose has Mg bonded to 4 N, the Boltz pose does not, so the graphs never match).
#
# Instead we hand Boltz a real component built from claox_userccd.cif by tools/build_boltz_ccd.py
# and loaded from boltz_cache/mols/CLAOX.pkl. Measured: Mg 0.055 A from the N4 centroid, Mg-N
# 2.02-2.11 A, all 46 heavy atoms mapped by calc_rmsd.
# The code must be <= 5 characters or Boltz silently truncates it ('CLA-IDE' -> 'CLA-I', not found).
#
# v5 ligand: chlorophyllide a carrying the 13(2)-hydroxyl (OCD1) that v4's CLA-IDE lacked. Built by
# tools/repair_cla_oxy.py from the CLA-OXY block of main_pipeline/input/chlides_fixed.mmcif, which
# needed six repairs before it was usable -- most importantly it declared NO Mg-N bonds at all,
# leaving the metal a disconnected fragment, which is the exact condition that made Boltz scatter
# the Mg 27 A out of the ring before v4 switched to a custom component.
#
# 'CLAOX' is used rather than a 3-letter code because every candidate was already taken: the CCD
# snapshot holds 45,227 components and 'CLX' in particular is a real one, so writing our molecule
# there would have made Boltz silently load a different ligand.
BOLTZ_LIGAND_CCD = 'CLAOX'


# Axial His, 1-based residue number in the Boltz-renumbered output (= resindex 84 + 1).
AXIAL_HIS_RESNUM = 85
# Declare the His NE2 - Mg contact as a covalent bond so Boltz places the histidine in genuine
# axial contact instead of the 2.939 A non-bonded floor it otherwise never crosses (measured
# identically across 3,186 designs on two scaffolds). Boltz applies a generic covalent length and
# lands at ~1.58 A, which is too short -- the Rosetta relax after each iteration corrects it to
# ~2.29 A, closer to the physical 2.15 A than the unconstrained path's 2.54 A.
# The CCD component is what keeps the metal in the macrocycle: with a SMILES ligand the same
# constraint drags the disconnected Mg 11.4 A out of the chlorin.
# FALSE in v6a, and not merely unused. Boltz got a patched ConnectionsPotential with a 2.0-2.6 A
# two-sided window; AF3 has no equivalent, and per Guy's README you never restrain the His-Mg bond
# you are then scoring (his unrestrained AF3 gave 2.2 A, against 2.65/2.83 A for the bonded
# variant). Coordination control moves entirely to the Rosetta relax gate at coordination_window.
# Set by chlide.loop.run() from the campaign YAML. The module-level defaults are the
# SAFE ones: AF3 cannot express a two-sided metal restraint, so a mis-wired run folds
# unrestrained rather than crushing His-Mg to 1.50 A and flipping 13-2 to R.
FOLD_ENGINE = os.environ.get('CHLIDE_FOLD_ENGINE', 'boltz')
CONSTRAIN_HIS_MG_BOND = os.environ.get('CHLIDE_CONSTRAIN_HIS_MG', '0') == '1'


def get_boltz_yaml_boilerplate(sequence: str, smiles: str, predict_affinity: bool):
    # LIVE for FOLD_ENGINE == 'boltz'. `predict_affinity` adds the `properties: affinity`
    # block that produces P(bind); CONSTRAIN_HIS_MG_BOND adds the His NE2 -- Mg bond that the
    # patched ConnectionsPotential holds in a two-sided 2.0-2.6 A window.
    boltz_yaml = f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {sequence}\n      msa: empty\n  - ligand:\n      id: B\n      ccd: {BOLTZ_LIGAND_CCD}\n"
    if predict_affinity:
        boltz_yaml += "properties:\n  - affinity:\n      binder: B\n"
    if CONSTRAIN_HIS_MG_BOND:
        boltz_yaml += (f"constraints:\n  - bond:\n      atom1: [A, {AXIAL_HIS_RESNUM}, NE2]\n"
                       f"      atom2: [B, 1, MG]\n")
    return boltz_yaml
        

def _input_backbones(dir_path: Path) -> list:
    """Real input backbones, excluding the driver's own derived artifacts.

    sample_sequences() writes `<stem>_fixbeta.pdb` NEXT TO the input backbone -- a copy with
    the b-factor column repurposed to mark fixed-identity residues. That file then looks like
    a second input backbone, so a COLD START in a directory where a previous attempt already
    ran dies with "More than one input backbone not currently supported".

    It only bites on a cold start (the guard checks start_iteration == 0), which is why
    resumes never hit it and why it survived eight campaigns: nobody restarted iteration 0 in
    a used directory until a smoke test did it twice in five minutes.
    """
    return sorted(x for x in dir_path.iterdir()
                  if x.is_file() and x.suffix == '.pdb' and not x.stem.endswith('_fixbeta'))


def check_input(dir_path: Path, model_weights_path: Path):
    if not dir_path.exists():
        raise FileNotFoundError(f"Path {dir_path} does not exist.")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path {dir_path} is not a directory.")
    
    num_files = len(_input_backbones(dir_path))
    if num_files == 0:
        raise FileNotFoundError(f"No PDB files found in {dir_path}")

    if not model_weights_path.exists():
        raise FileNotFoundError(f"Model weights file {model_weights_path} does not exist.")


def handle_directory_creation(input_dir, model_checkpoint):
    input_backbones_path = input_dir / 'input_backbones'
    check_input(input_backbones_path, model_checkpoint)

    sampling_dataframe_path = input_dir / 'sampling_dataframes'
    sampled_backbones_path = input_dir / 'sampled_backbones'
    sampling_dataframe_path.mkdir(exist_ok=True)
    sampled_backbones_path.mkdir(exist_ok=True)

    sdf_path = input_dir / 'lig_from_input_pdb.sdf'
    params_path = input_dir / 'lig_from_input_pdb.params'

    return input_backbones_path, sampling_dataframe_path, sampled_backbones_path, sdf_path, params_path


def construct_helper_files(sdf_path, params_path, backbone_path, ligand_smiles):

    input_protein = pr.parsePDB(str(backbone_path))
    assert isinstance(input_protein, pr.AtomGroup), f"Error loading backbone {backbone_path}"

    ligand = input_protein.select('not protein').copy()
    assert ligand.numResidues() == 1, f"Error selecting ligand from backbone {backbone_path}"

    # Create a pdb string for the ligand.
    pdb_stream = io.StringIO()
    pr.writePDBStream(pdb_stream, ligand)
    ligand_string = pdb_stream.getvalue()

    lignames_set = set(ligand.getResnames())

    # Write ligand to PDB file.
    pdb_path = Path(str(sdf_path.resolve()).rsplit('.')[0] + '.pdb')
    with pdb_path.open('w') as f:
        f.write(ligand_string)
    
    pdb_mol = Chem.MolFromPDBBlock(ligand_string)
    smi_mol = Chem.MolFromSmiles(ligand_smiles)

    pdb_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, pdb_mol)
    pdb_mol = AllChem.AddHs(pdb_mol, addCoords=True)
    AllChem.ComputeGasteigerCharges(pdb_mol)
    Chem.MolToMolFile(pdb_mol, str(sdf_path.resolve()))

    ligname = lignames_set.pop()
    p = Params.from_mol(pdb_mol, name=ligname)
    p.dump(params_path) # type: ignore

    with open(Path(params_path).parent / '.gitignore', 'w') as f:
        f.write('*')


class DesignCampaign:
    def __init__(self, 
        model_checkpoint, input_dir, ligand_rmsd_mask_atoms, ligand_atoms_enforce_buried, ligand_atoms_enforce_exposed, laser_inference_device, debug, ligand_3lc,
        rmsd_use_chirality, self_consistency_ligand_rmsd_threshold, self_consistency_protein_rmsd_threshold,
        laser_inference_dropout, num_iterations, num_top_backbones_per_round, laser_sampling_params, sequences_sampled_per_backbone, 
        sequences_sampled_at_once, boltz_inference_devices, ligand_smiles, boltz2x_executable_path, 
        use_reduce_protonation, keep_input_backbone_in_queue, keep_best_generator_backbone, use_boltz_conformer_potentials,
        boltz2_predict_affinity, drop_rmsd_mask_atoms_from_ligand_plddt_calc, use_boltz_1x, 
        boltz2_disable_kernels, boltz2_disable_nccl_p2p, objective_function, fixed_identity_residue_indices, 
        align_on_binding_site, burial_mask_alpha_hull_alpha,
        relax_in_loop=False, relax_python=None, relax_params=None, relax_top_n=12,
        relax_workers=12, relax_r0=2.15, relax_timeout=1800,
        coordination_window=(2.0, 2.6),
        axial_angle_max=11.5, lonepair_angle_max=17.0, mg_imidazole_oop_max=12.7,
        **kwargs
    ):
        self.debug = debug
        self.relax_in_loop = relax_in_loop
        self.relax_python = relax_python
        self.relax_params = relax_params
        self.relax_top_n = relax_top_n
        self.relax_workers = relax_workers
        self.relax_r0 = relax_r0
        self.relax_timeout = relax_timeout
        self.coordination_window = tuple(coordination_window)
        # Coordination GEOMETRY, checked before promotion. The distance window above says the
        # His is close enough; these say the bond is axial and the imidazole is turned the
        # right way. Defaults are the experimental p90 (docs/CALIBRATION.md).
        self.axial_angle_max = float(axial_angle_max)
        self.lonepair_angle_max = float(lonepair_angle_max)
        self.mg_imidazole_oop_max = float(mg_imidazole_oop_max)
        self._geom_rejected = 0
        self._geom_promoted = 0
        self._geom_fallback_pool = []
        self._relax_round = 0
        if relax_in_loop:
            print(f'RELAX in loop: top {relax_top_n} candidates, His-Mg window '
                  f'{self.coordination_window[0]}-{self.coordination_window[1]} A, r0={relax_r0}')
        print(f'GEOMETRY gate: axial <= {self.axial_angle_max} deg, lone-pair <= '
              f'{self.lonepair_angle_max} deg, Mg-off-imidazole <= {self.mg_imidazole_oop_max} deg')
        self.ligand_3lc = ligand_3lc
        self.boltz_inference_devices = boltz_inference_devices
        self.use_boltz_conformer_potentials = use_boltz_conformer_potentials
        self.use_boltz_1x = use_boltz_1x
        self.boltz2x_executable_path = boltz2x_executable_path
        self.predict_affinity = boltz2_predict_affinity
        self.use_reduce_protonation = use_reduce_protonation
        self.boltz2_disable_kernels = boltz2_disable_kernels
        self.boltz2_disable_nccl_p2p = boltz2_disable_nccl_p2p 
        self.objective_function = objective_function
        self.align_on_binding_site = align_on_binding_site
        self.fixed_identity_residue_indices = fixed_identity_residue_indices

        if self.fixed_identity_residue_indices is not None:
            print(f'Fixing residues: {self.fixed_identity_residue_indices}')

        if self.use_boltz_1x and self.predict_affinity:
            raise ValueError('Cannot use boltz1x with affinity prediction.')

        if 'pbind' in self.objective_function and (not self.predict_affinity):
            raise ValueError(f"predict_affinity must be True to use objective function {self.objective_function}")

        self.rmsd_use_chirality = rmsd_use_chirality
        self.self_consistency_ligand_rmsd_threshold = self_consistency_ligand_rmsd_threshold
        self.self_consistency_protein_rmsd_threshold = self_consistency_protein_rmsd_threshold
        self.keep_input_backbone_in_queue = keep_input_backbone_in_queue 
        self.drop_rmsd_mask_atoms_from_ligand_plddt_calc = drop_rmsd_mask_atoms_from_ligand_plddt_calc

        self.num_iterations = num_iterations
        self.sequences_sampled_per_backbone = sequences_sampled_per_backbone
        self.steady_sequences_per_backbone = sequences_sampled_per_backbone
        self.seed_sequences_per_backbone = kwargs.pop('seed_sequences_per_backbone', None)
        if self.seed_sequences_per_backbone:
            # Start wide; narrow once the queue holds top_k gate-passing backbones.
            self.sequences_sampled_per_backbone = self.seed_sequences_per_backbone
        self.sequences_sampled_at_once = sequences_sampled_at_once
        self.top_k = num_top_backbones_per_round

        self.ligand_rmsd_mask_atoms = ligand_rmsd_mask_atoms
        self.ligand_atoms_enforce_buried = ligand_atoms_enforce_buried 
        self.ligand_atoms_enforce_exposed = ligand_atoms_enforce_exposed 
        self.burial_mask_alpha_hull_alpha = burial_mask_alpha_hull_alpha
        self.ligand_smiles = ligand_smiles

        self.laser_sampling_params = laser_sampling_params
        self.laser_inference_dropout = laser_inference_dropout
        
        self.sampling_metadata = defaultdict(float)
        self.sampling_metadata['min_protein_rmsd'] = float('inf')
        self.sampling_metadata['min_ligand_rmsd'] = float('inf')

        self.input_backbones_path, self.sampling_dataframe_path, self.sampled_backbones_path, self.sdf_path, self.params_path = handle_directory_creation(input_dir, model_checkpoint)
        self.model, self.model_params = load_model_from_parameter_dict(model_checkpoint, torch.device(laser_inference_device))

        # Set model to eval mode, enable inference dropout if specified.
        self.model.eval()
        if self.laser_inference_dropout:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()

        # Set path and priority for input backbones.
        if self.keep_input_backbone_in_queue:
            self.backbone_queue = [(x, torch.inf) for x in _input_backbones(self.input_backbones_path)]
        else:
            self.backbone_queue = [(x, 0) for x in _input_backbones(self.input_backbones_path)]
        
        # Whether to keep the pose that generates the highest confidence sequences.
        self.keep_best_generator_backbone = keep_best_generator_backbone
        self.backbone_to_best_generation = defaultdict(float)

        # ====================================================================================
        # Resume support (not in upstream NISE).
        #
        # Upstream always rebuilds backbone_queue from input_backbones/, so any interruption --
        # a reboot, an OOM, a stray kill -- throws away the whole trajectory and restarts from
        # the original pose. This campaign lost 9 iterations to a host reboot that way.
        #
        # Everything needed to continue is already on disk: each finished iteration leaves both
        # sampling_dataframes/iter_N_data.pkl and sampled_backbones/iter_N/backbone_queue.txt.
        # Requiring BOTH is what makes an iteration "complete" and is what protects a partially
        # written final iteration (the queue file is the last thing an iteration writes).
        # ====================================================================================
        self.start_iteration = 0
        last_complete, last_queue = -1, None
        for qf in self.sampled_backbones_path.glob('iter_*/backbone_queue.txt'):
            m = re.match(r'iter_(\d+)$', qf.parent.name)
            if not m:
                continue
            n = int(m.group(1))
            if not (self.sampling_dataframe_path / f'iter_{n}_data.pkl').exists():
                continue
            if n > last_complete:
                last_complete, last_queue = n, qf

        if last_queue is not None:
            restored = []
            for line in last_queue.read_text().splitlines():
                if not line.strip():
                    continue
                score, _, path = line.partition('\t')
                p = Path(path.strip())
                if p.is_file():
                    restored.append((p, float(score)))
                else:
                    print(f'RESUME: dropping missing backbone {p}')
            if restored:
                self.backbone_queue = restored
                self.start_iteration = last_complete + 1
                # Seed the generator bookkeeping so keep_best_generator_backbone behaves as it
                # would have, rather than treating every restored backbone as unproven.
                for p, s in restored:
                    self.backbone_to_best_generation[str(p)] = max(
                        self.backbone_to_best_generation[str(p)], s)
                print(f'RESUME: iterations 0-{last_complete} already complete; continuing at '
                      f'iteration {self.start_iteration} with {len(restored)} backbone(s) from '
                      f'{last_queue}')
            else:
                print(f'RESUME: {last_queue} had no usable backbones; starting from iteration 0')

        # If the run already converged before this restart, say so and do not burn a ~2-hour
        # iteration rediscovering it. This is what makes the stop sticky across a crash/resume.
        self.already_converged = False
        if self.start_iteration > 0:
            cs = self.convergence_state(self.start_iteration - 1)
            if cs['stop'] and self.CONVERGENCE_AUTOSTOP:
                self.already_converged = True
                print(f'CONVERGED BEFORE RESTART: {cs["reason"]}\n'
                      f'  best {cs["best"]:.4f} at iteration {cs["best_iteration"]}; '
                      f'not starting iteration {self.start_iteration}.')

        # Only meaningful for a cold start -- a resumed queue legitimately holds up to top_k poses.
        if self.start_iteration == 0 and len(self.backbone_queue) > 1:
            raise NotImplementedError(f"More than one input backbone not currently supported.")

        construct_helper_files(self.sdf_path, self.params_path, self.backbone_queue[0][0], ligand_smiles)
        self.ligand_rmsd_mask_atoms = ligand_rmsd_mask_atoms

        assert self.sdf_path.exists(), f"Error creating SDF file {self.sdf_path}"
        assert self.params_path.exists(), f"Error creating params file {self.params_path}"
    
    def sample_sequences(self, backbone_path: str) -> Tuple[List[pr.AtomGroup], List[str], List[float], List[float]]:
        laser_nll, laser_bs_nll = [], []
        sampled_proteins, sampled_sequences = [], []
        num_sampled = 0
        while num_sampled < self.sequences_sampled_per_backbone:
            remaining_samples = self.sequences_sampled_per_backbone - num_sampled
            num_seq_to_sample = min(remaining_samples, self.sequences_sampled_at_once)

            backbone_path_ = backbone_path
            if self.fixed_identity_residue_indices is not None:
                # NOTE: there might be a better way to do this so that we don't lose information stored in the b-factor columns of backbones that we sample, 
                # but the important pLDDT information should be recorded in the log steps anyways.
                protein = pr.parsePDB(str(backbone_path))
                protein.setBetas(0.0)
                fixed_selection = protein.select(self.fixed_identity_residue_indices)
                if fixed_selection is None:
                    raise ValueError(f'ProDy encounted an error selecting residues with selection string: {self.fixed_identity_residue_indices}')
                fixed_selection.setBetas(1.0)
                backbone_path_ = Path(backbone_path).parent / f'{Path(backbone_path).stem}_fixbeta.pdb'
                pr.writePDB(str(backbone_path_), protein)
                self.laser_sampling_params.update({'fix_beta': True})

            sampling_output, full_atom_coords, nh_coords, sampled_probs, batch_data, protein_complex_data = _run_inference(self.model, self.model_params, Path(backbone_path_), num_seq_to_sample, **self.laser_sampling_params)
            protein_complex_data = protein_complex_data[0] # type: ignore
            for idx in range(num_seq_to_sample):
                curr_batch_mask = batch_data.batch_indices == idx

                curr_probs = sampled_probs[curr_batch_mask]
                nll = (-1 * torch.log10(curr_probs)).cpu().numpy().mean()
                bs_nll = (-1 * torch.log10(sampled_probs[curr_batch_mask][batch_data.first_shell_ligand_contact_mask[curr_batch_mask]])).cpu().numpy().mean()

                out_prot = output_protein_structure(full_atom_coords[curr_batch_mask], sampling_output.sampled_sequence_indices[curr_batch_mask], protein_complex_data.residue_identifiers, nh_coords[curr_batch_mask], curr_probs)
                out_lig = output_ligand_structure(protein_complex_data.ligand_info)

                out_complex = out_prot + out_lig
                out_complex.setTitle('LASErMPNN/NISE Generated Protein')
                sampled_proteins.append(out_complex)
                sampled_sequences.append(out_prot.ca.getSequence())

                laser_nll.append(nll)
                laser_bs_nll.append(bs_nll)

            num_sampled += num_seq_to_sample
        
        return sampled_proteins, sampled_sequences, laser_nll, laser_bs_nll

    def update_sampling_width(self, iidx: int) -> None:
        """Pick this iteration's per-backbone sample count.

        Seed phase while the queue holds fewer than top_k backbones; every entry in the queue
        passed the geometry gate to get there, so "queue is full" is the same statement as
        "we have top_k good candidates". No retry machinery: if an iteration does not fill the
        queue, the next simply runs at seed width again.
        """
        if not self.seed_sequences_per_backbone:
            return
        want = (self.seed_sequences_per_backbone if len(self.backbone_queue) < self.top_k
                else self.steady_sequences_per_backbone)
        if want != self.sequences_sampled_per_backbone:
            print(f'SAMPLING: iteration {iidx} -- {self.sequences_sampled_per_backbone} -> '
                  f'{want} sequences per backbone ({len(self.backbone_queue)} of '
                  f'{self.top_k} backbones in the queue)', flush=True)
            self.sequences_sampled_per_backbone = want

    def coordination_geometry(self, prot_sel, lig_sel) -> dict:
        """Axial-coordination geometry of a PREDICTION, before any relax.

        The distance window the relax gate uses is saturated by the shortlist stage and
        discriminates nothing; this says whether the bond is axial and whether the imidazole
        is turned the right way. v9 walked its histidine from 15 to 40 degrees over eight
        iterations with nothing watching, because a distance gate cannot see a rotation.

        Pure numpy on atoms already in memory -- no Rosetta, no re-parsing. Returns {} rather
        than raising if an atom is missing: this runs on every design and must never kill an
        iteration.
        """
        try:
            from metal_geometry import geometry, MG_NAMES, PYRROLE_NAMES
            lnames = list(lig_sel.getNames())
            lxyz = lig_sel.getCoords()
            idx = {n: i for i, n in enumerate(lnames)}
            pyr = next((g for g in PYRROLE_NAMES if all(x in idx for x in g)), None)
            mgn = next((x for x in MG_NAMES if x in idx), None)
            if pyr is None or mgn is None:
                return {}
            his = prot_sel.select(f'resnum {AXIAL_HIS_RESNUM}')
            if his is None:
                return {}
            hat = {n: c for n, c in zip(his.getNames(), his.getCoords())}
            if not all(a in hat for a in ('NE2', 'CE1', 'CD2')):
                return {}
            return geometry([lxyz[idx[x]] for x in pyr], lxyz[idx[mgn]],
                            hat['NE2'], hat['CE1'], hat['CD2'])
        except Exception:
            return {}

    def geometry_ok(self, geo: dict) -> bool:
        """True when the prediction's coordination is inside the experimental p90 bounds.

        An empty dict (atoms missing) passes: a measurement failure must not silently become a
        rejection -- that would be a gate firing for the wrong reason, which is the exact
        failure mode this whole gate exists to prevent.
        """
        if not geo:
            return True
        return (geo.get('axial_angle', 0.0) <= self.axial_angle_max
                and geo.get('his_lonepair_angle', 0.0) <= self.lonepair_angle_max
                and geo.get('mg_imidazole_oop', 0.0) <= self.mg_imidazole_oop_max)

    def relax_and_gate(self, ranked: list) -> list:
        """Rosetta-relax the leading candidates and keep those that actually coordinate the metal.

        Boltz cannot express the axial bond: unconstrained it floors at 2.939 A, and with the
        covalent bond constraint it overshoots to ~1.58 A. Neither is physical, and ligand pLDDT
        is uncorrelated with coordination (r = 0.039), so nothing in the Boltz output can select
        for it. A FlatHarmonic restraint at 2.15 A over ref2015 settles both cases into the
        physical window in ~25 s, and the post-relax distance -- how close the pocket lets the
        His sit -- is the discriminating measurement.

        Only the leading candidates are relaxed: just top_k of them become backbones, so relaxing
        all several hundred promoted designs would be wasted. The relaxed structure replaces the
        Boltz pose in the queue, so the next iteration is designed against real coordination.

        Fails soft in every direction -- a relax crash, a missing tool, or an empty survivor set
        all fall back to the unrelaxed ranking rather than emptying the queue and freezing the
        campaign.
        """
        head, tail = ranked[:self.relax_top_n], ranked[self.relax_top_n:]
        paths = [str(p) for p, _ in head]
        tool = TOOLS_DIR / 'rosetta_relax.py'
        if not tool.is_file() or not Path(self.relax_params).is_file():
            print(f'RELAX: missing {tool} or {self.relax_params}; skipping')
            return ranked

        # FIXED absolute output directory. The previous version derived it as
        # paths[0].parent.parent.parent, but a relaxed structure sits one level shallower than the
        # Boltz output it came from, so every generation of re-relaxing climbed a directory:
        # iter_N/boltz_results_boltz_inputs/relaxed -> iter_N/relaxed ->
        # input/sampled_backbones/relaxed -> /home/data/relaxed -> /home/relaxed, which is not
        # writable. Relax then failed silently for the rest of the campaign.
        self._relax_round += 1
        out_dir = Path(NISE_RUN_DIR) / 'relaxed' / f'round_{self._relax_round:03d}'
        cmd = [self.relax_python, str(tool), '--params', str(self.relax_params),
               '--out-dir', str(out_dir), '--workers', str(self.relax_workers),
               '--his-index', str(AXIAL_HIS_RESNUM), '--r0', str(self.relax_r0)] + paths
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.relax_timeout)
        except Exception as e:
            print(f'RELAX: failed to run ({type(e).__name__}: {e}); keeping unrelaxed ranking')
            return ranked

        by_input, errors = {}, []
        for line in proc.stdout.splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('ok'):
                by_input[r['input']] = r
            elif r.get('error'):
                errors.append(r['error'])
        if not by_input:
            # Be loud. The old message printed an empty stderr and hid a mkdir permission error
            # for five iterations; the per-design 'error' field is where the real cause lives.
            print(f'RELAX: no successful relaxations out of {len(paths)}. '
                  f'first errors: {errors[:2]}. rc={proc.returncode}. '
                  f'stderr: {proc.stderr.strip()[:300]}')
            return ranked

        lo, hi = self.coordination_window
        kept, rejected = [], []
        for (p, score) in head:
            r = by_input.get(str(p))
            if r is None:
                rejected.append((str(p), 'relax failed'))
                continue
            d = r['his_mg']
            if lo <= d <= hi:
                # Queue the PREDICTION, not the relaxed pose. The relax is a GATE here, not a
                # source of coordinates: it proves the His-Mg distance is physically reachable,
                # and that verdict is the only thing carried forward. v5 seeded r['output'], and
                # the 0.74 A backbone displacement the relax introduces then consumed half of the
                # next round's 1.5 A self-consistency budget before folding began. The paper
                # expands "the top three self-consistent structures by ligand confidence" -- the
                # predicted structures.
                kept.append((p, score))
                self.backbone_to_best_generation[str(p)] = max(
                    self.backbone_to_best_generation[str(p)], float(score))
            else:
                rejected.append((str(Path(p).parent.name), f'His-Mg {d:.2f} outside {lo}-{hi}'))
        print(f'RELAX: {len(kept)}/{len(head)} kept '
              f'(His-Mg in {lo}-{hi} A); ' +
              '; '.join(f'{n}: {why}' for n, why in rejected[:3]))
        if not kept:
            # Never let the coordination gate empty the queue -- that is how a campaign freezes.
            print('RELAX: nothing passed the coordination gate; falling back to unrelaxed ranking')
            return ranked
        return kept + tail

    def identify_backbone_candidates(self, sorted_designs_boltz: Sequence[Path], sorted_designs_laser: Sequence[Path], sampled_backbone_paths: Sequence[Path], reduce_executable_path, reduce_hetdict_path):
        assert len(sorted_designs_laser) == len(sorted_designs_boltz), f"Error: different number of designs in" # {boltz_output_subdir} and {laser_output_subdir}."
        smi_mol = Chem.MolFromSmiles(self.ligand_smiles)

        log_data = defaultdict(list)
        for laser, boltz, bb_path in zip(sorted_designs_laser, sorted_designs_boltz, sampled_backbone_paths):
            laser_prot = pr.parsePDB(str(laser))
            boltz_string_str = open(boltz, 'r').read()
            boltz_string_io = io.StringIO(boltz_string_str)
            boltz_prot = pr.parsePDBStream(boltz_string_io)

            confidence_data = {'laser_output_pdb_path': laser, 'boltz_output_pdb_path': boltz}
            with (boltz.parent / f'confidence_{boltz.stem}.json').open('r') as f:
                confidence_data.update(json.load(f))

            design_iptm = confidence_data['iptm']
            design_bind_probability, design_predicted_affinity = torch.nan, torch.nan
            if self.predict_affinity:
                with (boltz.parent / f'affinity_{boltz.stem.replace("_model_0", "")}.json').open('r') as f:
                    confidence_data.update(json.load(f))
                    design_bind_probability = confidence_data['affinity_probability_binary']
                    design_predicted_affinity = confidence_data['affinity_pred_value']

            log_data['affinity_probability_binary'].append(design_bind_probability)
            log_data['affinity_pred_value'].append(design_predicted_affinity)
            log_data['iptms'].append(design_iptm)

            # 13-2 stereocentre gate. See tools/ligand_chirality.py for why this is a targeted
            # per-centre check rather than rmsd_use_chirality=True (which rejects the correct
            # epimer too and freezes the run). An inverted epimer is a different molecule from
            # the one being designed for, and under a chirality-blind RMSD it scores BETTER than
            # the correct one -- 0.31 A vs 1.52 A on v5's two iteration-15 populations, because a
            # mirrored atom match fits more closely. So it has to be removed before the ranking,
            # not ranked by it. An unverifiable centre (None) is treated as a failure, never as a
            # pass. Forcing inf rather than adding a `continue` keeps every log_data column the
            # same length on every path.
            ligand_cfg = ligand_config(boltz_prot, self.ligand_smiles)
            log_data['ligand_config'].append(ligand_cfg)

            try:
                protein_rmsd, ligand_rmsd, laser_to_boltz_name_mapping = calc_rmsd(laser_prot, boltz_prot, self.ligand_smiles, self.rmsd_use_chirality, self.ligand_rmsd_mask_atoms, align_on_binding_site=self.align_on_binding_site)
                if len(laser_to_boltz_name_mapping) == 0:
                    raise ValueError('No atoms in common between laser and boltz structures.')
                boltz_to_laser_name_mapping = {v: k for k, v in laser_to_boltz_name_mapping.items()}
            except:
                protein_rmsd = np.nan
                ligand_rmsd = np.nan
                laser_to_boltz_name_mapping = {}

            if ligand_cfg != self.REQUIRED_LIGAND_CONFIG:
                ligand_rmsd = float('inf')

            log_data['ligand_rmsds'].append(ligand_rmsd)
            log_data['protein_rmsds'].append(protein_rmsd)

            if len(laser_to_boltz_name_mapping) == 0:
                print(f'{boltz}: Failed to map names between laser and boltz structures.')
                log_data['ligand_is_buried'].append(False)
                log_data['ligand_plddts'].append(torch.nan)
                log_data['protein_plddts'].append(torch.nan)
                # The geometry columns are appended further down, AFTER this continue, so
                # they must be padded here or they come up short exactly as often as name
                # mapping fails -- which is what crashed v11a's first iteration at
                # pd.DataFrame(iidx_data) with "All arrays must be of the same length".
                for _k in ('axial_angle', 'his_lonepair_angle', 'mg_imidazole_oop',
                           'mg_out_of_plane', 'n4_planarity'):
                    log_data[_k].append(float('nan'))
                log_data['geometry_ok'].append(False)
                continue

            # Remap the boltz structure ligand atoms with the name mapping.
            boltz_prot_only = boltz_prot.select('chid A')
            boltz_lig_only = boltz_prot.select('chid B and not element H')
            boltz_lig_only.setNames([boltz_to_laser_name_mapping[x] for x in boltz_lig_only.getNames()])
            boltz_lig_only.setResnames([self.ligand_3lc for _ in range(len(boltz_lig_only.getResnames()))])
            boltz_coords = boltz_lig_only.getCoords()

            atoms_enforced_buried_mask = np.array([x in self.ligand_atoms_enforce_buried for x in boltz_lig_only.getNames()])
            atoms_enforced_exposed_mask = np.array([x in self.ligand_atoms_enforce_exposed for x in boltz_lig_only.getNames()])

            pdb_output_path = str(boltz)
            pr.writePDB(pdb_output_path, boltz_prot_only + boltz_lig_only)

            # Compute ligand pLDDT over relevant atoms.
            rmsd_mask = np.array([x not in self.ligand_rmsd_mask_atoms for x in boltz_lig_only.getNames()])
            if not self.drop_rmsd_mask_atoms_from_ligand_plddt_calc:
                rmsd_mask = np.ones_like(rmsd_mask)
            design_ligand_plddt = boltz_lig_only.getBetas()[rmsd_mask].mean() / 100
            design_protein_plddt = boltz_prot_only.copy().ca.getBetas().mean() / 100

            log_data['ligand_plddts'].append(design_ligand_plddt)
            log_data['protein_plddts'].append(design_protein_plddt)
            confidence_data['design_ligand_plddt'] = design_ligand_plddt
            confidence_data['design_protein_plddt'] = design_protein_plddt

            # Check ligand burial constraints are obeyed in predicted structure.
            all_buried_mask = compute_fast_ligand_burial_mask(boltz_prot.ca.getCoords(), boltz_coords[atoms_enforced_buried_mask], num_rays=5, alpha=self.burial_mask_alpha_hull_alpha)
            none_buried_mask = compute_fast_ligand_burial_mask(boltz_prot.ca.getCoords(), boltz_coords[atoms_enforced_exposed_mask], num_rays=5, alpha=self.burial_mask_alpha_hull_alpha)
            
            # Measured for EVERY design and logged on every path: log_data columns must stay
            # equal length or the per-iteration DataFrame is corrupt (see the chirality note
            # above). The gate below skips only the promotion, never the logging.
            geo = self.coordination_geometry(boltz_prot_only, boltz_lig_only)
            geo_ok = self.geometry_ok(geo)
            for _k in ('axial_angle', 'his_lonepair_angle', 'mg_imidazole_oop',
                       'mg_out_of_plane', 'n4_planarity'):
                log_data[_k].append(geo.get(_k, float('nan')))
            log_data['geometry_ok'].append(geo_ok)

            if (all_buried_mask.all().item() and (not none_buried_mask.any().item())) or self.debug:
                log_data['ligand_is_buried'].append(True)
                if (ligand_rmsd < self.self_consistency_ligand_rmsd_threshold and protein_rmsd < self.self_consistency_protein_rmsd_threshold) or self.debug:

                    if self.use_reduce_protonation:
                        subprocess.run(
                            f'{reduce_executable_path} -DB {reduce_hetdict_path} -DROP_HYDROGENS_ON_ATOM_RECORDS -BUILD {pdb_output_path} > {pdb_output_path}_', 
                            shell=True, check=False, 
                            stdout=subprocess.DEVNULL if not self.debug else subprocess.PIPE, 
                            stderr=subprocess.DEVNULL if not self.debug else subprocess.PIPE
                        )
                        shutil.move(f'{pdb_output_path}_', pdb_output_path)
                    else:
                        # Protonate using RDKit
                        ligand_string = io.StringIO()
                        pr.writePDBStream(ligand_string, boltz_lig_only.copy())
                        pdb_mol = Chem.MolFromPDBBlock(ligand_string.getvalue())
                        pdb_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, pdb_mol)
                        pdb_mol = AllChem.AddHs(pdb_mol, addCoords=True)

                        ligand_prody = pr.parsePDBStream(io.StringIO(Chem.MolToPDBBlock(pdb_mol)))
                        ligand_prody.setResnames(self.ligand_3lc)
                        ligand_prody.setChids('B')
                        pr.writePDB(pdb_output_path, boltz_prot_only.copy() + ligand_prody)

                    score = compute_objective_function(confidence_data, self.objective_function)
                    if geo_ok:
                        self._geom_promoted += 1
                        self.backbone_to_best_generation[bb_path] = max(score, self.backbone_to_best_generation[bb_path])
                        self.backbone_queue.append((pdb_output_path, score))
                    else:
                        # Not promoted, and deliberately NOT recorded in
                        # backbone_to_best_generation either -- that dict feeds the
                        # keep_best_generator_backbone rescue below, which would otherwise
                        # reintroduce exactly what this gate excluded.
                        self._geom_rejected += 1
                        self._geom_fallback_pool.append((pdb_output_path, score))
            else:
                log_data['ligand_is_buried'].append(False)
        
        cfgs = log_data.get('ligand_config', [])
        if cfgs:
            from collections import Counter as _C
            tally = _C('unverified' if c is None else c for c in cfgs)
            n_ok = tally.get(self.REQUIRED_LIGAND_CONFIG, 0)
            print(f'CHIRALITY: {n_ok}/{len(cfgs)} at {self.REQUIRED_LIGAND_CONFIG} '
                  f'({100 * n_ok / len(cfgs):.0f}%); {dict(tally)}', flush=True)
            if n_ok == 0:
                # Either every prediction inverted, or the centre cannot be read at all. The
                # second is a tooling fault that would silently empty the queue every iteration.
                print('CHIRALITY: WARNING -- not one design passed. If these are all '
                      '"unverified", tools/ligand_chirality.py is failing to find the centre, '
                      'not the predictions failing to form it.', flush=True)

        n_geom_rejected, self._geom_rejected = self._geom_rejected, 0
        if n_geom_rejected:
            print(f'GEOMETRY: {n_geom_rejected} design(s) not promoted -- coordination outside '
                  f'axial<={self.axial_angle_max}/lonepair<={self.lonepair_angle_max}/'
                  f'offimid<={self.mg_imidazole_oop_max} deg', flush=True)

        n_geom_promoted, self._geom_promoted = self._geom_promoted, 0
        # The queue CARRIES OVER between iterations, so it is almost never empty -- the real
        # failure is not an empty queue but a stagnant one, where the gate rejects every new
        # candidate and the loop re-designs the same carried-over backbones forever. That is
        # what this tests: nothing NEW got through.
        if n_geom_promoted == 0 and n_geom_rejected:
            print(f'GEOMETRY: *** FALLBACK *** no new design passed the geometry gate this '
                  f'iteration ({n_geom_rejected} rejected). Promoting those on score alone; '
                  f'THIS ITERATION IS UNGATED.', flush=True)
            for _p, _sc in self._geom_fallback_pool:
                self.backbone_queue.append((_p, _sc))
        self._geom_fallback_pool = []
        ranked = sorted(self.backbone_queue, key=lambda x: float(x[1]), reverse=True)
        if self.relax_in_loop and ranked:
            ranked = self.relax_and_gate(ranked)
        self.backbone_queue = ranked[:self.top_k]

        # Adds the backbone which has generated the best scoring pose if it's not in the queue already.
        if self.keep_best_generator_backbone and len(self.backbone_to_best_generation) > 0:
            top_backbone = max(self.backbone_to_best_generation.items(), key=lambda x: x[1])
            print(top_backbone)
            if not (top_backbone[0] in [x[0] for x in self.backbone_queue]):
                self.backbone_queue = self.backbone_queue[:-1] + [top_backbone]

        return sorted_designs_laser, sorted_designs_boltz, log_data
    
    # ========================================================================================
    # Convergence auto-stop
    # ========================================================================================
    # v4 ran a fixed 15 iterations and was still improving when it stopped: p90 ligand pLDDT
    # climbed 0.901 -> 0.948 and the last two iterations were the two best. A fixed count either
    # cuts the run short or burns GPU time after the plateau, so v5 stops on the data instead.
    #
    # The rule: stop when the 90th-percentile ligand pLDDT has failed to beat the best-so-far for
    # PATIENCE consecutive iterations. Two details are load-bearing:
    #
    #   Best-so-far, not previous-iteration. The series is noisy and non-monotonic -- the sign of
    #   s_i - s_{i-1} flips six times in v4's fifteen points -- so a previous-iteration comparison
    #   is measuring noise.
    #
    #   A trailing mean, not the raw value. This buys margin rather than correctness: on v4 the
    #   longest non-improving streak drops from 6 (raw) to 5 (smoothed), i.e. from 1 to 2 short of
    #   firing. Replayed over all four campaigns, neither form stops any run before its best
    #   iteration, so smoothing is a safety margin, not a bug fix. (An earlier note here claimed
    #   the raw form would have wrongly killed nise_6hrk_his117 -- it would not: that campaign's
    #   p90 peaked at iteration 2 and never recovered, so stopping it at 9 was correct.)
    #
    # The decisive calibration is v4: its longest non-improving streak was 5, and it then produced
    # its three best iterations. PATIENCE 5 stops at iteration 9 when the best p90 came at 13, and
    # PATIENCE 3 stops at 7 -- both before the best designs existed. 7 leaves margin 2.
    # Do not lower it. Replay the rule (tools/replay_convergence.py) before changing any constant;
    # it fails loudly if a constant would have stopped any campaign before its peak.
    CONVERGENCE = dict(window=3, eps=0.002, patience=7, min_iterations=10)

    # The published protocol says to keep going PAST the plateau: "performing NISE iterations
    # beyond the plateau in ligand pLDDT or score was important for downstream filtering by
    # additional biophysical metrics. Indeed, both EPIC and APEX came from later iterations."
    # (Nature 656, 237, 2026.) Our own v8 agrees -- it stopped itself at iteration 16 of 25 and
    # half its top-10 by BUNs came from iteration 11 or later.
    #
    # So the auto-stop is a CONVENIENCE, not a correctness feature. With it off the metric is
    # still computed, logged and reported; only the halt is suppressed.
    CONVERGENCE_AUTOSTOP = os.environ.get('CHLIDE_CONVERGENCE_AUTOSTOP', '1') == '1'

    # The 13-2 centre must be S -- verified against PDB 5HPZ (13'2-hydroxy-chlorophyll a, the
    # structure that defines CCD code 68G; both independent copies are S). Note that yigall's
    # AMBER parameterisation of the same pigment (4D2/oxy/.../68G.mol2, MCPB/6G1.mol2) is R and
    # disagrees with that crystal structure.
    REQUIRED_LIGAND_CONFIG = 'S'

    def iteration_p90(self, pkl: Path) -> float:
        """90th-percentile ligand pLDDT for one finished iteration.

        nan-safe on purpose: ligand_plddts legitimately holds nan when the LASErMPNN <-> Boltz
        atom-name mapping fails for a design, and a single nan would otherwise poison the series.
        """
        v = pd.to_numeric(pd.read_pickle(pkl)['ligand_plddts'], errors='coerce').dropna()
        return float(np.nanpercentile(v, 90)) if len(v) else float('nan')

    def convergence_state(self, upto: int) -> dict:
        """Replay the stop rule over every complete iteration up to and including `upto`.

        Deterministic and stateless -- it reads the same iter_N_data.pkl files the resume logic
        already treats as ground truth, so there is no separate checkpoint that can drift out of
        sync with the iteration set after a crash.
        """
        cfg = self.CONVERGENCE
        series, smoothed = {}, {}
        best, best_iter, streak = -float('inf'), None, 0
        stop, reason = False, ''
        for i in range(0, upto + 1):
            pkl = self.sampling_dataframe_path / f'iter_{i}_data.pkl'
            queue = self.sampled_backbones_path / f'iter_{i}' / 'backbone_queue.txt'
            if not (pkl.exists() and queue.exists()):   # same completeness test as resume
                continue
            s = self.iteration_p90(pkl)
            if not np.isfinite(s):
                continue
            series[i] = s
            window = [series[k] for k in sorted(series)][-cfg['window']:]
            sm = float(np.mean(window))
            smoothed[i] = sm
            if len(window) < cfg['window']:
                # WARM-UP. Before `window` iterations exist, `sm` is a mean of one or two values,
                # not the 3-iteration mean every later comparison uses -- so it must not become
                # the baseline. v5 set `best` from iteration 0's RAW p90; because a single early
                # iteration is noisier (and often higher) than a 3-average, that inflated the bar
                # by ~0.0012 and held the streak at 2 while the raw series was still climbing.
                # Record the value for plotting, but establish no baseline and advance no streak.
                # The stop test below cannot fire here regardless: min_iterations (10) exceeds
                # window (3), so n_done is always too small during warm-up.
                continue
            if sm >= best + cfg['eps']:
                best, best_iter, streak = sm, i, 0
            else:
                streak += 1
            n_done = len(series)
            if streak >= cfg['patience'] and n_done >= cfg['min_iterations']:
                stop = True
                reason = (f"p90 ligand pLDDT (3-iteration mean) has not improved on "
                          f"{best:.4f} since iteration {best_iter} -- {streak} consecutive "
                          f"iterations")
        return dict(series=series, smoothed=smoothed, best=best if np.isfinite(best) else None,
                    best_iteration=best_iter, streak=streak, stop=stop, reason=reason, **cfg)

    def check_convergence(self, iidx: int) -> tuple[bool, dict]:
        """Evaluate the rule after iteration `iidx`, log one line, and persist the state."""
        cs = self.convergence_state(iidx)
        s = cs['series'].get(iidx)
        sm = cs['smoothed'].get(iidx)
        if s is not None:
            # `best` is None during the warm-up, before `window` iterations exist. Formatting it
            # as a float killed v6b at the end of iteration 0, and killed v8 the same way because
            # v8 was branched from v6b BEFORE that fix landed.
            best = cs['best']
            shown = f"best {best:.4f}@{cs['best_iteration']}" if best is not None \
                else f"warm-up, no baseline until {cs['window']} iterations"
            print(f"CONVERGENCE: iter {iidx}  p90 {s:.4f}  mean{cs['window']} {sm:.4f}  "
                  f"{shown}  streak {cs['streak']}/{cs['patience']}", flush=True)
        try:
            out = Path(NISE_RUN_DIR) / 'logs' / 'convergence.json'
            out.parent.mkdir(parents=True, exist_ok=True)
            # reporting artifact only -- never read back as truth, the pkls are the source
            out.write_text(json.dumps({**cs, 'stopped_at': iidx if cs['stop'] else None,
                                       'timestamp': time.time()}, indent=1, default=str))
        except Exception as e:  # noqa: BLE001
            print(f'CONVERGENCE: could not write convergence.json: {e}')
        if cs['stop'] and not self.CONVERGENCE_AUTOSTOP:
            # Report it, do not act on it. The number still goes into convergence.json and the
            # log, so the plateau is visible in the trajectory plots either way.
            print('CONVERGENCE: plateau reached but auto-stop is OFF -- continuing, because '
                  'later iterations are where BUNs is refined (Nature 656, 237).')
            return False, cs
        return cs['stop'], cs

    def log(self, use_wandb: bool, dataframe: pd.DataFrame):

        self.sampling_metadata['min_protein_rmsd'] = min(self.sampling_metadata['min_protein_rmsd'], dataframe['protein_rmsds'].dropna().min())
        self.sampling_metadata['min_ligand_rmsd'] = min(self.sampling_metadata['min_ligand_rmsd'], dataframe['ligand_rmsds'].dropna().min())

        logs = dict(self.sampling_metadata)

        logs['mean_sampled_ligand_plddt'] = dataframe['ligand_plddts'].dropna().mean()
        logs['mean_sampled_protein_plddt'] = dataframe['protein_plddts'].dropna().mean()
        logs['mean_sampled_protein_rmsd'] = dataframe['protein_rmsds'].dropna().mean()
        logs['mean_sampled_ligand_rmsd'] = dataframe['ligand_rmsds'].dropna().mean()
        logs['num_sequences_sampled'] = len(dataframe)

        logs['mean_affinity_probability'] = dataframe['affinity_probability_binary'].mean()
        logs['mean_affinity_pred_value'] = dataframe['affinity_pred_value'].mean()
        logs['mean_iptm'] = dataframe['iptms'].mean()

        logs['mean_laser_score'] = dataframe['laser_nll'].mean()
        logs['mean_laser_bs_score'] = dataframe['laser_bs_nll'].mean()

        try:
            if self.keep_input_backbone_in_queue:
                logs['max_sampled_backbone_priority'] = max(self.backbone_queue[1:], key=lambda x: x[1])[1]
            else:
                logs['max_sampled_backbone_priority'] = max(self.backbone_queue, key=lambda x: x[1])[1]
        except:
            logs['max_sampled_backbone_priority'] = 0

        print(logs)
        if use_wandb:
            scatter_fig = px.scatter(dataframe, x='ligand_rmsds', y='ligand_plddts', color='protein_rmsds', hover_data=['protein_rmsds', 'sequences'], range_color=[0.0, 2.5], range_y=[0.0, 1.0], range_x=[0.0, 5.0])
            logs['ligand_RMSD_vs_pLDDT_scatter'] = wandb.Html(plotly.io.to_html(scatter_fig)) # type: ignore
            wandb.log(logs)
    

def _fold_boltz(
    boltz_inputs_dir, boltz2x_executable_path, boltz_inference_devices, 
    boltz_output_dir, use_potentials, use_boltz_1x, disable_kernels, disable_nccl_p2p, debug
):
    """Fold every yaml in boltz_inputs_dir, writing the layout NISE expects downstream.

    Upstream issues ONE `boltz predict --devices N` and lets Lightning DDP shard the work.
    That deadlocks on this machine: with 4x RTX 4070 Ti all ranks park on a futex at 0% GPU
    utilisation and never progress, with or without NCCL_P2P_DISABLE=1 and NCCL_SHM_DISABLE=1
    (verified twice -- they hang before allocating any GPU memory).

    The work is embarrassingly parallel -- one independent structure per yaml -- so instead of
    one DDP job we shard the yamls across devices and run one single-GPU boltz process each,
    then consolidate the predictions back into <out_dir>/boltz_results_boltz_inputs/predictions/
    so the rest of NISE is unchanged. A single device keeps the original single-call path.
    """
    device_ints = [str(x).split(':')[-1] for x in boltz_inference_devices]

    def build(inputs_dir, out_dir, devices_arg, visible):
        cmd = (f'{boltz2x_executable_path} predict {inputs_dir} --devices {devices_arg}'
               f' --out_dir {out_dir} --output_format pdb --override')
        cmd = f'CUDA_VISIBLE_DEVICES={visible} {cmd}'
        if disable_nccl_p2p:
            cmd = f'NCCL_P2P_DISABLE=1 {cmd}'
        if use_potentials:
            cmd += ' --use_potentials'
        if use_boltz_1x:
            cmd += ' --model boltz1'
        if disable_kernels:
            cmd += ' --no_kernels'
        return cmd

    boltz_inputs_dir = Path(boltz_inputs_dir)
    boltz_output_dir = Path(boltz_output_dir)

    if len(device_ints) == 1:
        cmd = build(boltz_inputs_dir, boltz_output_dir, 1, device_ints[0])
        print(cmd)
        try:
            subprocess.run(cmd, shell=True, check=False,
                           stdout=subprocess.DEVNULL if not debug else None,
                           stderr=subprocess.DEVNULL if not debug else None)
        except Exception:
            print('Boltz crashed! This might be fine, trying to recover...')
        return

    # Shard the yamls round-robin so every device gets a comparable amount of work.
    yamls = sorted(boltz_inputs_dir.glob('*.yaml'))
    shard_dirs = []
    for i, dev in enumerate(device_ints):
        d = boltz_inputs_dir.parent / f'boltz_inputs_gpu{dev}'
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        shard_dirs.append(d)
    for j, y in enumerate(yamls):
        shutil.copy(y, shard_dirs[j % len(shard_dirs)] / y.name)

    procs = []
    for dev, d in zip(device_ints, shard_dirs):
        if not any(d.glob('*.yaml')):
            continue
        out = boltz_output_dir / f'gpu{dev}'
        cmd = build(d, out, 1, dev)
        print(cmd)
        log = open(boltz_output_dir / f'boltz_gpu{dev}.log', 'w')
        procs.append(subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT))
    for pr_ in procs:
        try:
            pr_.wait()
        except Exception:
            print('Boltz crashed! This might be fine, trying to recover...')

    # Consolidate into the single predictions/ directory the caller expects.
    dest = boltz_output_dir / 'boltz_results_boltz_inputs' / 'predictions'
    dest.mkdir(parents=True, exist_ok=True)
    for dev, d in zip(device_ints, shard_dirs):
        src = boltz_output_dir / f'gpu{dev}' / f'boltz_results_{d.name}' / 'predictions'
        if not src.exists():
            continue
        for stem_dir in src.iterdir():
            if not stem_dir.is_dir():
                continue
            target = dest / stem_dir.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(stem_dir), str(target))




def _fold_af3(
    boltz_inputs_dir, boltz2x_executable_path, boltz_inference_devices,
    boltz_output_dir, use_potentials, use_boltz_1x, disable_kernels, disable_nccl_p2p, debug
):
    """Fold every JSON in the input dir with AlphaFold3, writing the layout NISE expects.

    v6a swaps the folding engine; nothing downstream changes. The signature is deliberately the
    Boltz one so the call site in main() is untouched -- the Boltz-only arguments
    (boltz2x_executable_path, use_potentials, use_boltz_1x, disable_kernels, disable_nccl_p2p)
    are accepted and ignored. tools/af3_adapter.py explains the five AF3 traps this avoids; the
    expensive one is AF3_CONFORMER_SKIP, which defaults to the WRONG residue name and silently
    costs ~71 s per fold (~12 GPU-hours per 600-design iteration).

    Note there is no His-Mg bond constraint here. Boltz had a patched ConnectionsPotential with a
    2.0-2.6 A two-sided window; AF3 has no equivalent and, per Guy's README, you do not restrain
    the bond you are scoring. Coordination control is entirely the Rosetta relax gate's job.
    """
    device_ints = [str(x).split(':')[-1] for x in boltz_inference_devices]
    boltz_inputs_dir = Path(boltz_inputs_dir)
    boltz_output_dir = Path(boltz_output_dir)
    stems = sorted(j.stem for j in boltz_inputs_dir.glob('*.json'))
    if not stems:
        print('AF3: no JSON inputs to fold')
        return

    t0 = time.time()
    fold_shards(boltz_inputs_dir, boltz_output_dir / 'af3_out', device_ints,
                BOLTZ_LIGAND_CCD, debug=debug)
    ok, bad = to_boltz_layout(boltz_output_dir / 'af3_out', boltz_output_dir, stems)
    print(f'AF3: {ok}/{len(stems)} folded and adapted in {time.time() - t0:.0f}s'
          + (f' ({bad} failed)' if bad else ''), flush=True)




def predict_complex_structures(*args, **kwargs):
    """Dispatch to the configured engine.

    v7 and v8 were two 1200-line FILES whose only structural difference was which of the two
    functions above existed. Both now live here and the choice is made at runtime from
    FOLD_ENGINE, which chlide.loop sets from the campaign YAML. The signature is the Boltz one
    in both cases, so main()'s call site is untouched and the AF3 path accepts and ignores the
    Boltz-only arguments.
    """
    if FOLD_ENGINE == 'af3':
        return _fold_af3(*args, **kwargs)
    if FOLD_ENGINE == 'boltz':
        return _fold_boltz(*args, **kwargs)
    raise ValueError(f'FOLD_ENGINE must be "boltz" or "af3", not {FOLD_ENGINE!r}')
def main(use_wandb, reduce_executable_path, reduce_hetdict_path, **kwargs):

    design_campaign = DesignCampaign(**kwargs)

    for iidx in range(design_campaign.start_iteration, design_campaign.num_iterations):

        # Two-phase sampling: wide while seeding, narrow once the queue is full of
        # gate-passing backbones. Announced every time it changes -- a switch that happens
        # silently cannot be audited, and one that never fires means the run is quietly doing
        # every iteration at seed width.
        design_campaign.update_sampling_width(iidx)

        # Run laser on all backbone queue inputs.
        all_sampled_proteins, backbone_sample_indices, sampled_backbone_path, all_sampled_sequences = [], [], [], []
        laser_nll, laser_bs_nll = [], []
        for bidx, (backbone_path, score) in enumerate(design_campaign.backbone_queue):
            sampled_proteins, sampled_sequences, nlls, bs_nlls = design_campaign.sample_sequences(backbone_path)
            all_sampled_proteins.extend(sampled_proteins)
            backbone_sample_indices.extend([bidx] * len(sampled_proteins))
            sampled_backbone_path.extend([design_campaign.backbone_queue[bidx][0]] * len(sampled_proteins))
            all_sampled_sequences.extend(sampled_sequences)
            laser_nll.extend(nlls)
            laser_bs_nll.extend(bs_nlls)

        # LASErMPNN is finished for this iteration and will not run again until the next one, but
        # PyTorch's caching allocator keeps its sampling activations reserved for the whole Boltz
        # phase -- measured at ~4.2 GB on a 135-residue protein, against a 79 MiB model. Boltz
        # shards onto this same card, so hand the memory back to the driver rather than holding
        # it idle. Costs a few ms of re-allocation next iteration; changes no results.
        torch.cuda.empty_cache()

        # Sanity check output shapes before trying to fold or log anything.
        assert len(all_sampled_proteins) == len(backbone_sample_indices), f'Error in laser sampling: {len(all_sampled_proteins)} != {len(backbone_sample_indices)}'
        assert len(all_sampled_proteins) == len(laser_nll), f'Error in laser sampling: {len(all_sampled_proteins)} != {len(laser_nll)}'
        assert len(all_sampled_proteins) == len(laser_bs_nll), f'Error in laser sampling: {len(all_sampled_proteins)} != {len(laser_bs_nll)}'

        # Make subdirectories to write outputs to disk.
        sampling_subdir = design_campaign.sampled_backbones_path / f'iter_{iidx}'
        laser_output_subdir = sampling_subdir / 'laser_outputs'
        boltz_input_dir = sampling_subdir / 'boltz_inputs' 
        laser_output_subdir.mkdir(exist_ok=True, parents=True)
        # CLEAR the fold-input directory, do not just ensure it exists. The stems encode the
        # device count (chunk_{device}_seq_{n}), so restarting mid-iteration with a different
        # number of GPUs leaves orphans the new run never overwrites: going 4 GPUs -> 3 on
        # v6b iteration 1 left 375 stale chunk_3_* files, which were folded (~25% wasted GPU)
        # and then ignored, because all_boltz_model_paths only lists the stems just written.
        # Harmless to the science, pure waste, and invisible unless you count the directory.
        if boltz_input_dir.exists():
            shutil.rmtree(boltz_input_dir)
        boltz_input_dir.mkdir(parents=True)
        for stale in sampling_subdir.glob('boltz_inputs_gpu*'):
            shutil.rmtree(stale, ignore_errors=True)

        # Write all the boltz input directories.
        all_boltz_input_path_names = []
        sampled_sequence_chunks = np.array_split(all_sampled_sequences, len(design_campaign.boltz_inference_devices)) 
        for idx, chunk_sequences in enumerate(sampled_sequence_chunks):
            for seq_idx, seq in enumerate(chunk_sequences):
                # The INPUT FORMAT is engine-dependent too, not just the fold call: Boltz
                # reads .yaml, AF3 reads .json. Merging only the fold function left this
                # writing AF3 JSONs for a Boltz-led run, so Boltz found no *.yaml to fold and
                # the driver failed with "not all boltz predictions were written to disk".
                # The chunk_N_seq_M STEM is preserved either way, because every CSV, every
                # results/ identifier and the harvest join key depend on it.
                if FOLD_ENGINE == 'af3':
                    boltz_input_output_path = boltz_input_dir / f'chunk_{idx}_seq_{seq_idx}.json'
                    all_boltz_input_path_names.append(boltz_input_output_path.stem)
                    boltz_input_output_path.write_text(json.dumps(af3_json(
                        boltz_input_output_path.stem, str(seq), BOLTZ_LIGAND_CCD,
                        Path(NISE_RUN_DIR) / 'input' / 'claox_userccd.cif'), indent=2))
                else:
                    boltz_input_output_path = boltz_input_dir / f'chunk_{idx}_seq_{seq_idx}.yaml'
                    all_boltz_input_path_names.append(boltz_input_output_path.stem)
                    boltz_input_output_path.write_text(get_boltz_yaml_boilerplate(
                        str(seq), design_campaign.ligand_smiles,
                        design_campaign.predict_affinity))

        all_boltz_model_paths = [(sampling_subdir / 'boltz_results_boltz_inputs' / 'predictions' / x / f'{x}_model_0.pdb') for x in all_boltz_input_path_names]
        all_laser_output_paths = []
        for laser_output_structure, boltz_output_path in zip(all_sampled_proteins, all_boltz_model_paths):
            laser_output_path = laser_output_subdir / f'laser_{boltz_output_path.parent.stem}.pdb'
            pr.writePDB(str(laser_output_path), laser_output_structure)
            all_laser_output_paths.append(laser_output_path)


        curr_tries = 0
        max_tries = 10
        while curr_tries < max_tries and not all([x.exists() for x in all_boltz_model_paths]):
            if curr_tries != 0:
                print('Not all boltz predictions were completed or file system not updated.. retrying...')
                time.sleep(30)
            
            predict_complex_structures(
                boltz_input_dir, design_campaign.boltz2x_executable_path, design_campaign.boltz_inference_devices, 
                sampling_subdir, design_campaign.use_boltz_conformer_potentials, design_campaign.use_boltz_1x, 
                design_campaign.boltz2_disable_kernels, design_campaign.boltz2_disable_nccl_p2p, design_campaign.debug
            )
            curr_tries += 1

        assert all([x.exists() for x in all_boltz_model_paths]), f"Error: not all boltz predictions were written to disk."

        # Identify any new backbone candidates.
        sorted_designs_laser, sorted_designs_rosetta, log_data = design_campaign.identify_backbone_candidates(all_boltz_model_paths, all_laser_output_paths, sampled_backbone_path, reduce_executable_path, reduce_hetdict_path)

        with open(sampling_subdir / 'backbone_queue.txt', 'w') as f:
            f.write('\n'.join([f"{x[1]}\t{x[0]}" for x in design_campaign.backbone_queue]))

        iidx_data = {
            'sampled_backbone_path': sampled_backbone_path,
            'laser_nll': laser_nll,
            'laser_bs_nll': laser_bs_nll,
            'laser_paths': sorted_designs_laser,
            'rosetta_paths': sorted_designs_rosetta,
            'sequences': all_sampled_sequences,
            'curr_idx': [iidx] * len(laser_nll),
        }
        iidx_data.update(dict(log_data))

        # Write data to disk.
        # Say WHICH column is short. pandas only reports "All arrays must be of the same
        # length", which cost a full seed iteration to diagnose when the geometry columns
        # missed the name-mapping early exit.
        _lens = {k: len(v) for k, v in iidx_data.items() if hasattr(v, '__len__')}
        if _lens and len(set(_lens.values())) > 1:
            _mode = max(set(_lens.values()), key=list(_lens.values()).count)
            _bad = {k: n for k, n in _lens.items() if n != _mode}
            raise RuntimeError(
                f'log_data columns are ragged: most are {_mode}, but {_bad} differ. Every '
                f'append must happen on every path through the per-design loop, including '
                f'the early exits.')
        iidx_dataframe = pd.DataFrame(iidx_data)
        iidx_dataframe.to_pickle(design_campaign.sampling_dataframe_path / f'iter_{iidx}_data.pkl')

        design_campaign.log(use_wandb, iidx_dataframe)

        # Both completeness artifacts (the queue file and the pkl) are now on disk, so this
        # iteration counts towards the series and a stop here is safe to resume from.
        converged, cs = design_campaign.check_convergence(iidx)
        if converged:
            print('\n' + '=' * 78)
            print(f'CONVERGED -- stopping at iteration {iidx} of '
                  f'{design_campaign.num_iterations - 1}')
            print(f'  {cs["reason"]}')
            print(f'  best 3-iteration mean p90 ligand pLDDT: {cs["best"]:.4f} '
                  f'at iteration {cs["best_iteration"]}')
            print(f'  iterations saved: {design_campaign.num_iterations - 1 - iidx}')
            print(f'  rule: window {cs["window"]}, eps {cs["eps"]}, patience {cs["patience"]}, '
                  f'min_iterations {cs["min_iterations"]}')
            print('=' * 78, flush=True)
            break

    # ---- chain straight into the terminal pipeline -------------------------------------------
    # Without this the driver simply exits and the campaign sits idle until a human notices.
    # v7 converged at 01:37 and was not harvested until 10:45 -- nine hours of dead GPUs on a
    # shared box. Spawned detached (setsid) so it outlives this process, after a short delay so
    # the driver is gone before finish_at_iter.sh's pkill runs. Set NISE_NO_AUTO_HARVEST=1 to
    # suppress it, e.g. when stopping a run by hand to free the cards for someone else.
    last_iter = iidx
    if os.environ.get('NISE_NO_AUTO_HARVEST'):
        print(f'AUTO-HARVEST: suppressed by NISE_NO_AUTO_HARVEST; '
              f'run `bash tools/finish_at_iter.sh {last_iter}` when ready', flush=True)
    else:
        script = TOOLS_DIR / 'finish_at_iter.sh'
        log = Path(NISE_RUN_DIR) / 'logs' / 'finish.log'
        if not script.is_file():
            print(f'AUTO-HARVEST: {script} missing -- harvest NOT started', flush=True)
        else:
            try:
                subprocess.Popen(
                    ['setsid', 'bash', '-c',
                     f'sleep 15; exec bash {script} {last_iter} >> {log} 2>&1'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True)
                print(f'AUTO-HARVEST: terminal pipeline starting in 15 s for iteration '
                      f'{last_iter}; follow it in {log}', flush=True)
            except Exception as e:  # noqa: BLE001
                print(f'AUTO-HARVEST: could not start ({type(e).__name__}: {e}) -- '
                      f'run `bash tools/finish_at_iter.sh {last_iter}` by hand', flush=True)


if __name__ == "__main__":

    laser_sampling_params = {
        'sequence_temp': 0.5, 'first_shell_sequence_temp': 0.7, 
        'chi_temp': 1e-6, 'seq_min_p': 0.0, 'chi_min_p': 0.0,
        'disable_pbar': True, 'disabled_residues_list': ['X', 'C'], # Disables cysteine sampling by default.
        # ==================================================================================================== 
        # Optional: Pass a prody selection string of the form ('resnum 1 or resnum 3 or resnum 5...') to 
        # specify residues over which to constrain the sampling of ALA or GLY residues. 
        # This string can be generated using './identify_surface_residues.ipynb'
        # ==================================================================================================== 
        'budget_residue_sele_string': None, 
        'ala_budget': 4, 'gly_budget': 0, # May sample up to 4 Ala and 0 Gly over the selected region if not None.
        'disable_charged_fs': True, # Disables sampling D,E,K,R residues for buried residues around the ligand. 
    }


    # ====================================================================================
    # Campaign: chlorophyllide a bound to BDFP 44 (de novo, 135 aa), axial HIS85.
    # Input pose: bdfp_44_cla_yigall.pdb (NE2-Mg 2.45 A, HIS85 is the only histidine).
    # NB: the space-named copy 'bdfp 44 cla yigall.pdb' has the Mg element column set to '*'
    # instead of 'MG', which breaks LASErMPNN's element lookup. Use the underscored file.
    #
    # SMOKE_TEST = True  -> debug mode: BOTH selection gates are bypassed (`or self.debug`),
    #                       boltz stdout is shown, 1 sequence per device per backbone.
    #                       Use only to prove the ligand/boltz path end to end.
    # SMOKE_TEST = False -> production: gates active, 64 sequences per backbone.
    # ====================================================================================
    SMOKE_TEST = False

    params = dict(
        debug = (debug := SMOKE_TEST),
        # Upstream default is `True and not debug`, which silently turns W&B on for the
        # production run and then fails on the hardcoded entity='benf549' below.
        use_wandb = (use_wandb := False),

        input_dir = Path(__file__).resolve().parent / 'input',   # derived, never hardcoded

        ligand_3lc = 'CLAOX', # Should match CCD if using reduce.
        # Atom names below are the POST-protonation names written by
        # protonate_and_add_conect_records.py: it strips digits off the CCD name and appends
        # a per-prefix counter (MG->MG1, NA->NA1, O1A->OA1, CAA->CAA1, ...). The v5 hydroxyl
        # follows the same rule: OCD -> OCD1, replacing v4's H31 on CBD1.
        ligand_rmsd_mask_atoms = {'CAA1', 'CBA1', 'CGA1', 'OA1', 'OA2'}, # Atoms to IGNORE in RMSD calculation. The floppy C17 propionate arm.
        # Mg + the four pyrrole N: keeps the metal in the pocket. OCD1 is deliberately NOT here.
        # This set is an all-or-nothing gate, so every atom added to it makes the gate strictly
        # harder, and v4 showed the ring-E edge is exactly where burial fails (ND1 was outside the
        # strict alpha=9 surface in every design tested). A hydroxyl can legitimately face solvent
        # or H-bond to the protein; requiring it buried would bias the trajectory for no reason and
        # risks stalling the run the way alpha=9 did. Burial of OCD1 is REPORTED in the cascade
        # (SASA + alpha-hull), not gated on.
        ligand_atoms_enforce_buried = {'MG1', 'NA1', 'NB1', 'NC1', 'ND1'},
        ligand_atoms_enforce_exposed = {'OA1', 'OA2'}, # Atoms to enforce remain exposed relative to the convex hull when selecting new backbones. I would suggest only using this for linker regions attached to your ligand or clearly exposed charged polar groups.
        laser_sampling_params = laser_sampling_params,
        # 13(2)-hydroxy chlorophyllide a, disconnected-Mg form: 46 heavy atoms (C35H34MgN4O6),
        # matching input/CLAOX.pkl exactly -- verified by comparing the canonical heavy-atom
        # connectivity graphs of this SMILES and the CCD component (identical), and the C13(2)
        # stereocentre against the component's 3D conformer (both S).
        # The difference from v4 is exactly one atom: CBD1-H31 becomes CBD1-OCD1(-HOD1). The
        # descriptor at C13(2) reads S here and R in v4, but the spatial arrangement is unchanged
        # -- the OH sits where the H was, and the label flips only because O outranks the three
        # ring carbons whereas H ranked below them.
        # The four Mg-N bonds must NOT be present or RDKit ETKDGv3 cannot embed it and boltz
        # hard-fails with "Failed to compute 3D conformer". Used ONLY by calc_rmsd and
        # construct_helper_files, NOT by Boltz (see BOLTZ_LIGAND_CCD above). Must stay the
        # disconnected form: a dative-bond SMILES derived from the CCD fails
        # AssignBondOrdersFromTemplate with 'No matching found'.
        # The C17 propionate is [O-]: at pH 7 a propionic acid (pKa ~4.8) is
        # deprotonated. This string is not cosmetic -- construct_helper_files builds the
        # ROSETTA PARAMS from it via AddHs, so a neutral C(=O)O here puts an acid proton
        # (H13 on OA2) into every relaxed structure even when the CCD component and the
        # input pose are both correctly deprotonated. That is exactly what a reviewer sees.
        ligand_smiles = "C=Cc1c(C)c2cc3nc(c4c5[n-]c(cc6nc(cc1[n-]2)C(C)=C6CC)c(C)c5C(=O)[C@]4(O)C(=O)OC)[C@@H](CCC(=O)[O-])[C@@H]3C.[Mg+2]",

        objective_function = (objective_function := 'ligand_plddt'), # Current options: {'ligand_plddt', 'iptm', 'pbind', 'ligand_plddt_and_pbind', 'iptm_and_pbind'}, Check the top of the file for implemented strategies, if you find an alternative strategy to work well please make a git commit so others can test it out as well!
        drop_rmsd_mask_atoms_from_ligand_plddt_calc = True,
        keep_input_backbone_in_queue = False,
        keep_best_generator_backbone = True, # The highest scoring pose may not necessarily generate higher scoring poses, keeps the pose that has generated the best poses after the first iteration in the queue if not already the best scoring pose.
        # STAYS FALSE -- deliberately, and this was measured, not assumed.
        #
        # Setting it True is the obvious way to reject inverted epimers, and it does not work:
        # GetSubstructMatches(useChirality=True) requires every perceived centre to agree, and on
        # v5's iteration-15 poses RDKit perceives MG1 (a 4-coordinate magnesium, not a real
        # centre) and CA2 (present in predictions, absent from the reference) alongside the one
        # centre we care about. Both the inverted AND the correct population return ZERO matches,
        # so every ligand_rmsd becomes inf, the backbone queue empties at iteration 0 and the
        # campaign freezes -- the v4 funnel-collapse failure mode.
        #
        # The 13-2 centre is instead gated directly by tools/ligand_chirality.py, which reads the
        # CIP code at CBD1 alone. Keeping this False also keeps ligand_rmsd numerically identical
        # in meaning to v5, which is what makes v6b a single-variable comparison against it.
        rmsd_use_chirality = False,
        # CALIBRATION WARNING. Every threshold below, and the 0.8 ligand/protein pLDDT gates in
        # tools/harvest_global_top.py and tools/relax_pool.py, was tuned against BOLTZ confidence.
        # AF3's pLDDT is not numerically interchangeable with it, so after iterations 0-2 run
        # tools/recalibrate_gates.py and move them if the pass rates are wildly off. Leaving them
        # unchecked risks either passing everything (no selection pressure) or passing nothing
        # (a frozen funnel). v6a is a NEW baseline and is NOT comparable to v4/v5/v6b.
        self_consistency_ligand_rmsd_threshold = 2.5,
        self_consistency_protein_rmsd_threshold = 1.5,

        align_on_binding_site = False, # If align on binding site is True, protein RMSD (and self_consistency_protein_rmsd_threshold above) becomes binding site RMSD. Useful if you have a large protein with floppy regions away from the ligand. Binding site is computed as residues with sidechain atoms within 5.0A of the ligand in the lasermpnn output structure.
        # HIS85 is the sole histidine and the axial ligand for the Mg. resindex is 0-based and
        # positional, so it survives boltz renumbering the output 1-135.
        # Input is contiguous 1-135, so 85 - 1 = 84. Verified: resindex 84 is HIS85.
        fixed_identity_residue_indices = 'resindex 84', # An optional prody selection string of the form "resindex 0 2 3 4 5" or "resnum 1 3 5"...

        use_reduce_protonation = False, # If False, will use RDKit to protonate, these hydrogens will not preserve the input names and aren't placed conditioned on the sidechains but REDUCE sometimes drops hydrogens if geometry changes outside of expected bounds.
        reduce_hetdict_path = None, # Can set to None if use_reduce_protonation False
        reduce_executable_path = None, # Can set to None if use_reduce_protonation False; /nfs/polizzi does not exist on this machine.

        model_checkpoint = Path(LASER_PATH) / 'model_weights/laser_weights_0p1A_noise_ligandmpnn_split.pt',

        # Paper (Fry, Slaw & Polizzi 2025): 'The NISE protocol was run for 35 iterations, at
        # which point the ligand pLDDT had converged.'
        # Run 1 (6HRK) converged at iteration 5; 15 gives 3x margin for this shallower pocket.
        # Early stop applies: halt if the global best ligand pLDDT is flat for 5 iterations.
        # TOTAL iterations, not additional ones: the resume logic continues from the last
        # completed one. 9 were done (0-8) before the 24 Sep reboot; 21 gives 12 more.
        num_iterations = 2 if SMOKE_TEST else 25,   # v4 stopped at 15 while still improving
        num_top_backbones_per_round = 3,
        sequences_sampled_at_once = 30,

        # Patched Boltz: ConnectionsPotential's flat bottom is one-sided upstream (no lower
        # bound, buffer=2.0 hardcoded), so a declared bond collapses to ~1.52 A. The wrapper
        # supplies a 2.0-2.6 A two-sided bottom and measured 2.03 A with the Mg 0.12 A off the
        # N4 centroid. The shared boltz install is not modified.
        boltz2x_executable_path = ('/home/yigall/mambaforge/envs/boltz/bin/python '
                                   + str(TOOLS_DIR / 'boltz_coord_patch.py')),
        # >1 device is dispatched as independent single-GPU boltz processes by our
        # predict_complex_structures above; Lightning DDP deadlocks on these cards.
        # cuda:3 deliberately left free for other users on this shared box. laser_inference_device
        # is boltz_inference_devices[0] = cuda:0; the empty_cache() call in main() keeps that card
        # from being the memory outlier it used to be (7459 -> ~2900 MiB peak).
        # cuda:3 is LEFT FREE for other users on this shared box -- yigall runs AMBER
        # (pmemd.cuda) there. Taking all four cards puts our Boltz shard and his MD on the same
        # GPU at 100% utilisation, which slows both and risks OOM-ing his run, not just ours.
        boltz_inference_devices = (boltz_inference_devices := ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']),
        use_boltz_conformer_potentials = True, # Use Boltz-x mode, this is almost always better.
        boltz2_predict_affinity = True if ('pbind' in objective_function) else False,
        use_boltz_1x = False, # Run the same script using --model boltz-1, multi-device inference with this seems bugged with boltz v2.1.1
        boltz2_disable_kernels = False, # Disables cuEquivariance kernels, this is likely not necessary.
        boltz2_disable_nccl_p2p = True, # 4x RTX 4070 Ti: consumer cards have no P2P, NCCL hangs without this. On some systems with certain graphics cards, NCCL can hang indefinitely. This flag fixes this issue allowing running boltz / NISE with multiple GPUs. https://github.com/NVIDIA/nccl/issues/631 

        # Paper production scale: 1500 sampled sequences per iteration = 500 per backbone x 3
        # backbones. (They note ~150/iteration gives similar optimisation quality at ~40 min/iter
        # on one A4500, if this proves too slow.)
        # 200 x 3 backbones = 600/iteration, not v5/v6b's 1500. AF3 costs roughly 3-5x Boltz
        # per design, so 600 keeps an iteration in the same wall-clock band (~1.8-3.3 h). The
        # paper notes ~150/iteration gives similar optimisation quality, so 600 is not thin.
        # 500 x 3 backbones = 1500/iteration, matching v5/v7 and the paper's production scale.
        # This was 200 (600/iteration) on the assumption that AF3 costs 3-5x Boltz per design.
        # MEASURED on 4 GPUs with --input_dir batching: AF3 10.4 s/design/GPU vs v7's Boltz
        # 22-25 s -- AF3 is 2.2x FASTER, so the premise was wrong. It matters because ~41% of
        # AF3's output is the wrong epimer and gets rejected by the chirality gate: 600 would
        # leave only ~354 usable designs per iteration against v7's 1500.
        sequences_sampled_per_backbone = 500 if not debug else 1 * len(boltz_inference_devices),
        # BDFP44's pocket is far shallower than 6HRK's. Measured on the input pose: at alpha 9.0
        # only 10/45 ligand atoms are buried and the Mg plus three pyrrole N read as EXPOSED, so
        # ligand_atoms_enforce_buried would reject everything at iteration 0. alpha 50 is the
        # smallest value where the metal core is buried while the propionate stays exposed, and
        # it is stable out to 200 (24/45 buried throughout).
        burial_mask_alpha_hull_alpha = 50.0, # Set to a larger number for folds with wider pockets (ex: 7-helix bundle) (Ex: 100.0), see https://github.com/benf549/CARPdock/blob/main/visualize_hull.ipynb

        # ---- Rosetta relax inside the loop -------------------------------------------
        # Runs after NISE has renamed and re-protonated a promoted design (raw Boltz output
        # has no hydrogens and CCD atom names, which Rosetta rejects outright).
        relax_in_loop = True,
        relax_python = '/home/yigall/mambaforge/envs/Rosetta/bin/python',
        relax_params = str(Path(NISE_RUN_DIR) / 'input' / 'lig_from_input_pdb.params'),
        relax_top_n = 12,      # only top_k become backbones; relaxing all promoted is waste
        relax_workers = 12,    # ~25 s/design -> one wave of 12 is ~30 s per iteration
        relax_r0 = 2.15,       # physical Mg-N(His); matches metal_r0_angstrom used elsewhere
        relax_timeout = 1800,
        coordination_window = (2.0, 2.6),

        laser_inference_device = boltz_inference_devices[0],
        laser_inference_dropout = True,
    )
    if use_wandb:
        wandb.init(project='design-campaigns', entity='benf549', config=params)
    main(**params)
