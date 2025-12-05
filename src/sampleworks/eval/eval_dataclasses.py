from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.eval.occupancy_utils import occupancy_to_str


@dataclass
class Experiment:
    protein: str
    occ_a: float
    model: str
    method: str
    scaler: str
    ensemble_size: int
    guidance_weight: float
    gd_steps: int
    exp_dir: Path
    refined_cif_path: Path
    protein_dir_name: str
    rscc: float = np.nan  # these last three are placeholders for RSCC calculations.
    base_map_path: Union[Path, None] = None
    error: Union[Exception, None] = None


class ExperimentList(list[Experiment]):
    def summarize(self):
        logger.info(f"Proteins: {set(e.protein for e in self)}")
        logger.info(f"Models: {set(e.model for e in self)}")
        logger.info(f"Scalers: {set(e.scaler for e in self)}")


@dataclass
class ProteinConfig:
    """Configuration metadata for a set of protein maps and structures associated with some PDB id."""
    protein: str
    base_map_dir: Path
    selection: str
    resolution: float
    map_pattern: str
    structure_pattern: str = ""

    def __post_init__(self):
        # TODO validate structure patterns? Anything else we should check to avoid later errors?
        self.base_map_dir = Path(self.base_map_dir)  # just in case someone passes a string

    def get_base_map_path_for_occupancy(self, occupancy_a: float) -> Union[Path, None]:
        occ_str = occupancy_to_str(occupancy_a, use_6b8x_format=self.protein == "6b8x")
        map_path = self.base_map_dir / self.map_pattern.format(occ_str=occ_str)
        if map_path.exists():
            return map_path

        # TODO: this is a kluge we should work to remove @kchrispens
        alt_patterns = []
        if self.protein == "6b8x":
            alt_patterns.append(f"6b8x_{occupancy_to_str(occupancy_a)}_1.74A.ccp4")

        for alt in alt_patterns:
            alt_path = self.base_map_dir / alt
            if alt_path.exists():
                return alt_path

        logger.warning(f"Base map for protein {self.protein} ({map_path}) NOT FOUND")
        return None

    def load_map(
            self, map_path: Path, canonical_unit_cell=True, selection_coords=None, padding=0.0
    ) -> Union[XMap, None]:

        xmap = XMap.fromfile(str(map_path), resolution=self.resolution)
        if canonical_unit_cell:
            xmap = xmap.canonical_unit_cell()
        if selection_coords is not None:
            xmap = xmap.extract(selection_coords, padding=padding)

        return xmap

    def get_reference_structure_path(self, occupancy_a: float) -> Union[Path, None]:
        if not self.structure_pattern:
            return None

        occ_str = occupancy_to_str(occupancy_a, use_6b8x_format=self.protein == "6b8x")
        structure_path = self.base_map_dir / self.structure_pattern.format(occ_str=occ_str)
        if structure_path.exists():
            return structure_path

        # Try shifted version for 6b8x
        if self.protein == "6b8x":
            shifted_path = self.base_map_dir / self.structure_pattern.format(occ_str=occ_str).replace(
                ".cif", "_shifted.cif"
            )
            if shifted_path.exists():
                return shifted_path

        logger.warning(
            f"Reference structure for {self.protein} with occ {occupancy_a} not found: {structure_path}"
        )
        return None
