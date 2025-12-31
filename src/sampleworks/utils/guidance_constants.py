from __future__ import annotations

from enum import Enum


class GuidanceType(str, Enum):
    """Enum for guidance/scaler types used in diffusion model guidance.

    These values control which guidance algorithm is used during sampling:
    - FK_STEERING: Feynman-Kac steering approach
    - PURE_GUIDANCE: Pure gradient-based guidance (DPS/Training-free guidance)
    """

    FK_STEERING = "fk_steering"
    PURE_GUIDANCE = "pure_guidance"


class StructurePredictor(str, Enum):
    """Enum for supported structure prediction models.

    Currently supported models:
    - BOLTZ_1: Boltz-1 model
    - BOLTZ_2: Boltz-2 model
    - PROTENIX: Protenix model
    - RF3: RF3 (RosettaFold3) model
    """

    BOLTZ_1 = "boltz1"
    BOLTZ_2 = "boltz2"
    PROTENIX = "protenix"
    RF3 = "rf3"


class Boltz2Method(str, Enum):
    """Enum for Boltz2 sampling methods.

    See https://github.com/jwohlwend/boltz/blob/cb04aeccdd480fd4db707f0bbafde538397fa2ac/src/boltz/data/const.py#L440
    """

    MD = "MD"
    X_RAY_DIFFRACTION = "X-RAY DIFFRACTION"
    ELECTRON_MICROSCOPY = "ELECTRON MICROSCOPY"
    SOLUTION_NMR = "SOLUTION NMR"
    SOLID_STATE_NMR = "SOLID-STATE NMR"
    NEUTRON_DIFFRACTION = "NEUTRON DIFFRACTION"
    ELECTRON_CRYSTALLOGRAPHY = "ELECTRON CRYSTALLOGRAPHY"
    FIBER_DIFFRACTION = "FIBER DIFFRACTION"
    POWDER_DIFFRACTION = "POWDER DIFFRACTION"
    INFRARED_SPECTROSCOPY = "INFRARED SPECTROSCOPY"
    FLUORESCENCE_TRANSFER = "FLUORESCENCE TRANSFER"
    EPR = "EPR"
    THEORETICAL_MODEL = "THEORETICAL MODEL"
    SOLUTION_SCATTERING = "SOLUTION SCATTERING"
    OTHER = "OTHER"
    AFDB = "AFDB"
    BOLTZ_1 = "BOLTZ-1"
