import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProteinInput:
    """
    Parse and validate protein input from a CSV file.
    """

    name: str
    structure: Path
    density: Path
    resolution: float

    def __post_init__(self):
        self.structure = Path(self.structure)
        self.density = Path(self.density)
        if not self.name:
            raise ValueError("Protein name must not be empty.")
        if not self.structure.exists():
            raise FileNotFoundError(
                f"Structure file does not exist for protein '{self.name}': {self.structure}"
            )
        if not self.density.exists():
            raise FileNotFoundError(
                f"Density file does not exist for protein '{self.name}': {self.density}"
            )
        if self.resolution <= 0:
            raise ValueError(
                f"Resolution must be positive for protein '{self.name}', got {self.resolution}."
            )

    @classmethod
    def from_csv(cls, csv_path: Path) -> list["ProteinInput"]:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        csv_dir = csv_path.parent

        required_columns = {"name", "structure", "density", "resolution"}
        protein_inputs = []
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            if reader.fieldnames is None:
                raise ValueError(f"CSV file {csv_path} appears to be empty")

            missing_columns = required_columns - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(
                    f"CSV file missing required columns: {missing_columns}. "
                    f"Found columns: {reader.fieldnames}"
                )

            for row in reader:
                structure = Path(row["structure"].strip())
                if not structure.is_absolute():
                    structure = csv_dir / structure

                density = Path(row["density"].strip())
                if not density.is_absolute():
                    density = csv_dir / density

                try:
                    resolution = float(row["resolution"].strip())
                except ValueError:
                    raise ValueError(
                        f"Invalid resolution '{row['resolution']}' "
                        f"for protein '{row['name'].strip()}'"
                    )

                protein_inputs.append(
                    cls(
                        name=row["name"].strip(),
                        structure=structure,
                        density=density,
                        resolution=resolution,
                    )
                )

        return protein_inputs
