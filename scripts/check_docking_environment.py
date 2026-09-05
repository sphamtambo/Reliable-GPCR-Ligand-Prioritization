"""
Checks whether every binary and Python package notebook 5 (docking) needs is
present in the current environment. Run this directly on the target env
(bioenv) before attempting to execute 05_docking.ipynb -- catches a missing
dependency in one pass instead of discovering it mid-run, cell by cell.

Usage:
    python3 scripts/check_docking_environment.py
"""
import importlib
import shutil
import sys

# =============================================================================
# CLI binaries -- gnina and vina are the two docking engines (consensus
# scoring), obabel handles receptor protonation, mk_prepare_ligand.py is
# meeko's ligand-prep entry point (installed as a console script, not a
# plain import).
# =============================================================================
REQUIRED_BINARIES = {
    "gnina": "GNINA docking engine (CNN-based rescoring). Install per gnina's own "
             "build instructions -- not on this notebook's internet-access path, "
             "install directly into the target env.",
    "vina": "AutoDock Vina docking engine. conda install -c conda-forge vina, "
            "or pip install vina (has its own compiled extension).",
    "obabel": "OpenBabel CLI, used for receptor protonation. "
              "conda install -c conda-forge openbabel",
    "mk_prepare_ligand.py": "Meeko's ligand-prep console script -- should appear "
                             "automatically once the meeko Python package is installed "
                             "(pip install meeko), not a separate install.",
}

# =============================================================================
# Python packages
# =============================================================================
REQUIRED_PACKAGES = {
    "rdkit": "pip install rdkit  (or conda install -c conda-forge rdkit)",
    "numpy": "pip install numpy",
    "pandas": "pip install pandas",
    "meeko": "pip install meeko  -- also provides the mk_prepare_ligand.py CLI checked above",
    "pdbfixer": "pip install pdbfixer  -- NOT currently used in 05_docking.ipynb's "
                "receptor prep (Module B has no missing-residue repair step yet), "
                "but the CB2 predecessor project's own docking pipeline used it "
                "(evidenced by intermediate filenames like '6KPF_findMissingResidues.pdb' "
                "in its copied output -- that's a PDBFixer API method name). "
                "Recommended addition, not yet wired into the notebook.",
    "openmm": "pip install openmm  -- pdbfixer's own dependency, check this if pdbfixer "
              "import fails even after installing pdbfixer itself.",
    "dimorphite_dl": "pip install dimorphite-dl  -- pH-aware ligand protonation-state "
                      "enumeration (empirical pKa data), used in Module D for candidate "
                      "ligand prep only. Note: pip package name has a hyphen "
                      "('dimorphite-dl') but the importable module name has an "
                      "underscore ('dimorphite_dl') -- do not confuse the two when "
                      "installing.",
}


def check_binary(name: str) -> bool:
    return shutil.which(name) is not None


def check_package(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "version unknown")
        return True, version
    except ImportError as exc:
        return False, str(exc)


def main() -> None:
    print(f"Python: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()

    print("=" * 70)
    print("CLI BINARIES")
    print("=" * 70)
    missing_binaries = []
    for name, hint in REQUIRED_BINARIES.items():
        found = check_binary(name)
        status = "FOUND" if found else "MISSING"
        path = shutil.which(name) or ""
        print(f"  [{status:7s}] {name:22s} {path}")
        if not found:
            missing_binaries.append((name, hint))

    print()
    print("=" * 70)
    print("PYTHON PACKAGES")
    print("=" * 70)
    missing_packages = []
    for name, hint in REQUIRED_PACKAGES.items():
        found, detail = check_package(name)
        status = "FOUND" if found else "MISSING"
        print(f"  [{status:7s}] {name:12s} {detail}")
        if not found:
            missing_packages.append((name, hint))

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if not missing_binaries and not missing_packages:
        print("Everything required is present. Ready to run 05_docking.ipynb.")
        return

    if missing_binaries:
        print(f"\n{len(missing_binaries)} missing CLI binar{'y' if len(missing_binaries)==1 else 'ies'}:")
        for name, hint in missing_binaries:
            print(f"  - {name}\n      {hint}")

    if missing_packages:
        print(f"\n{len(missing_packages)} missing Python package(s):")
        for name, hint in missing_packages:
            print(f"  - {name}\n      {hint}")


if __name__ == "__main__":
    main()
