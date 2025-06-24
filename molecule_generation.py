from rdkit import Chem
import random
import pandas as pd
from rdkit.Chem import AllChem
import os
import time

start = time.perf_counter()

smiles_list = []

def has_carbon_with_two_double_bonds_in_ring(molecule):
    ring_atoms = set()

    atoms = molecule.GetAtoms()

    for atom in atoms:
        if atom.IsInRing():
            for ring in atom.GetOwningMol().GetRingInfo().AtomRings():
                ring_atoms.update(ring)

    for atom_index in ring_atoms:
        atom = molecule.GetAtomWithIdx(atom_index)
        if atom.GetAtomicNum() == 6:
            num_double_bonds = sum(1 for bond in atom.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE)
            if num_double_bonds == 2:
                return True
    return False

def has_alcohol(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "O" and len(atom.GetBonds()) == 1:
            bonded_to_carbon = any(
                bond.GetOtherAtom(atom).GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE for bond in
                atom.GetBonds())
            if bonded_to_carbon:
                return True
    return False

def has_aldehyde(mol):
    return any(
        bond.GetBondType() == Chem.BondType.DOUBLE and
        {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()} == {"C", "O"}
        for bond in mol.GetBonds()
    )

def has_bridgehead_carbon(molecule):
    if molecule is None:
        return False

    atoms = molecule.GetAtoms()

    ring_atoms = set()

    for atom in atoms:
        if atom.IsInRing():
            for ring in atom.GetOwningMol().GetRingInfo().AtomRings():
                ring_atoms.update(ring)

        if atom.GetAtomicNum() == 6:
            if len(atom.GetBonds()) >= 3:
                if atom.GetIdx() in ring_atoms:
                    return True

    return False

def has_peroxide_bond(molecule):
    return any(
        bond.GetBondType() == Chem.BondType.SINGLE and
        bond.GetBeginAtom().GetSymbol() == bond.GetEndAtom().GetSymbol() == "O"
        for bond in molecule.GetBonds()
    )


def check(matrix):
    for row in matrix[1:]:
        first_element, row_sum = row[0], sum(row[1:])
        if (first_element == 6 and row_sum > 4) or (first_element == 8 and row_sum > 2):
            return False
    return True

def print_matrix(matrix):
    print('\n'.join(' '.join(map(str, row)) for row in matrix), "\n-----------------")

def generate(i, j, matrix):
    if i == len(matrix):
        return

    if j >= len(matrix):
        generate(i + 1, i + 2, matrix)
        return

    num_c = matrix[1:].count([6])
    num_o = matrix[1:].count([8])

    for val in range(3):
        matrix[i][j] = matrix[j][i] = val

        if (matrix[i][0] == 6 and num_c >= num_o and sum(matrix[i]) - matrix[i][0] <= 4) or \
                (matrix[i][0] == 8 and num_c >= num_o and sum(matrix[i]) - matrix[i][0] <= 2):
            generate_molecule(matrix)
            generate(i, j + 1, matrix)

        matrix[i][j] = matrix[j][i] = 0  # Backtrack


def generate_molecule(matrix):
    mol = Chem.RWMol()
    num_c = num_o = 0

    atom_types = {6: 'C', 8: 'O'}

    for i in range(1, 4):
        atom_type = matrix[i][0]
        mol.AddAtom(Chem.Atom(atom_types.get(atom_type, 'C')))

    for i in range(4, len(matrix)):
        atom_type = random.choice([6, 8])
        matrix[i][0] = matrix[0][i] = atom_type
        mol.AddAtom(Chem.Atom('C')) if atom_type == 6 else mol.AddAtom(Chem.Atom('O'))
        num_c += atom_type == 6
        num_o += atom_type == 8

    for i in range(1, len(matrix)):
        for j in range(i + 1, len(matrix)):
            bond_type = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE}.get(matrix[i][j])
            if bond_type:
                mol.AddBond(i - 1, j - 1, bond_type)

    final_molecule = mol.GetMol()
    if ((check(matrix) and is_connected_molecule(final_molecule) and
         not has_peroxide_bond(final_molecule) and not has_bridgehead_carbon(final_molecule) and
         num_c >= num_o and not has_aldehyde(final_molecule) and not has_alcohol(final_molecule))
            and not has_carbon_with_two_double_bonds_in_ring(final_molecule)):
        smiles = Chem.MolToSmiles(final_molecule)
        if smiles not in generated_smiles:
            generated_smiles.add(smiles)
            smiles_list.append(smiles)
            print("Generated Molecule SMILES:", smiles)


def is_connected_molecule(molecule):
    num_atoms = molecule.GetNumAtoms()

    if num_atoms == 0:
        return False

    visited = [False] * num_atoms

    def dfs(atom_idx):
        stack = [atom_idx]
        while stack:
            current = stack.pop()
            if not visited[current]:
                visited[current] = True
                neighbors = [bond.GetOtherAtomIdx(current) for bond in molecule.GetAtomWithIdx(current).GetBonds()]
                stack.extend(neighbors)

    dfs(0)

    return all(visited)

generated_smiles = set()

for n in range(4, 10):
    matrix = [[0] * n for _ in range(n)]
    matrix[1][0] = matrix[0][1] = 6
    matrix[2][0] = matrix[0][2] = 8
    matrix[3][0] = matrix[0][3] = 6
    generate(1, 2, matrix)

excel_file_path = r"./molecule_smiles.xlsx"

df = pd.DataFrame({'SMILES': smiles_list})

df.to_excel(excel_file_path, index=False)

excel_file_path = r"./molecule_smiles.xlsx"
df = pd.read_excel(excel_file_path)
smiles_list = df['SMILES'].tolist()

main_folder = r"./gauss"

for i, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        folder_name = f"{smiles}"
        folder_path = os.path.join(main_folder, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mol_filename = os.path.join(folder_path, f"{folder_name}.mol")
        Chem.MolToMolFile(mol, mol_filename)

        chk_filename = os.path.join(folder_path, f"gauss.chk")

        gjf_filename = os.path.join(folder_path, f"gauss.gjf")

        gjf_content = f"%chk=./gauss.chk\n" \
                      f"%nprocs=50\n" \
                      f"%mem=30GB\n" \
                      f"#P opt freq UB3LYP/6-311+G(d,p)\n" \
                      f"\n" \
                      f"{folder_name}\n" \
                      f"\n" \
                      f"0 1\n"

        for conformer in mol.GetConformers():
            for atom in mol.GetAtoms():
                pos = conformer.GetAtomPosition(atom.GetIdx())
                atom_coordinates = f"{atom.GetSymbol():<2s}   {pos.x:>12.4f}   {pos.y:>12.4f}   {pos.z:>12.4f}\n"
                gjf_content += atom_coordinates
        gjf_content += "\n"

        with open(gjf_filename, "w") as gjf_file:
            gjf_file.write(gjf_content)

end = time.perf_counter()

runTime = end - start
runTime_ms = runTime * 1000
print("run time：", runTime, "s")
