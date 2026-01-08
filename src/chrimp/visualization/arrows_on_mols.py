from chrimp.visualization.arrow_drawing_utils import atom_atom_attack, bond_atom_atom_attack, aa_control_points, baa_control_points

from typing import List, Tuple
import matplotlib.pyplot as plt
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

np.random.seed(22)


def filter_hydrogens(mol, arrows, explicit_idx=[]):
    """
    Returns a copy of `mol` where only H atoms that appear in `keep_indices`
    remain.  All other hydrogens are removed.
    """
    idx_to_keep = explicit_idx
    for a in arrows:
        idx_to_keep.append(a[1])
        idx_to_keep.append(a[2])
        if a[0] == "ba":
            idx_to_keep.append(a[3])
    # Convert to a mutable molecule
    rwmol = Chem.RWMol(mol)

    # We will collect the atom indices to be removed (in reverse order!)
    # so that removal doesn't mess up subsequent indices.
    indices_to_remove = []
    
    for atom in rwmol.GetAtoms():
        if atom.GetSymbol() == "H" and atom.GetIdx() not in idx_to_keep:
            if len(atom.GetBonds()) > 0: # if it has at least one bond
                indices_to_remove.append(atom.GetIdx())

    removed_indices = []
    # Remove from highest to lowest index
    for idx in sorted(indices_to_remove, reverse=True):
        removed_indices.append(idx)
        atom = rwmol.GetAtomWithIdx(idx)
        other_atom = atom.GetBonds()[0].GetOtherAtom(atom)
        other_atom.SetNumExplicitHs(other_atom.GetNumExplicitHs() + 1)
        rwmol.RemoveAtom(idx)

    new_arrows = []
    for a in arrows:
        new_idx_1 = a[1] - len([idx for idx in removed_indices if idx < a[1]])
        new_idx_2 = a[2] - len([idx for idx in removed_indices if idx < a[2]])
        
        if a[0] == "ba":
            new_idx_3 = a[3] - len([idx for idx in removed_indices if idx < a[3]])
            new_arrows.append((a[0], new_idx_1, new_idx_2, new_idx_3))
        else:
            new_arrows.append((a[0], new_idx_1, new_idx_2))

    return rwmol.GetMol(), new_arrows

def rotate_coords(coords, angle_rad):
    """
    Rotate a set of coordinates by `angle_rad` radians.
    """
    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                           [np.sin(angle_rad), np.cos(angle_rad)]])
    return np.dot(coords, rot_matrix)

def translate_coords(coords, translation):
    """
    Translate a set of coordinates by `translation`.
    """
    return coords + translation

def radical_arrow_svg_path(coord_start, coord_end, color_hex, coord_int=None, end_arrow_head=None):
    if coord_int is None:
        p_ii = np.array([
            coord_start[0] + coord_start[1] - coord_end[1],
            coord_start[1] + coord_start[0] - coord_end[0]
        ])
        control = 2 * p_ii - 0.5 * (coord_start + coord_end)
    else:
        control = 2 * coord_int - 0.5 * (coord_start + coord_end)

    # End of the half-arrow head
    if end_arrow_head is None:
        p_iii = coord_start + 1.2*(coord_end-coord_start)
    else:
        p_iii = end_arrow_head

    # Generate SVG path for quadratic Bézier curve
    svg_path = f"M {p_iii[0]} {p_iii[1]} L {coord_end[0]} {coord_end[1]} Q {control[0]} {control[1]}, {coord_start[0]} {coord_start[1]}"
    svg_string = f"<path class='arrow-0' d='{svg_path}' style='fill:none;stroke:#{color_hex};stroke-width:0.5px;stroke-linecap:butt;stroke-opacity:1'/>"
    return svg_string

def homo_cleavage_arrow(coord_mid_bond, coord_atom, color_hex):
    # Calculate distance between p1 and p_mid
    distance = np.linalg.norm(coord_mid_bond - coord_atom)

    # Calculate p_i (stop point)
    direction = coord_mid_bond -coord_atom 
    perpendicular = np.array([direction[1], direction[0]])
    perpendicular_normalized = perpendicular / np.linalg.norm(perpendicular)
    p_i = coord_atom + perpendicular_normalized * (2/3 * distance)
    p_mid_slight_shift = (0.9*coord_mid_bond + 0.1*coord_atom)

    return radical_arrow_svg_path(p_mid_slight_shift, p_i, color_hex)


def radical_meeting_point(p1, p2, middle_image, height_ratio=0.1, epsilon:float = 0):
    # Calculate the meeting point
    direction = p2 - p1
    mid_point = (p1+p2)/2

    h = np.linalg.norm(direction)*height_ratio
    perpendicular = np.array([-direction[1], direction[0]])

    # Normalize to unit vector
    unit_direction = direction / np.linalg.norm(direction)
    perpendicular = perpendicular / np.linalg.norm(perpendicular)

    # Two candidate points at distance h
    candidate1 = mid_point + h * perpendicular
    candidate2 = mid_point - h * perpendicular

    # Choose the one further from middle_image
    dist1 = np.linalg.norm(candidate1 - middle_image)
    dist2 = np.linalg.norm(candidate2 - middle_image)

    if dist1 > dist2:
        shifted_meeting_point = candidate1 * (1-epsilon) + p1 * epsilon
    else:
        shifted_meeting_point = candidate2 * (1-epsilon) + p1 * epsilon
        perpendicular *= -1
    
    end_arrow_head = shifted_meeting_point + (perpendicular*np.sin(np.pi/3)-unit_direction*np.cos(np.pi/3))*h

    return shifted_meeting_point, end_arrow_head


def radical_attack_arrow(coord_atom, coord_other, middle_image, color_hex):
    p1, p2 = coord_atom, coord_other
    mid_point = (p1+p2)/2

    # Calculate the meeting point
    direction = p2 - p1

    big_h = np.linalg.norm(direction)/5
    h = np.linalg.norm(direction)/10

    perpendicular = np.array([-direction[1], direction[0]])

    # Normalize to unit vector
    perpendicular = perpendicular / np.linalg.norm(perpendicular)

    # Two candidate points at distance h
    candidate1 = mid_point + h * perpendicular
    candidate2 = mid_point - h * perpendicular

    # Choose the one further from middle_image
    dist1 = np.linalg.norm(candidate1 - middle_image)
    dist2 = np.linalg.norm(candidate2 - middle_image)

    if dist1 > dist2:
        p1_prime = mid_point*2 + big_h*perpendicular - p2
    else:
        p1_prime = mid_point*2 - big_h*perpendicular - p2
        perpendicular *= -1

    epsilon = 0.05
    meet_point_1, end_arrow_head_1 = radical_meeting_point(p1, p2, middle_image = middle_image, epsilon = epsilon)
    return radical_arrow_svg_path(p1, meet_point_1, color_hex = color_hex, coord_int=p1_prime, end_arrow_head=end_arrow_head_1)

def arrows_on_mol(
    molblock_or_mol,
    h_given:bool=True,
    arrows:List[Tuple]=[],
    ax=None,
    save_path=None,
    explicit_idx=[],
    invisible_circles=False,
    hv_icon: bool = False,
    arrow_palette='chrimp_paper',
    return_svg=False,
    radical_arrows:bool=False,
    bond_extend_ratio: float = 1,
):
    if isinstance(molblock_or_mol, str):
        mol = Chem.MolFromMolBlock(molblock_or_mol, sanitize=False, removeHs=False)
        for a in mol.GetAtoms():
            a.SetNoImplicit(True)
    else:
        mol = molblock_or_mol 
        print(f"Given mol is {mol.can_smiles}")

    mol, arrows = filter_hydrogens(mol, arrows, explicit_idx)

    AllChem.Compute2DCoords(mol)

    # Calculate mean bond length and set ACS style
    mean_bond_length = Draw.rdMolDraw2D.MeanBondLength(mol)
    if mean_bond_length <= 0.0:
        mean_bond_length =  0.5

    mean_bond_length = mean_bond_length*(1/bond_extend_ratio)

    # Set up drawing options and ACS mode
    draw_options = Draw.rdMolDraw2D.MolDrawOptions()
    Draw.rdMolDraw2D.SetACS1996Mode(draw_options, mean_bond_length)

    # Draw the molecule with noisy coordinates in ACS style
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(-1, -1)
    drawer.SetDrawOptions(draw_options)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Get SVG output
    svg_output = drawer.GetDrawingText()

    # Modify the SVG
    # Get width and height of the SVG
    width = int(re.search(r"width='(\d+)px'", svg_output).group(1))
    height = int(re.search(r"height='(\d+)px'", svg_output).group(1))

    # Add white space on 4 sides (modify origin -perc% and add 2*perc% to width and height of the viewbox)
    amount_white_around = 0.2
    svg_output = re.sub(
        r"viewBox='(.*?)'",
        f"viewBox='{-amount_white_around*width} {-amount_white_around*height} {width*(1+2*amount_white_around)} {height*(1+2*amount_white_around)}'",
        svg_output,
    )
    # We want to resize it to our new viewbox
    svg_output = re.sub(
        r"<rect style='opacity:1.0;fill:#FFFFFF;stroke:none' width='.*?' height='.*?' x='.*?' y='.*?'> </rect>",
        f"<rect style='opacity:1.0;fill:#FFFFFF;stroke:none' width='{width*(1+2*amount_white_around)}' height='{height*(1+2*amount_white_around)}' x='{-amount_white_around*width}' y='{-amount_white_around*height}'> </rect>",
        svg_output,
    )

    # Magical coords, because it draws a nice arrow head in the style of ChemDraw
    coords_polgon_on_orgin = np.array([[-2.5, -0.85], [-2, 0], [-2.5, 0.85], [0.5, 0]])

    all_coords = []
    for i, atom in enumerate(mol.GetAtoms()):
        x_pos, y_pos = drawer.GetDrawCoords(i)  # returns a Point2D
        all_coords.append(np.array([x_pos, y_pos]))

    if arrow_palette == "chrimp_paper":
        colors_hex_dict = {
            "a":  "6a00a7",
            "ba": "e06461",
            "i":  "fca635",
            "hv": "ff0000",
        }
        colors_hex = [colors_hex_dict[a[0]] for a in arrows]
    elif arrow_palette == 'liac':
        # Pick colors as uniform on the plasma colormap
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(arrows)))
        colors_hex = [f"{int(c[0]*255):02X}{int(c[1]*255):02X}{int(c[2]*255):02X}" for c in colors]
    elif arrow_palette == "red":
        colors_hex = ["ff0000"] * len(arrows)
    else:
        raise ValueError(f"{arrow_palette = } not recognized")

    for i, arrow in enumerate(arrows):
        # Define the arrow head
        svg_output = re.sub(
            r"</svg>\n",
            f'<defs>\n<marker id="Arrow{i}"\nviewBox="0 -3 3 6"\nrefX="3"\nrefY="0"\nmarkerWidth="4"\nmarkerHeight="6"\nmarkerUnits="strokeWidth"\norient="auto">\n<path d="M 0,-3 L 3,0 L 0,3"\nfill="none"\nstroke="#{colors_hex[i]}"\nstroke-width="1"\nstroke-linecap="butt" />\n</marker>\n</defs>\n</svg>\n',
            svg_output,
        )

    for i, arrow in enumerate(arrows):
        if arrow[0] == "hv":
            p1, p2 = all_coords[arrow[1]], all_coords[arrow[2]]
            p_mid = 0.5 * (p1 + p2)

            svg_output  = re.sub(
                r"</svg>\n",
                f"{homo_cleavage_arrow(p_mid, p1, color_hex = colors_hex[i])}\n</svg>\n",
                svg_output,
            )

            svg_output  = re.sub(
                r"</svg>\n",
                f"{homo_cleavage_arrow(p_mid, p2, color_hex = colors_hex[i])}\n</svg>\n",
                svg_output,
            )
        
        elif radical_arrows and arrow[0] == "a":
            p1, p2 = all_coords[arrow[1]], all_coords[arrow[2]]
            middle_image = np.array([width/2, height/2])

            svg_output  = re.sub(
                r"</svg>\n",
                f"{radical_attack_arrow(p1, p2, middle_image=middle_image, color_hex = colors_hex[i])}\n</svg>\n",
                svg_output,
            )

            svg_output  = re.sub(
                r"</svg>\n",
                f"{radical_attack_arrow(p2, p1, middle_image=middle_image, color_hex = colors_hex[i])}\n</svg>\n",
                svg_output,
            )

        elif radical_arrows and arrow[0] == "ba":
            p1, p2, p3 = all_coords[arrow[1]], all_coords[arrow[2]], all_coords[arrow[3]]
            p_mid = 0.5*(p1+p2)
            middle_image = np.array([width/2, height/2])
            meet_point, end_arrow_head = radical_meeting_point(p2, p3, middle_image, epsilon=0.05)
            control_point_1, control_point_2 = baa_control_points(p_mid, p2, meet_point)

            coords = [p_mid, control_point_1, control_point_2, meet_point]

            svg_output = re.sub(
                r"</svg>\n",
                f"<path class='arrow-{i}' d='M {coords[0][0]} {coords[0][1]} C {coords[1][0]} {coords[1][1]}, {coords[2][0]} {coords[2][1]}, {coords[3][0]} {coords[3][1]} L {end_arrow_head[0]} {end_arrow_head[1]}' style='fill:none;stroke:#{colors_hex[i]};stroke-width:0.5px;stroke-linecap:butt;stroke-opacity:1'/>\n</svg>\n",
                svg_output,
            )

            svg_output  = re.sub(
                r"</svg>\n",
                f"{radical_attack_arrow(p3, p2, middle_image=middle_image, color_hex = colors_hex[i])}\n</svg>\n",
                svg_output,
            )
            
            svg_output  = re.sub(
                r"</svg>\n",
                f"{homo_cleavage_arrow(p_mid, p1, color_hex = colors_hex[i])}\n</svg>\n",
                svg_output,
            )

        elif arrow[0] == "a":
            # If no bond exists between arrow[1] and arrow[2]
            a1 = mol.GetAtomWithIdx(arrow[1])
            share_bond_with_p2 = False
            for bond in a1.GetBonds():
                if bond.GetOtherAtomIdx(arrow[1]) == arrow[2]:
                    share_bond_with_p2 = True
                    break
            
            p1, p2 = all_coords[arrow[1]], all_coords[arrow[2]]
            if share_bond_with_p2:
                # We do a reverse ionization attack
                p_mid = 0.5 * (p1 + p2)
                angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                d = np.linalg.norm(p2 - p1)
                h = -d / 2.5

                dx = -np.sin(angle) * h
                dy = np.cos(angle) * h

                new_p3 = np.array([p1[0] + dx, p1[1] + dy])

                p1, p2 = new_p3, p_mid

            control_point, im_point = aa_control_points(p1, p2, reverse=True)
            
            # Instead of a quadratic Bezier, I will do the cubic equivalent (seems to work better with arrows allignments)
            c1 = 2/3 * control_point + 1/3 * p1
            c2 = 2/3 * control_point + 1/3 * p2

            # Coords become a new 4x2 array
            coords = np.array([p1, c1, c2, p2])

            # Get the angle of the line going from c2 to p2
            angle = np.arctan2(-p2[1] + c2[1], p2[0] - c2[0])
            polygon_coords = [rotate_coords(c, angle) for c in coords_polgon_on_orgin] + p2

            # Only the bezier curve
            svg_output = re.sub(
                r"</svg>\n",
                f"<path class='arrow-{i}' d='M {coords[0][0]} {coords[0][1]} C {coords[1][0]} {coords[1][1]}, {coords[2][0]} {coords[2][1]}, {coords[3][0]} {coords[3][1]}' style='fill:none;stroke:#{colors_hex[i]};stroke-width:0.5px;stroke-linecap:butt;stroke-opacity:1'/>\n</svg>\n",
                svg_output,
            )

            # Draw the arrow head
            svg_output = re.sub(
                r"</svg>\n",
                f"<polygon points='{','.join([f'{x},{y}' for x, y in polygon_coords])}' style='fill:#{colors_hex[i]}'/>\n</svg>\n",
                svg_output,
            )

        elif arrow[0] == "ba":
            # Check if a bond exitst between p2 and p3
            a2 = mol.GetAtomWithIdx(arrow[2])
            share_bond_with_p3 = False
            for bond in a2.GetBonds():
                if bond.GetOtherAtomIdx(arrow[2]) == arrow[3]:
                    share_bond_with_p3 = True
                    break

            p1, p2, p3 = all_coords[arrow[1]], all_coords[arrow[2]], all_coords[arrow[3]]
            p_mid = 0.5*(p1+p2)
            if share_bond_with_p3:
                p3 = 0.5*(all_coords[arrow[2]] + all_coords[arrow[3]])
            control_point_1, control_point_2 = baa_control_points(p_mid, p2, p3)

            # Coords become a new 4x2 array
            coords = np.array([p_mid, control_point_1, control_point_2, p3])

            # Get the angle of the line going from control_point_2 to p3
            angle = np.arctan2(-p3[1] + control_point_2[1], p3[0] - control_point_2[0])
            polygon_coords = [rotate_coords(c, angle) for c in coords_polgon_on_orgin] + p3

            # Without the arrow head
            svg_output = re.sub(
                r"</svg>\n",
                f"<path class='arrow-{i}' d='M {coords[0][0]} {coords[0][1]} C {coords[1][0]} {coords[1][1]}, {coords[2][0]} {coords[2][1]}, {coords[3][0]} {coords[3][1]}' style='fill:none;stroke:#{colors_hex[i]};stroke-width:0.5px;stroke-linecap:butt;stroke-opacity:1'/>\n</svg>\n",
                svg_output,
            )

            # Draw the arrow head
            svg_output = re.sub(
                r"</svg>\n",
                f"<polygon points='{','.join([f'{x},{y}' for x, y in polygon_coords])}' style='fill:#{colors_hex[i]}'/>\n</svg>\n",
                svg_output,
            )

        elif arrow[0] == "i":

            p1, p2 = all_coords[arrow[1]], all_coords[arrow[2]]

            p_mid = 0.5 * (p1 + p2)
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            d = np.linalg.norm(p2 - p1)
            h = -d / 2.5

            dx = -np.sin(angle) * h
            dy = np.cos(angle) * h

            new_p3 = np.array([p2[0] + dx, p2[1] + dy])

            # Now, acts like an atom-atom attack from p_mid to new_p3
            control_point, im_point = aa_control_points(p_mid, new_p3, reverse=True)

            # Instead of a quadratic Bezier, I will do the cubic equivalent (seems to work better with arrows allignments)
            c1 = 2/3 * control_point + 1/3 * p_mid
            c2 = 2/3 * control_point + 1/3 * new_p3

            # Coords become a new 4x2 array
            coords = np.array([p_mid, c1, c2, new_p3])

            # Get the angle of the line going from c2 to new_p3
            angle = np.arctan2(-new_p3[1] + c2[1], new_p3[0] - c2[0])
            polygon_coords = [rotate_coords(c, angle) for c in coords_polgon_on_orgin] + new_p3

            # Only the bezier curve
            svg_output = re.sub(
                r"</svg>\n",
                f"<path class='arrow-{i}' d='M {coords[0][0]} {coords[0][1]} C {coords[1][0]} {coords[1][1]}, {coords[2][0]} {coords[2][1]}, {coords[3][0]} {coords[3][1]}' style='fill:none;stroke:#{colors_hex[i]};stroke-width:0.5px;stroke-linecap:butt;stroke-opacity:1'/>\n</svg>\n",
                svg_output,
            )

            # Draw the arrow head
            svg_output = re.sub(
                r"</svg>\n",
                f"<polygon points='{','.join([f'{x},{y}' for x, y in polygon_coords])}' style='fill:#{colors_hex[i]}'/>\n</svg>\n",
                svg_output,
            )
    
    if hv_icon:
        button_width = width * 0.20
        button_height = height * 0.20
        button_x = -amount_white_around * width * 0.8 #+ button_width * 0.3
        button_y = -amount_white_around * height * 0.8 #+ button_height * 0.3
        
        # Font size proportional to button size
        font_size = button_height * 0.9
        
        svg_output = re.sub(
            r"</svg>\n",
            f'''<g class="hv_node" style="cursor:pointer;">
    <rect x="{button_x}" y="{button_y}" width="{button_width}" height="{button_height}" 
          rx="0.2" ry="0.2" 
          style="fill:#f0f0f0;stroke:#333;stroke-width:0.01;opacity:0.9"/>
    <text x="{button_x + button_width/2}" y="{button_y + button_height/2}" 
          text-anchor="middle" 
          dominant-baseline="middle"
          style="font-family:sans-serif;font-size:{font_size}px;font-weight:bold;fill:#333;user-select:none;">hv</text>
</g>
</svg>\n''',
            svg_output,
        )

    if invisible_circles: # These circles can be used to interact with the SVG (e.g. in the web UI)
        # Finally, if atoms draw a circle around them
        for i, atom in enumerate(mol.GetAtoms()):
            atom_idx = atom.GetIdx()
            x_pos, y_pos = drawer.GetDrawCoords(atom_idx)
            size = 3 if atom.GetSymbol() == "C" else 5
            svg_output = re.sub(
                r"</svg>\n",
                f'<circle cx="{x_pos}" cy="{y_pos}" r="{size}px" fill="#000000" fill-opacity="0.0" class="node" nodeinfo="{atom_idx}"/>\n</svg>\n',
                svg_output,
            )

        # Circles for bonds
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            atom1_idx = atom1.GetIdx()
            atom2_idx = atom2.GetIdx()
            x_pos_1, y_pos_1 = drawer.GetDrawCoords(atom1_idx)
            x_pos_2, y_pos_2 = drawer.GetDrawCoords(atom2_idx)
            x_pos, y_pos = 0.5*(x_pos_1 + x_pos_2), 0.5*(y_pos_1 + y_pos_2)
            svg_output = re.sub(
                r"</svg>\n",
                f'<circle cx="{x_pos}" cy="{y_pos}" r="3px" fill="#000000" fill-opacity="0.0" class="node" nodeinfo="({atom1_idx},{atom2_idx})"/>\n</svg>\n',
                svg_output,
            )

        # Add the definition of arrowhead
        svg_output = re.sub(
            r"</svg>\n",
            """<defs>\n<marker\nid="arrowhead"\nmarkerWidth="9"\nmarkerHeight="4.2"\nrefX="7"\nrefY="2.1"\norient="auto">\n<polygon points="9 2.1, 0 0, 0 4.2" fill="red" />\n</marker>\n</defs>\n</svg>\n""",
            svg_output,
        )

    save_file = save_path

    if save_file is not None:
        # Save the SVG to a file
        with open(f"{save_file}.svg", "w") as f:
            f.write(svg_output)
    # Show the result to the user
    #from PIL import Image
    from io import BytesIO
    import cairosvg

    from PIL import Image

    img = Image.open(BytesIO(cairosvg.svg2png(bytestring=svg_output.encode("utf-8"), output_width=1000)))

    if ax is not None:
        # Resize image to fit the ax, keeping the aspect ratio
        aspect_ratio = img.width / img.height
        width_height_ratio = ax.get_window_extent().width / ax.get_window_extent().height

        # Get limiting dimension
        if aspect_ratio > width_height_ratio:
            new_width = ax.get_window_extent().width
            new_height = new_width / aspect_ratio
        else:
            new_height = ax.get_window_extent().height
            new_width = new_height * aspect_ratio

        img = img.resize((int(new_width), int(new_height)))

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(', '.join([str(a) for a in arrows]))
    else:
        plt.imshow(img)
        plt.axis("off")
        plt.title(', '.join([str(a) for a in arrows]))
        if save_file is not None:
            plt.savefig(f"{save_file}.png")
        if not return_svg:
            plt.show()

    # Clear the plt
    plt.clf()

    if return_svg:
        return svg_output
    else:
        return None

if __name__ == "__main__":
    from chrimp.world.molecule_set import MoleculeSet
    #smiles = "N1(C)C(=O)N(C)C=2N=CN(C)C2C1=O" # Caffeine
    #smiles = "FC(C(=O)O)(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)F" # PFOA
    #smiles = "CC([H])=O.CO.CO.[H+]" # acetal
    #smiles = "C=CC=C.C=C"
    #smiles = "NC1=CC=NC(Cl)=C1.CO"

    #smiles = "C[O-].NC1C=CN=C(Cl)C=1"
    #arrows = [("a", 1 , 7), ('i', 7, 6)]

    #smiles = "C[O-].NC1=CC=NC(Cl)=C1"
    #arrows = [("a", 1, 7), ('ba', 7, 9, 3), ('ba', 3, 4, 5), ('i', 5, 6)]

    #smiles = "C[O-].NC1=CC=NC(Cl)=C1"

    smiles = "CC=O.CO.[H+]" # with move [('a', 2, 5)]
    ms = MoleculeSet.from_smiles(smiles)
    arrows_on_mol(ms.molblock, arrows=[('a', 2, 5)], save_path="my_mol_2")





