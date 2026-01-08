import cairosvg
import io
import re
from PIL import Image
import matplotlib.pyplot as plt
import svgutils.transform as sg


from chrimp.world.mechsmiles import MechSmiles


def remove_white_background(svg_string: str) -> str:
    """
    Remove white background rectangles from an SVG string to make it transparent.

    Args:
        svg_string: SVG string that may contain white background rectangles

    Returns:
        SVG string with white backgrounds removed
    """
    # Remove white background rect elements using greedy matching
    return re.sub(r"<rect style='opacity:1\.0;fill:#FFFFFF;.*>", "", svg_string)


def make_arrow_svg(
    arrow_len: float,
    h_max: float,
    arrow_thickness: float = 1.6,
    head_factor: float = 3.0,
) -> str:
    """
    Return a standalone SVG containing a horizontal arrow.
    """
    from textwrap import dedent

    arrow_head_len = h_max / 2.0 / head_factor
    arrow_head_half = arrow_head_len / head_factor
    y_mid = h_max / 2.0
    y_top = y_mid - arrow_head_half
    y_bot = y_mid + arrow_head_half

    svg = f"""
    <svg width="{arrow_len}" height="{h_max}"
         viewBox="0 0 {arrow_len} {h_max}"
         xmlns="http://www.w3.org/2000/svg">
      <desc>Horizontal arrow – line plus filled triangular head</desc>

      <!-- shaft -->
      <line x1="0" y1="{y_mid}"
            x2="{arrow_len - arrow_head_len}" y2="{y_mid}"
            stroke="#000" stroke-width="{arrow_thickness}" />

      <!-- head -->
      <path d="
            M {arrow_len} {y_mid}
            L {arrow_len - arrow_head_len} {y_top}
            L {arrow_len - arrow_head_len} {y_bot}
            Z"
            fill="#000" stroke="none" />
    </svg>
    """
    return dedent(svg).lstrip()


def combine_svgs_with_arrows(
    svg_list: list[str],
    arrow_len: float = 80,
    arrow_thickness: float = 1.6,
    force_trailing_arrow: bool = False,
) -> str:
    """
    Combine multiple SVG strings in a linear fashion with arrows between them.

    Args:
        svg_list: List of SVG strings to combine
        arrow_len: Length of arrows between structures (in px)
        arrow_thickness: Thickness of arrow lines
        force_trailing_arrow: If True, add an arrow even after the last structure

    Returns:
        A single combined SVG string
    """
    if len(svg_list) == 0:
        return ""

    if len(svg_list) == 1 and not force_trailing_arrow:
        return svg_list[0]

    # Parse all SVG figures and get their dimensions
    figures = [sg.fromstring(svg) for svg in svg_list]
    dimensions = []
    h_max = 0

    for fig in figures:
        w, h = get_svg_dimensions(fig)
        dimensions.append((w, h))
        h_max = max(h_max, h)

    # Calculate total width
    num_arrows = len(svg_list) - 1 + (1 if force_trailing_arrow else 0)
    total_w = sum(w for w, h in dimensions) + arrow_len * num_arrows

    # Create the master canvas
    master = sg.SVGFigure(str(total_w), str(h_max))

    # Position each structure and arrow
    current_x = 0
    elements = []

    for i, (fig, (w, h)) in enumerate(zip(figures, dimensions)):
        # Position the structure
        root = fig.getroot()
        root.moveto(current_x, (h_max - h) / 2)
        elements.append(root)
        current_x += w

        # Add arrow (except after the last structure, unless force_trailing_arrow is True)
        if i < len(figures) - 1 or force_trailing_arrow:
            arrow_svg = make_arrow_svg(arrow_len, h_max, arrow_thickness)
            fig_arrow = sg.fromstring(arrow_svg)
            arrow_root = fig_arrow.getroot()
            arrow_root.moveto(current_x, 0)
            elements.append(arrow_root)
            current_x += arrow_len

    # Append all elements to master
    master.append(elements)
    master.root.attrib["viewBox"] = f"0 0 {total_w} {h_max}"

    # Add a white background rectangle at the beginning
    svg_string = master.to_str()

    # Decode if bytes
    if isinstance(svg_string, bytes):
        svg_string = svg_string.decode("utf-8")

    white_bg = f'<rect x="0" y="0" width="{total_w}" height="{h_max}" fill="#FFFFFF" opacity="1.0"/>'
    svg_with_bg = re.sub(r"(<svg[^>]*>)", r"\1" + white_bg, svg_string, count=1)

    return svg_with_bg


def get_svg_dimensions(svg_obj) -> tuple[float, float]:
    """
    Safely extract width and height from an SVG object.

    Args:
        svg_obj: An svgutils SVGFigure object

    Returns:
        Tuple of (width, height) as floats
    """
    size = svg_obj.get_size()

    if size is not None and len(size) == 2:
        try:
            # Standard case: get_size() returns dimensions
            w, h = [float(x[:-2]) for x in size]  # Remove 'pt' suffix
            if w > 0 and h > 0:
                return w, h
        except (ValueError, TypeError, IndexError):
            # If conversion fails, fall through to fallback methods
            pass

    # Fallback: try to extract from SVG attributes directly
    root = svg_obj.root
    if root is not None:
        width = root.attrib.get("width", "0")
        height = root.attrib.get("height", "0")

        try:
            # Remove 'pt' or 'px' suffix if present
            width = float(re.sub(r"(pt|px)$", "", str(width)))
            height = float(re.sub(r"(pt|px)$", "", str(height)))

            if width > 0 and height > 0:
                return width, height
        except (ValueError, TypeError):
            pass

        # Try viewBox as last resort
        viewbox = root.attrib.get("viewBox", "")
        if viewbox:
            try:
                parts = viewbox.split()
                if len(parts) == 4:
                    width = float(parts[2])
                    height = float(parts[3])
                    if width > 0 and height > 0:
                        return width, height
            except (ValueError, TypeError, IndexError):
                pass

    # If all else fails, return a default size
    return 100.0, 100.0


def combine_rows_vertically(row_svg_list: list[str], row_spacing: float = 20) -> str:
    """
    Combine multiple SVG strings (each representing a row) vertically.

    Args:
        row_svg_list: List of SVG strings to stack vertically
        row_spacing: Vertical spacing between rows (in px)

    Returns:
        A single combined SVG string with rows stacked vertically
    """
    if len(row_svg_list) == 0:
        return ""

    if len(row_svg_list) == 1:
        return row_svg_list[0]

    # Parse all SVG rows and get their dimensions
    rows = [sg.fromstring(svg) for svg in row_svg_list]
    dimensions = []
    max_w = 0

    for row in rows:
        w, h = get_svg_dimensions(row)
        dimensions.append((w, h))
        max_w = max(max_w, w)

    # Calculate total height
    total_h = sum(h for w, h in dimensions) + row_spacing * (len(row_svg_list) - 1)

    # Create the master canvas
    master = sg.SVGFigure(str(max_w), str(total_h))

    # Position each row
    current_y = 0
    elements = []

    for row, (w, h) in zip(rows, dimensions):
        # Center the row horizontally if it's narrower than max_w
        x_offset = (max_w - w) / 2
        root = row.getroot()
        root.moveto(x_offset, current_y)
        elements.append(root)
        current_y += h + row_spacing

    # Append all elements to master
    master.append(elements)
    master.root.attrib["viewBox"] = f"0 0 {max_w} {total_h}"

    # Add a white background rectangle at the beginning
    svg_string = master.to_str()

    # Decode if bytes
    if isinstance(svg_string, bytes):
        svg_string = svg_string.decode("utf-8")

    white_bg = f'<rect x="0" y="0" width="{max_w}" height="{total_h}" fill="#FFFFFF" opacity="1.0"/>'
    svg_with_bg = re.sub(r"(<svg[^>]*>)", r"\1" + white_bg, svg_string, count=1)

    return svg_with_bg


class MechanismVisualizer:
    def __init__(self, mech_smiles_list: list[str]):
        self.mech_smiles_list = [MechSmiles(msmi) for msmi in mech_smiles_list]

    def equilibrate(
        self,
    ):
        """
        Equilibrate all the steps to not lose or create any atom
        """
        pass

    def show(self, save_path=None, return_svg=False, max_msmi_in_one_row: int = -1):
        return self.show_linear(
            save_path=save_path,
            return_svg=return_svg,
            max_msmi_in_one_row=max_msmi_in_one_row,
        )

    def show_linear(
        self, save_path=None, return_svg=False, max_msmi_in_one_row: int = -1
    ):
        """
        The main util of this class, shows the global mechanism

        Args:
            save_path: Optional path to save the SVG file
            return_svg: If True, return the SVG string instead of displaying
            max_msmi_in_one_row: If > 0, split structures into multiple rows with this many structures per row
        """

        # We will first collect all of the reactants side of the MechSmiles
        structures_svg = [m.show_reac(return_svg=True) for m in self.mech_smiles_list]
        structures_svg.append(self.mech_smiles_list[-1].show_prod(return_svg=True))

        # Remove white backgrounds from all SVGs to prevent overlapping
        structures_svg = [remove_white_background(svg) for svg in structures_svg]

        # From the SVG, reconstruct a long SVG with arrows in between them
        if max_msmi_in_one_row > 0 and len(structures_svg) > max_msmi_in_one_row:
            # Split structures into rows
            rows_svg = []
            for i in range(0, len(structures_svg), max_msmi_in_one_row):
                row_structures = structures_svg[i : i + max_msmi_in_one_row]
                # Add trailing arrow for all rows except the last one
                is_last_row = i + max_msmi_in_one_row >= len(structures_svg)
                row_svg = combine_svgs_with_arrows(
                    row_structures, force_trailing_arrow=not is_last_row
                )
                rows_svg.append(row_svg)

            # Combine rows vertically
            combined_svg = combine_rows_vertically(rows_svg)
        else:
            # Original behavior: single row
            combined_svg = combine_svgs_with_arrows(structures_svg)

        if save_path is not None:
            with open(save_path, "wb") as f:
                f.write(
                    combined_svg.encode("utf-8")
                    if isinstance(combined_svg, str)
                    else combined_svg
                )

        if return_svg:
            plt.close("all")  # Close any previous plots
            return combined_svg
        elif save_path is None:
            # Show the svg with matplotlib
            png_bytes = cairosvg.svg2png(
                bytestring=combined_svg.encode("utf-8")
                if isinstance(combined_svg, str)
                else combined_svg,
                scale=3,
                dpi=1000,
            )
            image = Image.open(io.BytesIO(png_bytes))
            plt.imshow(image)
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    mechanism = [
        "CCCC(=O)NC1=C[CH:3]=[C:4](OC)C=C1.C[C:1](Cl)=[O:2]|((4, 3), 1);((1, 2), 2)",
        "CCCC(=O)NC1=CC([C:1](C)([O-:2])[Cl:3])[C+](OC)C=C1|(2, 1);((1, 3), 3)",
        "CCCC(=O)NC1=C[C:3](C(C)=O)([H:4])[C+:1](OC)C=C1.[Cl-:2]|(2, 4);((4, 3), 1)",
    ]

    svg_res = MechanismVisualizer(mechanism).show_linear(return_svg=True)

    mechanism = [
        "COC1=CC=C([N+:1]([O-])=[O:2])C=C1C(C)=O.[Cl:3][H:4].[H]Cl.[Zn:5]|(5, 1);((1, 2), 4);((4, 3), 3)",
        "COC1=CC=C([N+:3]([O-])(O)[Zn+:2])C=C1C(C)=O.Cl.[Cl-:1]|(1, 2);((2, 3), 3)",
        "COC1=CC=C([N:1]([OH:2])[O-:3])C=C1C(C)=O.[Cl:4][H:5].[Cl][Zn+]|(3, 1);((1, 2), 5);((5, 4), 4)",
        "COC1=CC=C(N=O)C=C1C(C)=O.O.[Cl-:2].[Cl][Zn+:1]|(2, 1)",
        "COC1=CC=C([N:1]=[O:2])C=C1C(C)=O.Cl.[Cl:4][H:5].[Zn:3]|(3, 1);((1, 2), 5);((5, 4), 4)",
        "COC1=CC=C([N:3](O)[Zn+:2])C=C1C(C)=O.Cl.[Cl-:1]|(1, 2);((2, 3), 3)",
        "COC1=CC=C([N-:2]O)C=C1C(C)=O.[Cl:3][H:4].[Cl][Zn+:1]|(2, 4);((4, 3), 1)",
        "COC1=CC=C(N[OH:1])C=C1C(C)=O.[H:2][Cl:3].[Zn]|(1, 2);((2, 3), 3)",
        "COC1=CC=C([NH:2][OH2+:1])C=C1C(C)=O.[Cl-].[Zn:3]|(3, 2);((2, 1), 1)",
        "COC1=CC=C([NH:2][Zn+:1])C=C1C(C)=O.O.[Cl-:3]|(3, 1);((1, 2), 2)",
        "COC1=CC=C([NH-:3])C=C1C(C)=O.O.[Cl:1][H:2].[Cl][Zn+]|(3, 2);((2, 1), 1)",
        "COC1=CC=C(N)C=C1C(C)=O.O.[Cl-:2].[Cl][Zn+:1]|(2, 1)",
    ]

    svg_res = MechanismVisualizer(mechanism).show_linear(
        return_svg=True, max_msmi_in_one_row=5
    )

    with open("first_whole_mechanism.svg", "w") as f:
        f.write(svg_res)
