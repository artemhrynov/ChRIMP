import cairosvg
import io
from textwrap import dedent
from PIL import Image
import svgutils.transform as sg
import matplotlib.pyplot as plt
from typing import Union

from chrimp.world.molecule_set import MoleculeSet


class MechSmilesVisualizer:
    def __init__(self):
        pass

    def show_reac(self, msmi, save_path=None, return_svg=False, hv_icon=None):
        ms = MoleculeSet.from_smiles(msmi.smiles)
        # Create a dictionary to map atom indices to their corresponding atoms
        arrow_list = [
            msmi.process_smiles_arrow(a, ms.atom_map_dict) for a in msmi.smiles_arrows
        ]
        ms = MoleculeSet.from_smiles(msmi.smiles)

        return ms.show_move_svg(
            arrow_list, save_path=save_path, return_svg=return_svg, hv_icon=hv_icon
        )

    def show_prod(self, msmi, save_path=None, return_svg=False, hv_icon=None):
        return msmi.ms_prod.show_move_svg(
            [], save_path=save_path, return_svg=return_svg, hv_icon=hv_icon
        )

    def show_cond(self, msmi, save_path=None, return_svg=False, hv_icon=None):
        conds = msmi.conds
        return MoleculeSet.from_smiles(".".join(x for x in conds)).show_move_svg(
            [], save_path=save_path, return_svg=return_svg, hv_icon=hv_icon
        )

    def show(self, msmi, save_path=None, return_svg=False, hv_icon=False):
        """
        Returns an SVG containing the whole reaction with its mechanism. The number of panes n will be MechSmiles.value.split("|") that are not empty strings

        If n=1, it is a SMILES, if n=2, it is a MechSmiles, if n>2, it is a MechSmiles with multiple mechanistic steps.
        The final image will be shown on (n-1)//2+1 lines, if n is odd, the last row will have one pane in the center.
        Arrows are drawn from idx 0 to idx -1 (if n>2), as well as in between each n and n+1 (if n>1)
        """

        value_split = msmi.value.split("|")

        if len(value_split) <= 1:
            # If only one pane, show the SMILES
            msmi.show_reac(save_path=save_path, hv_icon=hv_icon)

        if len(value_split) == 2:
            # If two panes, show the MechSmiles, an arrow and the product in one row
            reac = self.show_reac(
                msmi, save_path=save_path, return_svg=True, hv_icon=hv_icon
            )
            prod = self.show_prod(
                msmi, save_path=save_path, return_svg=True, hv_icon=False
            )
            cond = (
                self.show_cond(msmi, save_path=save_path, return_svg=True)
                if len(msmi.conds) > 0
                else None
            )

            combined_svg = reaction_svg(reac, prod, cond)

            if save_path is not None:
                with open(save_path, "wb") as f:
                    f.write(combined_svg)
            if return_svg:
                plt.close("all")  # Close any previous plots
                return combined_svg

            else:
                # Show the svg with matplotlib
                png_bytes = cairosvg.svg2png(bytestring=combined_svg, scale=3, dpi=1000)
                image = Image.open(io.BytesIO(png_bytes))
                plt.imshow(image)
                plt.axis("off")
                plt.show()

        else:
            raise NotImplementedError(
                "Multiple mechanistic steps are not implemented yet"
            )
            # Something like:

            # n_panes = len(value_split)
            # n_rows = (n_panes - 1) // 2 + 1
            # pane_width = 800
            # pane_height = 500
            # horizontal_space = 0.4  # Fraction of the pane width
            # vertical_space = 0.2  # Fraction of the pane height

            # smiles_list = [value_split[0]]
            # svg_list = []
            # for i in range(1, n_panes):
            #     # Create the SVG from the MechSmiles string
            #     mechsmi = MechSmiles(f"{smiles_list[i-1]}|{value_split[i]}")
            #     smiles_list.append(mechsmi.prod)
            #     svg_list.append(mechsmi.show(save_path=None, return_svg=True))

            # svg_list.append(mechsmi.show_prod(save_path=None, return_svg=True))

            # total_width = pane_width * (2 + horizontal_space)
            # total_height = (
            #     n_rows * pane_height * (1 + vertical_space)
            #     - vertical_space * pane_height
            # )


def reaction_svg(
    reac_svg: str,
    prod_svg: str,
    cond_svg: Union[str, None] = None,
    arrow_len: float = 80,  # px
    arrow_thickness: int = 2,
) -> str:
    """
    Combine two SVG strings so they appear side-by-side with a → in between.

    Returns a **single SVG string** that you can write to disk, display in
    Jupyter (`IPython.display.SVG(data=...)`) or convert to PNG with cairosvg.
    """
    # 1. Wrap the input strings as svgutils figures
    fig_r = sg.fromstring(reac_svg)
    fig_p = sg.fromstring(prod_svg)

    # 2. Get their intrinsic sizes (svgutils gives strings like '120pt')
    w_r, h_r = [float(x[:-2]) for x in fig_r.get_size()]  # remove the 'pt'
    w_p, h_p = [float(x[:-2]) for x in fig_p.get_size()]
    h_max = max(h_r, h_p)

    # 3. Build a tiny SVG containing just the arrow
    arrow_svg = make_arrow_svg(arrow_len, h_max)
    fig_a = sg.fromstring(arrow_svg)

    # 4. Create the final empty canvas
    total_w = w_r + arrow_len + w_p
    master = sg.SVGFigure(str(total_w), str(h_max))

    # 5. Move individual pieces into place
    fig_r_root = fig_r.getroot()
    fig_r_root.moveto(0, (h_max - h_r) / 2)

    fig_p_root = fig_p.getroot()
    fig_p_root.moveto(w_r + arrow_len, (h_max - h_p) / 2)

    fig_a_root = fig_a.getroot()
    fig_a_root.moveto(w_r, 0)
    if cond_svg is not None:
        fig_c = sg.fromstring(cond_svg)
        fig_c_root = fig_c.getroot()
        w_c, h_c = [float(x[:-2]) for x in fig_c.get_size()]
        fig_c_root.moveto(w_r + (arrow_len - w_c) / 2, ((h_max / 2) - h_c) / 2)
        master.append([fig_r_root, fig_p_root, fig_c_root, fig_a_root])
    else:
        master.append([fig_r_root, fig_p_root, fig_a_root])
    master.root.attrib["viewBox"] = f"0 0 {total_w} {h_max}"

    return master.to_str()


def make_arrow_svg(
    arrow_len: float,
    h_max: float,
    arrow_thickness: float = 1.6,
    head_factor: float = 3.0,
) -> str:
    """
    Return a standalone SVG containing a horizontal arrow.
    The geometry and parameters are identical to the version I sent earlier,
    but the 'd' attribute is now comment-free so it parses cleanly with lxml.
    """
    # --- geometry ---
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
