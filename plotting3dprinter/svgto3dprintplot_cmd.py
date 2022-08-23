import argparse
from .svg_to_gcode import svg_to_gcode

def  main():
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Convert SVG into gcode for plotting on a 3d printer')
    argparser.add_argument(
        'svg_path',
        type=str,
        default='Path to svg file')
    argparser.add_argument(
            'gcode_path',
            type=str,
            help="output gcode path",
            )

    argparser.add_argument(
            '--stroke',
            action='store_true',
            help="Generate outline/stroke")
    argparser.add_argument(
            '--fill',
            action='store_true',
            help="Fill paths using ray-casting")
    argparser.add_argument(
            '--verbose',
            action='store_true',
            help="Show some plots during the SVG coordinate creation process.")

    argparser.add_argument(
            '-precision',
            type=int,
            default=5,
            help="Amount of sub-divisions per spline")


    argparser.add_argument(
            '-xy_feedrate',
            type=int,
            default=1000,
            help="speed of XY movements [mm/s]")

    argparser.add_argument(
            '-z_feedrate',
            type=int,
            default=200,
            help="speed of Z movements [mm/s]")

    """ True offset:
    G1 X34.13 F3600
    G1 Y32.36 F3600
    """

    argparser.add_argument(
            '-x_offset',
            type=int,
            default=45,
            help="X - Offset/margin from bottom left corner of print bed in mm")

    argparser.add_argument(
            '-y_offset',
            type=int,
            default=40,
            help="Y - Offset/margin from bottom of print bed in mm")

    argparser.add_argument(
            '-z_surface',
            type=int,
            default=-2,
            help="Z-coordinate for drawing/cutting to begin [mm].")


    argparser.add_argument(
            '-z_up',
            type=int,
            default=0,
            help="Z-coordinate height for transport moves [mm].")

    argparser.add_argument(
            '-z_safe',
            type=int,
            default=0,
            help="Amount to add to z_up to determine the CNC tool starting and ending Z-coordinate.")

    argparser.add_argument(
            '-cut_depth',
            type=int,
            default=0,
            help="If cutting material, how deep to go [mm]. Default = 0")

    argparser.add_argument(
            '-min_fill_segment_size',
            type=float,
            default=2.0,
            help="Don't fill with lines shorter than this value")

    argparser.add_argument(
            '-longest_edge',
            type=float,
            default=None,
            help=" Rescale longest edge of the image to this length")

    argparser.add_argument(
            '-bed_size_x',
            type=float,
            default=250,
            help="X size of bed in mm")

    argparser.add_argument(
            '-bed_size_y',
            type=float,
            default=200,
            help="Y size of bed in mm")

    argparser.add_argument(
            '-num_passes',
            type=int,
            default=1,
            help="Number of passes for CNC tool to perform.")
        


    args = argparser.parse_args()
    svg_to_gcode(
        svg_path=args.svg_path,
        gcode_path=args.gcode_path,
        precision=args.precision,
        xy_feedrate=args.xy_feedrate,
        z_feedrate=args.z_feedrate,
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        z_surface=args.z_surface,
        z_up=args.z_up,
        z_safe=args.z_safe,
        cut_depth=args.cut_depth,
        create_outline=args.stroke,
        create_fill=args.fill,
        min_fill_segment_size=args.min_fill_segment_size,
        longest_edge=args.longest_edge,
        bed_size_x=args.bed_size_x,
        bed_size_y=args.bed_size_y,
        verbose=args.verbose,
        passes=args.num_passes,
    )
