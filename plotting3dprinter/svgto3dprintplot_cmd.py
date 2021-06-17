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
            '-precision',
            type=int,
            default=5,
            help="Amount of sub-divisions per spline")


    argparser.add_argument(
            '-speed',
            type=int,
            default=3600,
            help="speed of XY movements")


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
            '-z_draw',
            type=int,
            default=5,
            help="Z axis position for drawing")


    argparser.add_argument(
            '-z_up',
            type=int,
            default=7,
            help="Z axis position for transport moves")

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


    args = argparser.parse_args()
    svg_to_gcode(
                args.svg_path,
                args.gcode_path,
                precision=args.precision,
                speed = args.speed,
                x_offset = args.x_offset,
                y_offset = args.y_offset,
                z_draw = args.z_draw,
                z_up = args.z_up,
                create_outline=args.stroke,
                create_fill=args.fill,
                min_fill_segment_size=args.min_fill_segment_size, # Don't fill with lines shorter than this value
                longest_edge = args.longest_edge, # Rescale longest edge to this value
                bed_size_x=args.bed_size_x,
                bed_size_y=args.bed_size_y,

    )
