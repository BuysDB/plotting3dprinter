import xml.etree.ElementTree as et
from .svg_interpreter import svg_to_coordinate_chomper, repart, svg_to_segment_blocks, svg_to_coordinate_limits
from .raycaster import cast_rays
import matplotlib.pyplot as plt
import matplotlib
from copy import copy
import numpy as np
from itertools import chain

def get_pass_list(z_surface:float, z_bottom:float, passes:int) -> list: # Obtain list of Z-heights
    return np.linspace(z_surface,z_bottom,passes).tolist()

def get_optimal_scaler(bed_size_x:int,bed_size_y:int,im_length_x:int, im_length_y:int, longest_edge:float):
    X = bed_size_x
    Y = bed_size_y
    a = im_length_x
    b = im_length_y
    if ((X/Y)-1)*((a/b)-1) >= 0: # if the aspect ratio of the image and the print bed are similar...
        S = np.linspace(0,1,10000)
        dfds = np.array([-2*a*(X-s*a) - 2*b*(Y-s*b) for s in S]) # obtained from the derivative of f(s) = ((X - a*s)^2 + (Y - b*s)^2) where s is the scaler
        condition = np.array([(Y - s*b) >= 0 and (X - s*a) >= 0 for s in S])
        scaler_idx = np.argmin(np.abs(dfds)[condition])
        scaler = S[scaler_idx]
    else:
        scale_x = longest_edge/im_length_x
        scale_y = longest_edge/im_length_y
        scaler = min(scale_x,scale_y)
    return scaler

def get_optimal_ordering(blockset):
    remaining_blocks = copy(blockset)
    i = 0

    current_block = remaining_blocks.pop()
    yield current_block[-1]

    while len(remaining_blocks)>0:
        # find nearest block
        if len(remaining_blocks)==0:
            return
        end = current_block[-1][-1][1]

        best = None
        best_dist = None
        best_idx = None

        for j,rblock in enumerate(remaining_blocks):

            (xstart,ystart) = rblock[-1][0][0]

            dist = np.sqrt( np.power(xstart-end[0],2) +
                           np.power(ystart-end[1],2))
            if best is None or dist<best_dist:
                best = rblock
                best_dist=dist
                best_idx=j

        current_block = best

        yield best[-1]
        remaining_blocks.pop(best_idx)


def svg_to_gcode(svg_path,
                gcode_path,
                precision=10,

                xy_feedrate = 4600,
                z_feedrate = 250,

                x_offset = 50,
                y_offset = 50,

                z_surface = 5,
                z_up = 7, # z-coordinate of the tool
                z_safe = 20, # the z-coordinate of the CNC tool where it is well-above the surface
                cut_depth = 0, # 0 when drawing

                create_outline=True,
                create_fill=False,
                min_fill_segment_size=2, # Don't fill with lines shorter than this value
                longest_edge = None, # Rescale longest edge to this value
                bed_size_x=250,
                bed_size_y=200,
                style_filter=None,
                scale_factor = None,
                ray_distance=4,
                ignoreids=None,

 ):

    assert cut_depth >= 0, 'The cut depth must be a positive value [mm] (or 0 in the case of drawing).'
    z_bottom = z_surface - cut_depth
    assert z_surface < z_up, 'The z-coordinate of the drawing/cutting surface must be below the safe travel height of the CNC tool.'
    assert z_safe >= z_up, 'The safe height of the CNC tool must be higher than the height at which the tool moves over the surface.'
    if z_bottom == z_surface:
        assert passes == 1, 'You only need to do 1 pass with the CNC tool since the bottom z-coordinate is equal to the surface z-coordinate.'
    else:
        assert passes > 1, 'You should set passes to a value greater than 1 since the bottom z-coordinate is lower than the surface coordinate.'

    # Path filter is executed on each PATH found in the SVG file.
    # When returning True, the path will be included in the output
    # All paths are included in the calculation of the extends of the svg file
    path_filter =  None if style_filter is None else lambda path: style_filter in path.attrib.get('style','')

    cmdcolor = { # The colors used in the debug plot
            #, for the different PATH elements of the input SVG file
        'L':'r',
        'l':'r',
        'M':'k',
        'V':'b',
        'v':'b',
        'H':'b',
        'h':'b'
    }

    svg_parser_args = {
         'svg_path':svg_path,
         'precision':precision,
         'path_filter':path_filter
    }

    # Determine the minimum and max coordinates found in the svg file
    limits = svg_to_coordinate_limits(**svg_parser_args)

    # Determine the factor used to scale all coordinates
    if scale_factor is not None:
        scaler = scale_factor
    elif longest_edge is not None:
        scaler = get_optimal_scaler(bed_size_x,
                           bed_size_y,
                           limits.xmax-limits.xmin,
                           limits.ymax-limits.ymin,
                           longest_edge)
    else:
        scaler = 1

    # Prepare output plot for debugging:
    fig, ax = plt.subplots()
    pathcolors = plt.get_cmap('tab10')

    # Write gcode file
    with open(gcode_path,'w') as o:

        # Move pen up:
        o.write(f'G0 Z{z_safe} F{z_feedrate}\n') # Prepare for homing travel move
        o.write('G28 X Y\n') # perform home
        o.write(f'G0 Z{z_up} F{z_feedrate}\n') # Prepare for travel move

        segment_blocks = list(
            svg_to_segment_blocks(**svg_parser_args)
        )

        # Determine block, start and ends
        blockset = []
        for block in segment_blocks:
            blockset.append([block[0][0], block[-1][1], block])
            
        for z_pass in get_pass_list(z_surface, z_bottom, passes):
            prev = None
            for block in get_optimal_ordering(blockset):

                lines = []
                if create_fill:
                    segarr = np.array( block )
                    coordinates = []
                    for (x1,y1),(x2,y2) in block :
                        coordinates.append([x1,y1])
                        coordinates.append([x2,y2])

                    coords = np.array( coordinates )
                    coords = coords[np.isnan( coords ).sum(1)==0]
                    idx = 0
                    for (ax,ay),(bx,by) in cast_rays(block, coords,ray_distance=ray_distance):
                        d = np.sqrt( np.power(ax-bx,2) + np.power(ay-by,2) )
                        if d<min_fill_segment_size:
                            continue
                        if idx%2==0:
                            lines.append([(ax,ay),(bx,by)])
                        else:
                            lines.append([(bx,by),(ax,ay)])
                        idx+=1

                todo=[]
                if create_outline:
                    todo.append(block)
                if create_fill:
                    todo.append(lines)

                for ii,( (x1,y1),(x2,y2) ) in enumerate( chain( *todo ) ):

                    if longest_edge is not None:
                        x1-=limits.xmin
                        x2-=limits.xmin
                        y1-=limits.ymin
                        y2-=limits.ymin

                    # Convert video coordinates to xy coordinates (flip y)
                    y1 = (limits.ymax-limits.ymin)-y1
                    y2 = (limits.ymax-limits.ymin)-y2
                    # Scale
                    x1*=scaler
                    x2*=scaler
                    y1*=scaler
                    y2*=scaler

                    x1 += x_offset
                    x2 += x_offset
                    y1 += y_offset
                    y2 += y_offset

                    if x1>bed_size_x or y1>bed_size_y or x2>bed_size_x or y2>bed_size_y:
                        plt.plot([x1,x2],[y1,y2],c='purple')
                        continue # just ignore ?
                        #raise ValueError(f'Coordinates generated which fall outside of supplied printer bed size, adjust printer bed size or add "-longest_edge {min(bed_size_x,bed_size_y)}" to the command to scale the coordinates')

                    if prev is None or prev!=(x1,y1):
                        # Perform travel move:
                        print('traveling ...', (x1,y1))
                        if prev is not None:
                            # go up first
                            # Calculate distance:
                            dist = np.sqrt( np.power(prev[0]-x1,2) +
                                           np.power(prev[1]-y1,2))
                            if dist>10:
                                print('Large travel', dist)
                                o.write(f'G0 X{prev[0]:.2f} Y{prev[1]:.2f} Z{z_safe} F{xy_feedrate}\n')
                            else:
                                o.write(f'G0 X{prev[0]:.2f} Y{prev[1]:.2f} Z{z_up} F{xy_feedrate}\n')
                            plt.plot([prev[0],x1],[prev[1],y2],c='grey')
                        else:
                            print('Initial travel move')
                            o.write(f'G0 X{x1:.2f} Y{y1:.2f} Z{z_safe} F{xy_feedrate}\n')

                        o.write(f'G0 X{x1:.2f} Y{y1:.2f} Z{z_up} F{xy_feedrate}\n')
                        # Drop pen down at current position
                        o.write(f'G1 X{x1:.2f} Y{y1:.2f} Z{z_pass} F{xy_feedrate}\n')

                    o.write(f'G1 X{x1:.2f} Y{y1:.2f} Z{z_pass} F{xy_feedrate}\n')
                    #print(x2,y2)
                    o.write(f'G1 X{x2:.2f} Y{y2:.2f} Z{z_pass} F{xy_feedrate}\n')
                    plt.plot([x1,x2],[y1,y2],c='r')
                    prev = (x2,y2)

                #o.write(f'#NEXT\n')

                # block ended.. travel
                #o.write(f'G1 X{x_offset+x2:.2f} Y{y_offset+y:.2f} Z{z_up} F{speed}\n')

        o.write(f'G0 Z{z_safe} F{z_feedrate}\n')

        #plt.plot( coordinates[:,0],  coordinates[:,1])

        print('All done')
        plt.xlim(-10,bed_size_x+10)

        plt.axvline(0,c='r',ls=':')
        plt.axvline(bed_size_x,c='r',ls=':')
        plt.axvline(bed_size_x-x_offset,c='r',ls=':')

        plt.axhline(0,c='r',ls=':')
        plt.axhline(bed_size_y,c='r',ls=':')
        plt.axhline(bed_size_y-y_offset,c='r',ls=':')


        plt.savefig(f'{gcode_path.replace(".gcode","")}.png', dpi=300)
