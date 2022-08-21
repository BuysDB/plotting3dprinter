import xml.etree.ElementTree as et
from .svg_interpreter import svg_to_coordinate_chomper, repart, svg_to_segment_blocks
from .raycaster import cast_rays
import matplotlib.pyplot as plt
import matplotlib
from copy import copy
import numpy as np


def get_pass_list(z_surface:float, z_bottom:float, passes:int) -> list:
    return np.linspace(z_surface,z_bottom,passes).tolist()

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
            #print(dist)
            if best is None or dist<best_dist:
                best = rblock
                best_dist=dist
                best_idx=j

        print('next best block is ',best_idx, best_dist )
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
                z_up = 7,
                z_safe = 20, # the z-coordinate of the CNC tool where it is well-above the surface
                cut_depth = 0,
                create_outline=True,
                create_fill=False,
                min_fill_segment_size=2, # Don't fill with lines shorter than this value
                longest_edge = None, # Rescale longest edge to this value
                bed_size_x=250,
                bed_size_y=200,
                verbose=False,
                passes=1, # number of passes (relevant for cutting material)
 ):
    assert cut_depth > 0, 'The cut depth must be a positive value [mm].'
    z_bottom = z_surface - cut_depth

    assert z_surface < z_up, 'The z-coordinate of the drawing/cutting surface must be below the safe travel height of the CNC tool.'
    assert z_safe >= z_up, 'The safe height of the CNC tool must be higher than the height at which the tool moves over the surface.'

    if z_bottom == z_surface:
        assert passes == 1, 'You only need to do 1 pass with the CNC tool since the bottom z-coordinate is equal to the surface z-coordinate.'
    else:
        assert passes > 1, 'You should set passes to a value greater than 1 since the bottom z-coordinate is lower than the surface coordinate.'
    
    cmdcolor = {
        'L':'r',
        'l':'r',
        'M':'k',
        'V':'b',
        'v':'b',
        'H':'b',
        'h':'b'

    }

    z_feedrate=xy_feedrate
    tree = et.parse(svg_path)
    # Add namespace for svg
    ns = {'sn': 'http://www.w3.org/2000/svg'}
    # get root element
    root = tree.getroot()

    # Find coordinate extends, for rescaling
    max_x = None
    min_x = None
    max_y = None
    min_y = None

    for i,path in enumerate(root.findall('.//sn:path', ns)):
        # Parse thew path in d:
        d  = path.attrib['d'].replace(',', ' ')
        parts = d.split()

        #coordinates = np.array(list(
    #        map(list, list( svg_to_coordinate_chomper(
    #        inp=repart(parts), PRECISION=precision) ))
#        ))

        coordinates = []
        commands = []
        for t in svg_to_coordinate_chomper(
            inp=repart(parts), PRECISION=precision, verbose=verbose):
            #print(t)
            (x,y),c = t

            coordinates.append([x,y])
            commands.append(c)

        #coordinates = [ [x,y]  for (x,y),c in  svg_to_coordinate_chomper(
    #        inp=repart(parts), PRECISION=precision)]
        coordinates = np.array(coordinates)

        bot = np.nanmin( coordinates[:,0])
        if min_x is None or bot<min_x:
            min_x = bot

        top = np.nanmax( coordinates[:,0])
        if max_x is None or top>max_x:
            max_x = top


        bot = np.nanmin( coordinates[:,1])
        if min_y is None or bot<min_y:
            min_y = bot

        top = np.nanmax( coordinates[:,1])
        if max_y is None or top>max_y:
            max_y = top

    # perform scaling such that the longest edge < longest_edge mm
    if longest_edge is not None:
        scale_x = longest_edge/(max_x - min_x)
        scale_y = longest_edge/(max_y - min_y)
        scaler = min(scale_x,scale_y)

    fig, ax = plt.subplots()


    pathcolors = plt.get_cmap('tab10')
    with open(gcode_path,'w') as o:

        # Move pen up:
        o.write(f'G01 Z{z_up+z_safe} F{z_feedrate}\n') # Perform up
        o.write('G28 X0 Y0\n') # perform home
        o.write(f'G01 Z{z_up} F{z_feedrate}\n') # Perform up

        # loop over passes
        pass_list = get_pass_list(z_surface, z_bottom, passes)
        for z_pass in pass_list:

            segment_blocks = list( svg_to_segment_blocks(svg_path) )
            prev=None
            # Determine block, start and ends
            blockset = []
            for block in segment_blocks:
                blockset.append([block[0][0], block[-1][1], block])

            if create_outline:
                prev = None
                for block in get_optimal_ordering(blockset):
                    for ii,( (x1,y1),(x2,y2) ) in enumerate( block ):

                        if longest_edge is not None:
                            x1-=min_x
                            x2-=min_x
                            y1-=min_y
                            y2-=min_y
                            # Convert video coordinates to xy coordinates (flip y)
                            y1 = (max_y-min_y)-(y1)
                            y2 = (max_y-min_y)-(y2)
                            # Scale
                            x1*=scaler
                            x2*=scaler
                            y1*=scaler
                            y2*=scaler

                        if x1>bed_size_x or y1>bed_size_y or x2>bed_size_x or y2>bed_size_y:
                            raise ValueError(f'Coordinates generated which fall outside of supplied printer bed size, adjust printer bed size or add "-longest_edge {min(bed_size_x,bed_size_y)}" to the command to scale the coordinates')

                        if prev is None or prev!=(x1,y1):
                            # Perform travel move:
                            print('traveling ...', (x1,y1))
                            if prev is not None:
                                # go up first
                                o.write(f'G01 X{x_offset+prev[0]:.2f} Y{y_offset+prev[1]:.2f} Z{z_up} F{xy_feedrate}\n')
                                plt.plot([prev[0],x1],[prev[1],y2],c='grey')
                            o.write(f'G01 X{x_offset+x1:.2f} Y{y_offset+y1:.2f} Z{z_up} F{xy_feedrate}\n')
                            # Drop pen down at current position
                            o.write(f'G01 X{x_offset+x1:.2f} Y{y_offset+y1:.2f} Z{z_pass} F{z_feedrate}\n')

                        #if prev!=(x1,y1):
                        #    print(x1,y1)

                        o.write(f'G01 X{x_offset+x1:.2f} Y{y_offset+y1:.2f} Z{z_pass} F{xy_feedrate}\n')
                        #print(x2,y2)
                        o.write(f'G01 X{x_offset+x2:.2f} Y{y_offset+y2:.2f} Z{z_pass} F{xy_feedrate}\n')
                        plt.plot([x1,x2],[y1,y2],c='r')
                        prev = (x2,y2)

                #o.write(f'#NEXT\n')

                # block ended.. travel
                #o.write(f'G1 X{x_offset+x2:.2f} Y{y_offset+y:.2f} Z{z_up} F{xy_feedrate}\n')

        o.write(f'G01 Z{z_up+z_safe} F{xy_feedrate}\n')
        o.write(f'G28 X0 Y0') # go back to home position

        #plt.plot( coordinates[:,0],  coordinates[:,1])

        print('All done')
        plt.xlim(-10,bed_size_x+10)
        plt.ylim(-10,bed_size_y+10)

        plt.axvline(0,c='r',ls=':')
        plt.axvline(bed_size_x,c='r',ls=':')
        plt.axvline(bed_size_x-x_offset,c='r',ls=':')

        plt.axhline(0,c='r',ls=':')
        plt.axhline(bed_size_y,c='r',ls=':')
        plt.axhline(bed_size_y-y_offset,c='r',ls=':')


        plt.savefig(f'{gcode_path.replace(".gcode","")}.png', dpi=300)
