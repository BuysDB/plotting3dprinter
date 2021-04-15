import xml.etree.ElementTree as et
from .svg_interpreter import svg_to_coordinate_chomper, repart
from .raycaster import cast_rays
import matplotlib.pyplot as plt
import matplotlib

import numpy as np

def svg_to_gcode(svg_path,
                gcode_path,
                precision=10,
                speed = 4600,
                x_offset = 50,
                y_offset = 50,
                z_draw = 5,
                z_up = 7,
                create_outline=True,
                create_fill=False,
                min_fill_segment_size=2, # Don't fill with lines shorter than this value
                longest_edge = None # Rescale longest edge to this value
 ):

    v_z=speed
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
            inp=repart(parts), PRECISION=precision):
            print(t)
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
        o.write(f'G1 Z{z_up+20} F{v_z}\n') # Perform up
        o.write('G28 X Y\n') # perform home
        o.write(f'G1 Z{z_up} F{v_z}\n') # Perform up

        # iterate news items
        for i,path in enumerate(root.findall('.//sn:path', ns)):
            pathcolor = pathcolors.colors[i%pathcolors.N]
            # Parse thew path in d:
            d  = path.attrib['d'].replace(',', ' ')
            parts = d.split()




            #coordinates = [ [x,y]  for x,y,c svg_to_coordinate_chomper(
        #        inp=repart(parts), PRECISION=precision)] ]
            coordinates = []
            commands = []
            for (x,y),c in svg_to_coordinate_chomper(
                inp=repart(parts), PRECISION=precision):
                coordinates.append([x,y])
                commands.append(c)

            ### transform the coordinates:
            coordinates = np.array(coordinates)
            # translate:
            if longest_edge is not None:
                coordinates[:,0]-=min_x
                coordinates[:,1]-=min_y
                # Convert video coordinates to xy coordinates (flip y)
                coordinates[:,1] = (max_y-min_y)-(coordinates[:,1])
                # Scale

                coordinates*=scaler

            ### Generate the outline GCODE, and store the segments
            is_down = False
            current = 0,0
            prev= None
            segments = []

            for (x,y), command in zip(coordinates,commands):

                cmdcolor = {
                    'L':'r',
                    'l':'r',
                    'M':'k',
                    'V':'b',
                    'v':'b',
                    'H':'b',
                    'h':'b'

                }

                if np.isnan(x):
                    # penup:
                    if create_outline:
                        o.write(f'G1 Z{z_up} F{v_z}\n')
                    current = x,y
                    is_down=False
                    continue
                else:
                    if not is_down:
                        if create_outline:
                            # Move the head to the target location, while still being up
                            o.write(f'G1 X{x_offset+x:.2f} Y{y_offset+y:.2f} Z{z_up} F{speed}\n')

                            o.write(f'G1 Z{z_draw} F{v_z}\n')
                            is_down=True
                        prev=None

                    if (x,y) != prev:
                        if  not np.isnan(current[0]):
                            if create_outline:
                                o.write(f'G1 X{x_offset+x:.2f} Y{y_offset+y:.2f} F{speed}\n')
                            segments.append( [[current[0],current[1]], [x, y]])
                            plt.plot( [current[0],x], [current[1], y],
                            c=cmdcolor.get(command,'grey') #pathcolor
                            ,lw=0.5)
                        current = x,y
                    else:
                        # Dont write duplicate coordinates. Waste of space
                        pass
                    prev = current

            o.write(f'G1 Z{z_up} F{v_z}\n')
            segarr = np.array( segments )
            coords = np.array( coordinates )
            coords = coords[np.isnan( coords ).sum(1)==0]

            if create_fill:
                prev = None
                idx=0
                for ((ax,ay),(bx,by)) in  cast_rays(segarr, coords, 1, debug=True) :

                    # Dont write very tiny segments
                    d = np.sqrt( np.power(ax-bx,2) + np.power(ay-by,2) )
                    if d<min_fill_segment_size:
                        continue

                    if idx%2==0: # Switch direction to not put stress on the pen in one direction only
                        o.write(f'G1 Z{z_up} F{v_z}\n')
                        o.write(f'G1 X{x_offset+ax:.2f} Y{y_offset+ay:.2f} F{speed}\n')
                        o.write(f'G1 Z{z_draw} F{v_z}')
                        o.write(f'G1 X{x_offset+ax:.2f} Y{y_offset+ay:.2f} F{speed}\n')
                        o.write(f'G1 X{x_offset+bx:.2f} Y{y_offset+by:.2f} F{speed}\n')
                        o.write(f'G1 Z{z_up} F{v_z}\n')

                    else:
                        o.write(f'G1 Z{z_up} F{v_z}\n')
                        o.write(f'G1 X{x_offset+bx:.2f} Y{y_offset+by:.2f} F{speed}\n')
                        o.write(f'G1 Z{z_draw} F{v_z}')
                        o.write(f'G1 X{x_offset+bx:.2f} Y{y_offset+by:.2f} F{speed}\n')
                        o.write(f'G1 X{x_offset+ax:.2f} Y{y_offset+ay:.2f} F{speed}\n')
                        o.write(f'G1 Z{z_up} F{v_z}\n')
                    idx+=1

            o.write(f'G1 Z{z_up} F{v_z}\n')

            #plt.plot( coordinates[:,0],  coordinates[:,1])

        print('All done')
        plt.savefig(f'{gcode_path.replace(".gcode","")}.png', dpi=300)
