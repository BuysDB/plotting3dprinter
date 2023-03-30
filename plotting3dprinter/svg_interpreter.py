from more_itertools import windowed
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as et
from collections import namedtuple

Limits = namedtuple('Limits', ['xmin', 'xmax','ymin','ymax'])

def interpolateBezier( points, steps=10, t=None):
    points = tuple(points)
    if len(points)==3:
        mapper = lambda t,p: (1-t)**2 * p[0] + 2*(1-t)*t*p[1] + t**2*p[2]
    elif len(points)==4:
        mapper = lambda t,p: (np.power( (1-t),3)*p[0] +\
         3* np.power((1-t),2) *t *p[1] +\
         3*(1-t)*np.power(t,2)*p[2] +\
         np.power(t,3)*p[3])
    else:
        raise Exception('Can only interpolate cubic and quadratic splines (3 or 4 parameters, got: %s'  % str(points))

    if t is not None:
        return   mapper(t, [q[0] for q in points]), mapper(t, [q[1] for q in points])
    xGen = ( mapper(t, [q[0] for q in points]) for t in np.linspace(0, 1, steps) )
    yGen = ( mapper(t, [q[1] for q in points]) for t in np.linspace(0, 1, steps) )

    return zip(xGen, yGen)


def parse_coord(inp):
    return np.array([ float(next(inp)),  float(next(inp))])

def svg_to_coordinate_chomper(inp,
                              yield_control:bool=False,
                              verbose:bool=False):
                              PRECISION:int=5,
    prev = None
    try:
        while True:
            chunk = next(inp)
            if chunk == 'M':
                #print('Got new start coordinate')

                start = parse_coord(inp)
                prev = start
                yield [np.nan,np.nan], 'M'
                yield list(start), 'M'

                if verbose:
                    print(f'M {start}')
                    #plt.scatter([start[0]],[start[1]])
                continue

            if chunk == 'm':
                #print('Got new start coordinate')

                #m = next(inp)
                if prev is None:
                    start = parse_coord(inp)
                else:
                    start = parse_coord(inp) +prev
                prev = start
                yield [np.nan,np.nan], 'm'
                yield list(start), 'm'
                #yield [np.nan,np.nan]
                #print(f'm {start}')
                continue



            #print(chunk)
            if chunk in 'zZ':
                # Go to start:
                #print("Returning to start coordinate")
                prev= start
                yield start, 'z'

                #print("Done")
                continue


            if chunk.strip()=='l':
                # Line to command:
                yield prev, 'l'
                cur = parse_coord(inp)+prev
                yield cur, 'l'
                prev = cur

                continue

            if chunk.strip()=='L':
                # Line to command:
                yield prev, 'L'
                cur = parse_coord(inp)
                yield cur, 'L'
                prev = cur
                if verbose:
                    print(f'L {prev} > {cur}')
                continue

            if chunk[0]=='c': # bezier mode
                 #c dx1,dy1 dx2,dy2 dx,dy

                dxdy1 = parse_coord(inp) + prev
                dxdy2 = parse_coord(inp)  + prev
                dxdy = parse_coord(inp)  + prev

                #print('C Bezier',prev,dxdy1,dxdy2,dxdy)

                if yield_control:
                    yield dxdy1
                    yield dxdy2
                    yield dxdy

                else:
                    # Resample the bezier curve
                    for x,y in interpolateBezier(
                        [
                            prev,
                            dxdy1   ,
                            dxdy2   ,
                            dxdy
                        ], steps=PRECISION

                    ):
                        yield np.array([x,y]), 'c'


                prev = dxdy
                #yield prev
                continue

            if chunk[0]=='C': # bezier mode
                 #c dx1,dy1 dx2,dy2 dx,dy

                dxdy1 = parse_coord(inp)
                dxdy2 = parse_coord(inp)
                dxdy = parse_coord(inp)

                #print('C Bezier',prev,dxdy1,dxdy2,dxdy)

                if yield_control:
                    yield dxdy1
                    yield dxdy2
                    yield dxdy

                else:
                    # Resample the bezier curve
                    for x,y in interpolateBezier(
                        [
                            prev,
                            dxdy1   ,
                            dxdy2   ,
                            dxdy
                        ], steps=PRECISION

                    ):
                        yield np.array([x,y]), 'C'

                prev = dxdy
                #yield prev
                continue

            if chunk[0]=='A': # ARC mode:
                #print('Got arc')
                rx,ry = parse_coord(inp)

                x_ax_rot = float( next(inp) )
                large_arc = int( next(inp) )
                sweep = int( next(inp) )
                cx,cy = parse_coord(inp)

                raise NotImplementedError()
                #yield from arc_sampler( rx,ry,x_ax_rot,large_arc,sweep, cx,cy,n_segs = 30 )
                continue


            if chunk[0]=='q': # quadratic bezier mode
                 #c dx1,dy1 dx2,dy2 dx,dy

                dxdy1 = parse_coord(inp) + prev
                dxdy = parse_coord(inp)  + prev

                #print('Q Bezier',prev,dxdy1, dxdy)

                if yield_control:
                    yield dxdy1
                    yield dxdy

                else:
                    # Resample the bezier curve
                    for x,y in interpolateBezier(
                        [
                            prev,
                            dxdy1   ,
                            dxdy
                        ], steps=PRECISION

                    ):
                        yield np.array([x,y]), 'q'

                prev = dxdy
                #yield prev
                continue


            if chunk[0]=='Q': # quadratic bezier , absolute
                 #c dx1,dy1 dx2,dy2 dx,dy
                # C x1 y1, x2 y2, x y

                dxdy1 = parse_coord(inp)
                dxdy = parse_coord(inp)

                #print('Q Bezier',dxdy1, dxdy)

                if yield_control:
                    yield dxdy1
                    yield dxdy

                else:
                    # Resample the bezier curve
                    for x,y in interpolateBezier(
                        [
                            prev,
                            dxdy1   ,
                            dxdy
                        ], steps=PRECISION

                    ):
                        yield np.array([x,y]), 'Q'

                prev = dxdy
                #yield prev
                continue

            if chunk not in 'HhVv':
                raise ValueError(f'Unknown command {chunk}')
                #print("MISS:",chunk)
                prev +=  parse_coord(chunk)
                #print(prev)
                yield list(prev) #prev #+start
            elif chunk=='h':
                # Parse the next chunk: (single horizontal coordinate)

                yield list(prev), 'h'

                chunk2 = next(inp)
                x = float(chunk2)
                prev[0] += x

                yield list(prev), 'h' #+start
            elif chunk=='H':
                # Parse the next chunk: (single horizontal coordinate)

                yield list(prev), 'H'

                chunk2 = next(inp)
                x = float(chunk2)
                prev[0] = x

                if verbose:
                    print(f'H {x} ({prev})')

                yield list(prev), 'H' #+start

            elif chunk=='v':
                # Parse the next chunk: (single horizontal coordinate)
                yield list(prev), 'v'

                chunk2 = next(inp)
                y = float(chunk2)
                prev[1] += y

                yield list(prev), 'v' #+start

            elif chunk=='V':
                # Parse the next chunk: (single horizontal coordinate)
                yield list(prev), 'V'

                chunk2 = next(inp)
                y = float(chunk2)
                prev[1] = y
                if verbose:
                    print(f'H {y} ({prev})')
                yield list(prev), 'V' #+start
    except StopIteration:
        pass

def repart(inp):
    for p in inp:
        if len(p)>1 and p[0].upper() in 'HZMCQ':
            yield p[0]
            yield p[1:]
        else:
            yield p


def svg_to_coordinate_limits(svg_path, precision, path_filter):

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
        # Parse the path in d:
        keep_path = True
        if path_filter is not None and not path_filter(path):
            keep_path = False
        if 'd' not in path.attrib:
            continue

        d  = path.attrib['d'].replace(',', ' ')
        parts = d.split()
        unfiltered_coordinates = []
        coordinates = []
        for t in svg_to_coordinate_chomper(
            inp=repart(parts), PRECISION=precision):
            #print(t)
            (x,y),c = t
            unfiltered_coordinates.append([x,y])
            if keep_path:
                coordinates.append([x,y])

        #coordinates = [ [x,y]  for (x,y),c in  svg_to_coordinate_chomper(
    #        inp=repart(parts), PRECISION=precision)]
        coordinates = np.array(coordinates)
        unfiltered_coordinates = np.array(unfiltered_coordinates)

        bot = np.nanmin( unfiltered_coordinates[:,0])
        if min_x is None or bot<min_x:
            min_x = bot

        top = np.nanmax( unfiltered_coordinates[:,0])
        if max_x is None or top>max_x:
            max_x = top


        bot = np.nanmin( unfiltered_coordinates[:,1])
        if min_y is None or bot<min_y:
            min_y = bot

        top = np.nanmax( unfiltered_coordinates[:,1])
        if max_y is None or top>max_y:
            max_y = top

    return  Limits(min_x, max_x, min_y, max_y)

def svg_to_segment_blocks(svg_path,precision=5, path_filter=None):
    tree = et.parse(svg_path)
    ns = {'sn': 'http://www.w3.org/2000/svg'}
    root = tree.getroot()
    for i,path in enumerate(root.findall('.//sn:path', ns)):
        if path_filter is not None and not path_filter(path):
            continue

        # Parse the path in d:
        d  = path.attrib['d'].replace(',', ' ')

        parts = d.split()
        coordinates = []
        for (x,y),c in svg_to_coordinate_chomper(
            inp=repart(parts), PRECISION=precision):
            coordinates.append([x,y])
        #print(path,d,coordinates)
        if len(coordinates)>0:
            yield np.array(list(coordinates_to_segments( coordinates )))


def coordinates_to_segments(coordinates):
    for (x,y) in coordinates:

        if np.isnan(x):
            current = x,y
            is_down=False
            continue
        else:
            if not is_down:
                # Move the head to the target location, while still being up
                is_down=True
                prev=None

            if (x,y) != prev:
                if  not np.isnan(current[0]):

                    yield [[current[0],current[1]], [x, y]]

                current = x,y
            else:
                # Dont write duplicate coordinates. Waste of space
                pass
            prev = current
