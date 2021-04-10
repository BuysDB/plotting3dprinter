from more_itertools import windowed
import matplotlib.pyplot as plt
import numpy as np


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

def svg_to_coordinate_chomper( inp, yield_control=False,PRECISION=5):
    prev = None
    try:
        while True:
            chunk = next(inp)
            if chunk == 'M':
                #print('Got new start coordinate')

                start = parse_coord(inp)
                prev = start
                yield np.nan,np.nan
                #yield list(start)
                continue

            if chunk == 'm':
                #print('Got new start coordinate')

                #m = next(inp)
                if prev is None:
                    start = parse_coord(inp)
                else:
                    start = parse_coord(inp) +prev
                prev = start
                #yield list(start)
                yield np.nan,np.nan
                continue



            #print(chunk)
            if chunk in 'zZ':
                # Go to start:
                #print("Returning to start coordinate")
                prev= start
                yield start
                #print("Done")
                continue


            if chunk.strip()=='l':
                # Line to command:
                yield prev
                cur = parse_coord(inp)+prev
                yield cur
                prev = cur

                continue

            if chunk.strip()=='L':
                # Line to command:
                yield prev
                cur = parse_coord(inp)
                yield cur
                prev = cur

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
                        yield np.array([x,y])


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
                        yield np.array([x,y])

                prev = dxdy
                #yield prev
                continue

            if chunk[0]=='A': # ARC mode:
                print('Got arc')
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

                print('Q Bezier',prev,dxdy1, dxdy)

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
                        yield np.array([x,y])

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
                        yield np.array([x,y])

                prev = dxdy
                #yield prev
                continue

            if chunk not in 'HhVv':
                raise ValueError(f'Unknown command {chunk}')
                print("MISS:",chunk)
                prev +=  parse_coord(chunk)
                #print(prev)
                yield list(prev) #prev #+start
            elif chunk=='h':
                # Parse the next chunk: (single horizontal coordinate)

                yield list(prev)

                chunk2 = next(inp)
                x = float(chunk2)
                prev[0] += x

                yield list(prev) #+start
            elif chunk=='H':
                # Parse the next chunk: (single horizontal coordinate)

                yield list(prev)

                chunk2 = next(inp)
                x = float(chunk2)
                prev[0] = x

                yield list(prev) #+start

            elif chunk=='v':
                # Parse the next chunk: (single horizontal coordinate)
                yield list(prev)

                chunk2 = next(inp)
                y = float(chunk2)
                prev[1] += y

                yield list(prev) #+start

            elif chunk=='V':
                # Parse the next chunk: (single horizontal coordinate)
                yield list(prev)

                chunk2 = next(inp)
                y = float(chunk2)
                prev[1] = y

                yield list(prev) #+start
    except StopIteration:
        pass

def repart(inp):
    for p in inp:
        if len(p)>1 and p[0].upper() in 'HZMCQ':
            yield p[0]
            yield p[1:]
        else:
            yield p
