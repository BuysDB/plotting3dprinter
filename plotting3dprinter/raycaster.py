import numpy as np
import matplotlib.pyplot as plt
from more_itertools import windowed

T = np.array([[0, -1], [1, 0]])

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return

def np_perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def np_cross_product(a, b):
    return np.dot(a, np_perp(b))

def np_seg_intersect(a, b, considerCollinearOverlapAsIntersect = False):
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
    # http://www.codeproject.com/Tips/862988/Find-the-intersection-point-of-two-line-segments
    r = a[1] - a[0]
    s = b[1] - b[0]
    v = b[0] - a[0]
    num = np_cross_product(v, r)
    denom = np_cross_product(r, s)
    # If r x s = 0 and (q - p) x r = 0, then the two lines are collinear.
    if np.isclose(denom, 0) and np.isclose(num, 0):
        # 1. If either  0 <= (q - p) * r <= r * r or 0 <= (p - q) * s <= * s
        # then the two lines are overlapping,
        if(considerCollinearOverlapAsIntersect):
            vDotR = np.dot(v, r)
            aDotS = np.dot(-v, s)
            if (0 <= vDotR  and vDotR <= np.dot(r,r)) or (0 <= aDotS  and aDotS <= np.dot(s,s)):
                return True
        # 2. If neither 0 <= (q - p) * r = r * r nor 0 <= (p - q) * s <= s * s
        # then the two lines are collinear but disjoint.
        # No need to implement this expression, as it follows from the expression above.
        return None
    if np.isclose(denom, 0) and not np.isclose(num, 0):
        # Parallel and non intersecting
        return None
    u = num / denom
    t = np_cross_product(v, s) / denom
    if u >= 0 and u <= 1 and t >= 0 and t <= 1:
        res = b[0] + (s*u)
        return res
    # Otherwise, the two line segments are not parallel but do not intersect.
    return None


def cast_rays(segarr, coords, ray_distance = 2, debug =False):

    # create rays through the complete shape
    minx = coords[:,0].min()
    maxx = coords[:,0].max()
    miny = coords[:,1].min()
    maxy = coords[:,1].max()


    for y_ray in np.arange(miny,maxy,ray_distance):

        rays = np.array([[[ minx ,  y_ray ],
            [ maxx ,  y_ray  ]]])

        if debug:
            plt.plot(rays[:,:,0].flatten(),rays[:,:,1].flatten(),c='r',alpha=0.1)

        intersection_coords = []
        for i in range(segarr.shape[0]):
            intersect = np_seg_intersect(segarr[i,:,:], rays[0,:])
            #if debug:
            #    print(intersect)
            if intersect is None:
                #plt.plot(segarr[i,:,0], segarr[i,:,1],c='grey')
                pass
            else:

                intersection_coords.append(intersect)
                #plt.scatter([intersect[0]],[intersect[1]],c='k')
                #plt.plot(segarr[i,:,0], segarr[i,:,1],c='red')

        if len(intersection_coords)>1:
            # Order the coordinates on the x axis
            intersection_x_coords = np.array(intersection_coords)[:,0]
            intersection_x_coords.sort()

            prev = None
            for i,(s,e) in enumerate( windowed(intersection_x_coords,2) ):

                if i%2==0:
                    #print(y_ray,s,e,prev)
                    if prev is not None and (s,e)==prev: # Overlap
                        continue

                    if debug:
                        plt.plot([s,e],[y_ray,y_ray],c='k',alpha=0.2)
                    yield ([s,y_ray],[e,y_ray])

                    prev = (s,e)
