import torch
import torch.nn as nn
import numpy as np

from functional import *


""" SDF: Signed Distance Functions
Re-implementations of SDFs based on the following Inigo Quilez's excellent article.
https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
"""


# ---------------- Primitives ---------------- #

def sphere(r):
    """ GLSL
    float sdSphere( vec3 p, float s )
    {
        return length(p)-s;
    }
    """
    def sdf(p):
        d = length(p) - r
        return d
    return sdf


def box(s):
    """ GLSL
    float sdBox( vec3 p, vec3 b )
    {
        vec3 q = abs(p) - b;
        return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
    }
    """
    def sdf(p):
        q = abs(p) - s
        d = length(relu(q)) - relu(-maximum(*unconcat(q)))
        return d
    return sdf


def torus(r1, r2):
    """ GLSL
    float sdTorus( vec3 p, vec2 t )
    {
        vec2 q = vec2(length(p.xz)-t.x,p.y);
        return length(q)-t.y;
    }
    """
    def sdf(p):
        px, py, pz = unconcat(p)
        q = concat(length(concat(px, pz)) - r1, py)
        d = length(q) - r2
        return d
    return sdf


def link(l, r1, r2):
    """ GLSL
    float sdLink( vec3 p, float le, float r1, float r2 )
    {
        vec3 q = vec3( p.x, max(abs(p.y)-le,0.0), p.z );
        return length(vec2(length(q.xy)-r1,q.z)) - r2;
    }
    """
    def sdf(p):
        px, py, pz = unconcat(p)
        q = concat(px, relu(abs(py) - l), pz)
        qx, qy, qz = unconcat(q)
        d = length(concat(length(concat(qx, qy)) - r1, qz)) - r2
        return d
    return sdf


def cylinder(r, h):
    """ GLSL
    float sdCappedCylinder( vec3 p, float h, float r )
    {
        vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
        return min(max(d.x,d.y),0.0) + length(max(d,0.0));
    }
    """
    def sdf(p):
        px, py, pz = unconcat(p)
        d = abs(concat(length(concat(px, pz)), py)) - concat(r, h)
        d = -relu(-maximum(*unconcat(d))) + length(relu(d))
        return d
    return sdf


def cone(c, h):
    """ GLSL
    float sdCone( in vec3 p, in vec2 c, float h )
    {
        // c is the sin/cos of the angle, h is height
        // Alternatively pass q instead of (c,h),
        // which is the point at the base in 2D
        vec2 q = h*vec2(c.x/c.y,-1.0);
            
        vec2 w = vec2( length(p.xz), p.y );
        vec2 a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );
        vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
        float k = sign( q.y );
        float d = min(dot( a, a ),dot(b, b));
        float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
        return sqrt(d)*sign(s);
    }
    """
    def sdf(p):
        px, py, pz = unconcat(p)
        cx, cy = unconcat(c)
        q = h * concat(cx / cy, -one(cx))
        qx, qy = unconcat(q)
        w = concat(length(concat(px, pz)), py)
        wx, wy = unconcat(w)
        a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0)
        b = w - q * concat(clamp(wx / qx, 0.0, 1.0), one(wx))
        k = sign(qy)
        d = minimum(dot(a, a), dot(b, b))
        s = maximum(k * (wx * qy - wy * qx), k * (wy - qy))
        d = sqrt(d) * sign(s)
        return d
    return sdf


def capsule(r, h):
    """ GLSL
    float sdVerticalCapsule( vec3 p, float h, float r )
    {
        p.y -= clamp( p.y, 0.0, h );
        return length( p ) - r;
    }
    """
    def sdf(p):
        px, py, pz = unconcat(p)
        py = py - clamp(py, 0.0, h)
        d = length(concat(px, py, pz)) - r
        return d
    return sdf


def ellipsoid(r):
    """ GLSL
    float sdbEllipsoidV1( in vec3 p, in vec3 r )
    {
        float k1 = length(p/r);
        return (k1-1.0)*min(min(r.x,r.y),r.z);
    }
    float sdbEllipsoidV2( in vec3 p, in vec3 r )
    {
        float k1 = length(p/r);
        float k2 = length(p/(r*r));
        return k1*(k1-1.0)/k2;
    }
    """
    def sdf(p):

        k1 = length(p / r)
        d = (k1 - 1.0) * minimum(*unconcat(r))
        return d

        # NOTE: The improved version does not work well
        # https://www.iquilezles.org/www/articles/ellipsoids/ellipsoids.htm
        k1 = length(p / r)
        k2 = length(p / (r ** 2))
        d = k1 * (k1 - 1.0) / k2
        return d

    return sdf


def rhombus(s, h):
    """ GLSL
    float sdRhombus(vec3 p, float la, float lb, float h, float ra)
    {
        p = abs(p);
        vec2 b = vec2(la,lb);
        float f = clamp( (ndot(b,b-2.0*p.xz))/dot(b,b), -1.0, 1.0 );
        vec2 q = vec2(length(p.xz-0.5*b*vec2(1.0-f,1.0+f))*sign(p.x*b.y+p.z*b.x-b.x*b.y)-ra, p.y-h);
        return min(max(q.x,q.y),0.0) + length(max(q,0.0));
    }
    """
    def sdf(p):
        p = abs(p)
        px, py, pz = unconcat(p)
        sx, sy = unconcat(s)
        f = clamp(ndot(s, s - 2.0 * concat(px, pz)) / dot(s, s), -1.0, 1.0)
        q = concat(length(concat(px, pz) - 0.5 * s * concat(1.0 - f, 1.0 + f)) * sign(px * sy + pz * sx - sx * sy), py - h)
        return -relu(-maximum(*unconcat(q))) + length(relu(q))
    return sdf


def triprism(s, h):
    """ GLSL
    float sdTriPrism( vec3 p, vec2 h )
    {
        vec3 q = abs(p);
        return max(q.z-h.y,max(q.x*0.866025+p.y*0.5,-p.y)-h.x*0.5);
    }
    """
    def sdf(p):
        q = abs(p)
        px, py, pz = unconcat(p)
        qx, qy, qz = unconcat(q)
        d = maximum(qz - h, maximum(qx * np.cos(np.pi / 6.0) + py * np.sin(np.pi / 6.0), -py) - s * 0.5)
        return d
    return sdf


def hexprism(s, h):
    """ GLSL
    float sdHexPrism( vec3 p, vec2 h )
    {
        const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
        p = abs(p);
        p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
        vec2 d = vec2(
            length(p.xy-vec2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x),
            p.z-h.y );
        return min(max(d.x,d.y),0.0) + length(max(d,0.0));
    }
    """
    def sdf(p):
        k = tensor(p, [-np.cos(np.pi / 6.0), np.sin(np.pi / 6.0), np.tan(np.pi / 6.0)])
        kx, ky, kz = unconcat(k)
        q = abs(p)
        qx, qy, qz = unconcat(q)
        t = 2.0 * relu(-dot(concat(kx, ky), concat(qx, qy)))
        qx = qx + t * kx
        qy = qy + t * ky
        d = concat(length(concat(qx, qy) - concat(clamp(qx, -kz * s, kz * s), expand(s, qx))) * sign(qy - s), qz - h)
        d = -relu(-maximum(*unconcat(d))) + length(relu(d))
        return d
    return sdf


def octahedron(s):
    """GLSL
    float sdOctahedron( vec3 p, float s)
    {
        p = abs(p);
        float m = p.x+p.y+p.z-s;
        vec3 q;
            if( 3.0*p.x < m ) q = p.xyz;
        else if( 3.0*p.y < m ) q = p.yzx;
        else if( 3.0*p.z < m ) q = p.zxy;
        else return m*0.57735027;
            
        float k = clamp(0.5*(q.z-q.y+s),0.0,s); 
        return length(vec3(q.x,q.y-s+k,q.z-k)); 
    }
    """
    def sdf(p):
        p = abs(p)
        px, py, pz = unconcat(p)
        m = px + py + pz - s
        q = torch.where(
            3.0 * px < m,
            concat(px, py, pz),
            torch.where(
                3.0 * py < m,
                concat(py, pz, px),
                concat(pz, px, py),
            ),
        )
        qx, qy, qz = unconcat(q)
        k = clamp(0.5 * (qz - qy + s), 0.0, s); 
        d = length(concat(qx, qy - s + k, qz - k))
        d = torch.where((3.0 * px < m) | (3.0 * py < m) | (3.0 * pz < m), d, m * np.tan(np.pi / 6.0))
        return d
    return sdf


def pyramid(h):
    """ GLSL
    float sdPyramid( vec3 p, float h)
    {
        float m2 = h*h + 0.25;
            
        p.xz = abs(p.xz);
        p.xz = (p.z>p.x) ? p.zx : p.xz;
        p.xz -= 0.5;

        vec3 q = vec3( p.z, h*p.y - 0.5*p.x, h*p.x + 0.5*p.y);
        
        float s = max(-q.x,0.0);
        float t = clamp( (q.y-0.5*p.z)/(m2+0.25), 0.0, 1.0 );
            
        float a = m2*(q.x+s)*(q.x+s) + q.y*q.y;
        float b = m2*(q.x+0.5*t)*(q.x+0.5*t) + (q.y-m2*t)*(q.y-m2*t);
            
        float d2 = min(q.y,-q.x*m2-q.y*0.5) > 0.0 ? 0.0 : min(a,b);
            
        return sqrt( (d2+q.z*q.z)/m2 ) * sign(max(q.z,-p.y));
    }
    """
    def sdf(p):
        m = h ** 2 + 0.25
        px, py, pz = unconcat(p)
        px, pz = abs(px), abs(pz)
        px, pz = maximum(px, pz), minimum(pz, px)
        px, pz = px - 0.5, pz - 0.5
        q = concat(pz, h * py - 0.5 * px, h * px + 0.5 * py)
        qx, qy, qz = unconcat(q)
        s = relu(-qx)
        t = clamp((qy - 0.5 * pz) / (m + 0.25), 0.0, 1.0)
        a = m * (qx + s) ** 2 + qy ** 2
        b = m * (qx + 0.5 * t) ** 2 + (qy - m * t) ** 2
        d = torch.where(minimum(qy, -qx * m - qy * 0.5) > 0.0, zero(a), minimum(a, b))
        d = sqrt((d + qz ** 2) / m) * sign(maximum(qz, -py))
        return d
    return sdf


# ---------------- Geometric Transformations ---------------- #


def translation(sdf, t):
    """ GLSL
    vec3 opTx( in vec3 p, in transform t, in sdf3d primitive )
    {
        return primitive( invert(t)*p );
    }
    """
    def wrapper(p):
        d = sdf(p - t)
        return d
    return wrapper


def rotation(sdf, R):
    """ GLSL
    vec3 opTx( in vec3 p, in transform t, in sdf3d primitive )
    {
        return primitive( invert(t)*p );
    }
    """
    def wrapper(p):
        d = sdf(matmul(transpose(R), p))
        return d
    return wrapper


def scaling(sdf, s):
    """ GLSL
    float opScale( in vec3 p, in float s, in sdf3d primitive )
    {
        return primitive(p/s)*s;
    }
    """
    def wrapper(p):
        d = sdf(p / s) * s
        return d
    return wrapper


def elongation(sdf, s):
    """ GLSL
    float opElongate( in sdf3d primitive, in vec3 p, in vec3 h )
    {
        vec3 q = abs(p)-h;
        return primitive( max(q,0.0) ) + min(max(q.x,max(q.y,q.z)),0.0);
    }
    """
    def wrapper(p):
        q = abs(p) - s
        d = sdf(relu(q)) - relu(-maximum(*unconcat(q)))
        return d
    return wrapper


def rounding(sdf, r):
    """  GLSL
    float opRound( in sdf3d primitive, float rad )
    {
        return primitive(p) - rad
    }
    """
    def wrapper(p):
        d = sdf(p) - r
        return d
    return wrapper


def onion(sdf, t):
    """ GLSL
    float opOnion( in float sdf, in float thickness )
    {
        return abs(sdf)-thickness;
    }
    """
    def wrapper(p):
        d = abs(sdf(p)) - t
        return d
    return wrapper


def infinite_repetition(sdf, c):
    """ GLSL
    float opRep( in vec3 p, in vec3 c, in sdf3d primitive )
    {
        vec3 q = mod(p+0.5*c,c)-0.5*c;
        return primitive( q );
    }
    """
    def wrapper(p):
        q = mod(p + 0.5 * c, c) - 0.5 * c
        d = sdf(q)
        return d
    return wrapper


def finite_repetition(sdf, c, l):
    """ GLSL
    vec3 opRepLim( in vec3 p, in float c, in vec3 l, in sdf3d primitive )
    {
        vec3 q = p-c*clamp(round(p/c),-l,l);
        return primitive( q );
    }
    """
    def wrapper(p):
        q = p - c * clamp(round(p / c), -l, l)
        d = sdf(q)
        return d
    return wrapper


def twist(sdf, k):
    """ GLSL
    float opTwist( in sdf3d primitive, in vec3 p )
    {
        const float k = 10.0; // or some other amount
        float c = cos(k*p.y);
        float s = sin(k*p.y);
        mat2  m = mat2(c,-s,s,c);
        vec3  q = vec3(m*p.xz,p.y);
        return primitive(q);
    }
    """
    def wrapper(p):
        px, py, pz = unconcat(p)
        c = cos(k * py)
        s = sin(k * py)
        m = stack(concat(c, -s), concat(s, c))
        p = concat(px, pz)
        q = concat(matmul(m, p), py)
        d = sdf(q)
        return d
    return wrapper


def bend(sdf, k):
    """ GLSL
    float opCheapBend( in sdf3d primitive, in vec3 p )
    {
        const float k = 10.0; // or some other amount
        float c = cos(k*p.x);
        float s = sin(k*p.x);
        mat2  m = mat2(c,-s,s,c);
        vec3  q = vec3(m*p.xy,p.z);
        return primitive(q);
    }
    """
    def wrapper(p):
        px, py, pz = unconcat(p)
        c = cos(k * px)
        s = sin(k * px)
        m = stack(concat(c, -s), concat(s, c))
        q = concat(matmul(m, concat(px, py)), pz)
        d = sdf(q)
        return d
    return wrapper


def symmetry_x(sdf):
    """ GLSL
    float opSymX( in vec3 p, in sdf3d primitive )
    {
        p.x = abs(p.x);
        return primitive(p);
    }
    """
    def wrapper(p):
        px, py, pz = unconcat(p)
        d = sdf(concat(abs(px), py, pz))
        return d
    return wrapper


def symmetry_y(sdf):
    """ GLSL
    float opSymX( in vec3 p, in sdf3d primitive )
    {
        p.x = abs(p.x);
        return primitive(p);
    }
    """
    def wrapper(p):
        px, py, pz = unconcat(p)
        d = sdf(concat(px, abs(py), pz))
        return d
    return wrapper


def symmetry_z(sdf):
    """ GLSL
    float opSymX( in vec3 p, in sdf3d primitive )
    {
        p.x = abs(p.x);
        return primitive(p);
    }
    """
    def wrapper(p):
        px, py, pz = unconcat(p)
        d = sdf(concat(px, py, abs(pz)))
        return d
    return wrapper
