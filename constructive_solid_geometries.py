import torch
import torch.nn as nn
import numpy as np

from functional import *


# ---------------- CSG (Constructive Solid Geometry) ---------------- #


def union(sdf1, sdf2):
    """ GLSL
    float opUnion( float d1, float d2 ) { return min(d1,d2); }
    """
    def wrapper(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        d = minimum(d1, d2)
        return d
    return wrapper


def subtraction(sdf1, sdf2):
    """ GLSL
    float opSubtraction( float d1, float d2 ) { return max(-d1,d2); }
    """
    def wrapper(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        d = maximum(-d1, d2)
        return d
    return wrapper


def intersection(sdf1, sdf2):
    """ GLSL
    float opIntersection( float d1, float d2 ) { return max(d1,d2); }
    """
    def wrapper(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        d = maximum(d1, d2)
        return d
    return wrapper


def smooth_union(sdf1, sdf2, k):
    """ GLSL
    float opSmoothUnion( float d1, float d2, float k ) 
    {
        float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
        return mix( d2, d1, h ) - k*h*(1.0-h); 
    }
    """
    def wrapper(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
        d = mix(d2, d1, h) - k * h * (1.0 - h)
        return d
    return wrapper


def smooth_subtraction(sdf1, sdf2, k):
    """" GLSL
    float opSmoothSubtraction( float d1, float d2, float k ) 
    {
        float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
        return mix( d2, -d1, h ) + k*h*(1.0-h); 
    }
    """
    def wrapper(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0)
        d = mix(d2, -d1, h) + k * h * (1.0 - h)
        return d
    return wrapper


def smooth_intersection(sdf1, sdf2, k):
    """ GLSL
    float opSmoothIntersection( float d1, float d2, float k ) 
    {
        float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
        return mix( d2, d1, h ) + k*h*(1.0-h); 
    }
    """
    def wrapper(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
        d = mix(d2, d1, h) + k * h * (1.0 - h)
        return d
    return wrapper
