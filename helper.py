import cmath
import numpy as np
from mpmath import iv
import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)
h = 3.25
numer_part = 100

# FriCAS Setup
def Fu_4(x_1,x_2,y_2):
    return F_4(x_1,x_2,y_1[1],y_2) - F_4(x_1,x_2,y_1[0],y_2)
def Fuv_4(x_1,x_2):
    return Fu_4(x_1,x_2,y_2[1]) - Fu_4(x_1,x_2,y_2[0])
def Fxuv_4(x_2):
    return Fuv_4(x_1[1],x_2) - Fuv_4(x_1[0],x_2)
def Fxyuv_4():
    return Fxuv_4(x_2[1]) - Fxuv_4(x_2[0])
def Fu_4out(x_1,x_4,y_4):
    return F_4out(x_1,x_4,y_1[1],y_4) - F_4out(x_1,x_4,y_1[0],y_4)
def Fuv_4out(x_1,x_4):
    return Fu_4out(x_1,x_4,y_4[1]) - Fu_4out(x_1,x_4,y_4[0])
def Fxuv_4out(x_4):
    return Fuv_4out(x_1[1],x_4) - Fuv_4out(x_1[0],x_4)

# Antiderivatives using FriCAS: first/second/third summand
def F4(x_1,x_2,y_1,y_2):
    if x_1 == x_2 and y_1 < y_2:
        return ((((120*h*y_2-120*h*y_1)*x_2**4+(-480*h*y_2+480*h*y_1)*x_1*x_2**3+(720*h*y_2-720*h*y_1)*x_1**2*x_2**2+((-480*h*y_2+480*h*y_1)*x_1**3+(120*h**4*y_2**4-480*h**4*y_1*y_2**3+720*h**4*y_1**2*y_2**2-480*h**4*y_1**3*y_2+120*h**4*y_1**4))*x_2+((120*h*y_2-120*h*y_1)*x_1**4+(-120*h**4*y_2**4+480*h**4*y_1*y_2**3-720*h**4*y_1**2*y_2**2+480*h**4*y_1**3*y_2-120*h**4*y_1**4)*x_1))*cmath.log(abs((x_2**2-2*x_1*x_2+(x_1**2+(h**2*y_2**2-2*h**2*y_1*y_2+h**2*y_1**2)))**(1/2)+(h*y_2-1*h*y_1)))+((-48*x_2**4+192*x_1*x_2**3+(-288*x_1**2+(144*h**2*y_2**2-288*h**2*y_1*y_2+144*h**2*y_1**2))*x_2**2+(192*x_1**3+(-288*h**2*y_2**2+576*h**2*y_1*y_2-288*h**2*y_1**2)*x_1)*x_2+(-48*x_1**4+(144*h**2*y_2**2-288*h**2*y_1*y_2+144*h**2*y_1**2)*x_1**2+(-48*h**4*y_2**4+192*h**4*y_1*y_2**3-288*h**4*y_1**2*y_2**2+192*h**4*y_1**3*y_2-48*h**4*y_1**4)))*(x_2**2-2*x_1*x_2+(x_1**2+(h**2*y_2**2-2*h**2*y_1*y_2+h**2*y_1**2)))**(1/2)+(((-120*h**4*y_2**4+480*h**4*y_1*y_2**3-720*h**4*y_1**2*y_2**2+480*h**4*y_1**3*y_2-120*h**4*y_1**4)*x_2*cmath.log(2)+(3*x_2**5+(-30*h*y_2+30*h*y_1)*x_2**4+((280*h*y_2-280*h*y_1)*x_1+(-40*h**2*y_2**2+80*h**2*y_1*y_2-40*h**2*y_1**2))*x_2**3+(-420*h*y_2+420*h*y_1)*x_1**2*x_2**2+((280*h*y_2-280*h*y_1)*x_1**3+120*h**3*y_2**3*x_1)*x_2))))))/(2880*h**2)
    elif x_1 == x_2 and y_1 > y_2:
        return (((240*h**4*y_2**4-960*h**4*y_1*y_2**3+1440*h**4*y_1**2*y_2**2-960*h**4*y_1**3*y_2+240*h**4*y_1**4)*x_2+(-240*h**4*y_2**4+960*h**4*y_1*y_2**3-1440*h**4*y_1**2*y_2**2+960*h**4*y_1**3*y_2-240*h**4*y_1**4)*x_1)*cmath.log(abs((x_2**2-2*x_1*x_2+(x_1**2+(h**2*y_2**2-2*h**2*y_1*y_2+h**2*y_1**2)))**(1/2)+(x_2+(-1*x_1+(-1*h*y_2+h*y_1)))))+(((-48*x_2**4+192*x_1*x_2**3+(-288*x_1**2+(144*h**2*y_2**2-288*h**2*y_1*y_2+144*h**2*y_1**2))*x_2**2+(192*x_1**3+(-288*h**2*y_2**2+576*h**2*y_1*y_2-288*h**2*y_1**2)*x_1)*x_2+(-48*x_1**4+(144*h**2*y_2**2-288*h**2*y_1*y_2+144*h**2*y_1**2)*x_1**2+(-48*h**4*y_2**4+192*h**4*y_1*y_2**3-288*h**4*y_1**2*y_2**2+192*h**4*y_1**3*y_2-48*h**4*y_1**4)))*(x_2**2-2*x_1*x_2+(x_1**2+(h**2*y_2**2-2*h**2*y_1*y_2+h**2*y_1**2)))**(1/2)+(((-120*h**4*y_2**4+480*h**4*y_1*y_2**3-720*h**4*y_1**2*y_2**2+480*h**4*y_1**3*y_2-120*h**4*y_1**4)*x_2*cmath.log(2)+(3*x_2**5+(-30*h*y_2+30*h*y_1)*x_2**4+((280*h*y_2-280*h*y_1)*x_1+(-40*h**2*y_2**2+80*h**2*y_1*y_2-40*h**2*y_1**2))*x_2**3+(-420*h*y_2+420*h*y_1)*x_1**2*x_2**2+((280*h*y_2-280*h*y_1)*x_1**3+120*h**3*y_2**3*x_1)*x_2))))))/(2880*h**2)
    elif x_1 == x_2 and y_1 == y_2:
        return ((((-48*x_2**4+192*x_1*x_2**3+(-288*x_1**2+(144*h**2*y_2**2-288*h**2*y_1*y_2+144*h**2*y_1**2))*x_2**2+(192*x_1**3+(-288*h**2*y_2**2+576*h**2*y_1*y_2-288*h**2*y_1**2)*x_1)*x_2+(-48*x_1**4+(144*h**2*y_2**2-288*h**2*y_1*y_2+144*h**2*y_1**2)*x_1**2+(-48*h**4*y_2**4+192*h**4*y_1*y_2**3-288*h**4*y_1**2*y_2**2+192*h**4*y_1**3*y_2-48*h**4*y_1**4)))*(x_2**2-2*x_1*x_2+(x_1**2+(h**2*y_2**2-2*h**2*y_1*y_2+h**2*y_1**2)))**(1/2)+(((-120*h**4*y_2**4+480*h**4*y_1*y_2**3-720*h**4*y_1**2*y_2**2+480*h**4*y_1**3*y_2-120*h**4*y_1**4)*x_2*cmath.log(2)+(3*x_2**5+(-30*h*y_2+30*h*y_1)*x_2**4+((280*h*y_2-280*h*y_1)*x_1+(-40*h**2*y_2**2+80*h**2*y_1*y_2-40*h**2*y_1**2))*x_2**3+(-420*h*y_2+420*h*y_1)*x_1**2*x_2**2+((280*h*y_2-280*h*y_1)*x_1**3+120*h**3*y_2**3*x_1)*x_2))))))/(2880*h**2)
    else:
        return (((240*h**4*y_2**4-960*h**4*y_1*y_2**3+1440*h**4*y_1**2*y_2**2-960*h**4*y_1**3*y_2+240*h**4*y_1**4)*x_2+(-240*h**4*y_2**4+960*h**4*y_1*y_2**3-1440*h**4*y_1**2*y_2**2+960*h**4*y_1**3*y_2-240*h**4*y_1**4)*x_1)*cmath.log(abs((x_2**2-2*x_1*x_2+(x_1**2+(h**2*y_2**2-2*h**2*y_1*y_2+h**2*y_1**2)))**(1/2)+(x_2+(-1*x_1+(-1*h*y_2+h*y_1)))))+(((120*h*y_2-120*h*y_1)*x_2**4+(-480*h*y_2+480*h*y_1)*x_1*x_2**3+(720*h*y_2-720*h*y_1)*x_1**2*x_2**2+((-480*h*y_2+480*h*y_1)*x_1**3+(120*h**4*y_2**4-480*h**4*y_1*y_2**3+720*h**4*y_1**2*y_2**2-480*h**4*y_1**3*y_2+120*h**4*y_1**4))*x_2+((120*h*y_2-120*h*y_1)*x_1**4+(-120*h**4*y_2**4+480*h**4*y_1*y_2**3-720*h**4*y_1**2*y_2**2+480*h**4*y_1**3*y_2-120*h**4*y_1**4)*x_1))*cmath.log(abs((x_2**2-2*x_1*x_2+(x_1**2+(h**2*y_2**2-2*h**2*y_1*y_2+h**2*y_1**2)))**(1/2)+(h*y_2-1*h*y_1)))+((-48*x_2**4+192*x_1*x_2**3+(-288*x_1**2+(144*h**2*y_2**2-288*h**2*y_1*y_2+144*h**2*y_1**2))*x_2**2+(192*x_1**3+(-288*h**2*y_2**2+576*h**2*y_1*y_2-288*h**2*y_1**2)*x_1)*x_2+(-48*x_1**4+(144*h**2*y_2**2-288*h**2*y_1*y_2+144*h**2*y_1**2)*x_1**2+(-48*h**4*y_2**4+192*h**4*y_1*y_2**3-288*h**4*y_1**2*y_2**2+192*h**4*y_1**3*y_2-48*h**4*y_1**4)))*(x_2**2-2*x_1*x_2+(x_1**2+(h**2*y_2**2-2*h**2*y_1*y_2+h**2*y_1**2)))**(1/2)+(((-240*h**4*y_2**4+960*h**4*y_1*y_2**3-1440*h**4*y_1**2*y_2**2+960*h**4*y_1**3*y_2-240*h**4*y_1**4)*x_2+(240*h**4*y_2**4-960*h**4*y_1*y_2**3+1440*h**4*y_1**2*y_2**2-960*h**4*y_1**3*y_2+240*h**4*y_1**4)*x_1)*cmath.log(abs(-x_2+x_1))+((-120*h**4*y_2**4+480*h**4*y_1*y_2**3-720*h**4*y_1**2*y_2**2+480*h**4*y_1**3*y_2-120*h**4*y_1**4)*x_2*cmath.log(2)+(3*x_2**5+(-30*h*y_2+30*h*y_1)*x_2**4+((280*h*y_2-280*h*y_1)*x_1+(-40*h**2*y_2**2+80*h**2*y_1*y_2-40*h**2*y_1**2))*x_2**3+(-420*h*y_2+420*h*y_1)*x_1**2*x_2**2+((280*h*y_2-280*h*y_1)*x_1**3+120*h**3*y_2**3*x_1)*x_2))))))/(2880*h**2)

# Last summand (leave dx4)
def F4_x4out(x_1,x_4,y_1,y_4):
    if x_1 == x_4 and y_1 < y_4:
        return (-12*h**4*(y_4-y_1)**4*cmath.log(h*(abs(y_4-y_1)+y_4-y_1))+24*h**4*(y_4-y_1)**4*cmath.log(abs(y_4-y_1))+(-24*x_4**3+72*x_1*x_4**2+(-72*x_1**2+(36*h**2*y_4**2-72*h**2*y_1*y_4+36*h**2*y_1**2))*x_4+(24*x_1**3+(-36*h**2*y_4**2+72*h**2*y_1*y_4-36*h**2*y_1**2)*x_1))*(x_4**2-2*x_1*x_4+(x_1**2+(h**2*y_4**2-2*h**2*y_1*y_4+h**2*y_1**2)))**(1/2) + (12*h**4*y_4**4-96*h**4*y_1*y_4**3+144*h**4*y_1**2*y_4**2-96*h**4*y_1**3*y_4)*cmath.log(abs(-2*h)) + ((48*h**4*y_1*y_4**3-72*h**4*y_1**2*y_4**2+48*h**4*y_1**3*y_4)*cmath.log(2)+(36*h**2*y_1*y_4*x_4**2-8*h**3*y_4**3*x_4+(8*h**3*y_4**3*x_1+(-3*h**4*y_4**4+28*h**4*y_1*y_4**3-42*h**4*y_1**2*y_4**2+28*h**4*y_1**3*y_4)))))/(288*h**2)
    elif x_1 == x_4 and y_1 > y_4:
        return (-12*h**4*(y_4-y_1)**4*cmath.log(h*(abs(y_1-y_4)+y_1-y_4))+24*h**4*(y_4-y_1)**4*cmath.log(abs(y_4-y_1))+(-24*x_4**3+72*x_1*x_4**2+(-72*x_1**2+(36*h**2*y_4**2-72*h**2*y_1*y_4+36*h**2*y_1**2))*x_4+(24*x_1**3+(-36*h**2*y_4**2+72*h**2*y_1*y_4-36*h**2*y_1**2)*x_1))*(x_4**2-2*x_1*x_4+(x_1**2+(h**2*y_4**2-2*h**2*y_1*y_4+h**2*y_1**2)))**(1/2) + (12*h**4*y_4**4-96*h**4*y_1*y_4**3+144*h**4*y_1**2*y_4**2-96*h**4*y_1**3*y_4)*cmath.log(abs(-2*h)) + ((48*h**4*y_1*y_4**3-72*h**4*y_1**2*y_4**2+48*h**4*y_1**3*y_4)*cmath.log(2)+(36*h**2*y_1*y_4*x_4**2-8*h**3*y_4**3*x_4+(8*h**3*y_4**3*x_1+(-3*h**4*y_4**4+28*h**4*y_1*y_4**3-42*h**4*y_1**2*y_4**2+28*h**4*y_1**3*y_4)))))/(288*h**2)
    elif y_1 == y_4:
        return 1/288*(24*h**4*y_4**4*cmath.log(2)-36*h**4*y_4**4*cmath.log(-2*h)+11*h**4*y_4**4+8*h**3*x_1*y_4**3-8*h**3*x_4*y_4**3+36*h**2*x_4**2*y_4**2+24*(x_1**2-2*x_1*x_4+x_4**2)**(1/2)*x_1**3-72*(x_1**2-2*x_1*x_4+x_4**2)**(1/2)*x_1**2*x_4+72*(x_1**2-2*x_1*x_4+x_4**2)**(1/2)*x_1*x_4**2-24*(x_1**2-2*x_1*x_4+x_4**2)**(1/2)*x_4**3)/h**2
    else:
        return (((48*h*y_4-48*h*y_1)*x_4**3+(-144*h*y_4+144*h*y_1)*x_1*x_4**2+(144*h*y_4-144*h*y_1)*x_1**2*x_4+((-48*h*y_4+48*h*y_1)*x_1**3+(12*h**4*y_4**4-48*h**4*y_1*y_4**3+72*h**4*y_1**2*y_4**2-48*h**4*y_1**3*y_4+12*h**4*y_1**4)))*cmath.log(abs((x_4**2-2*x_1*x_4+(x_1**2+(h**2*y_4**2-2*h**2*y_1*y_4+h**2*y_1**2)))**(1/2)+(h*y_4-1*h*y_1)))+((-24*h**4*y_4**4+96*h**4*y_1*y_4**3-144*h**4*y_1**2*y_4**2+96*h**4*y_1**3*y_4-24*h**4*y_1**4)*cmath.log(abs((x_4**2-2*x_1*x_4+(x_1**2+(h**2*y_4**2-2*h**2*y_1*y_4+h**2*y_1**2)))**(1/2)+(-1*x_4+(x_1+(h*y_4-1*h*y_1)))))+((-24*x_4**3+72*x_1*x_4**2+(-72*x_1**2+(36*h**2*y_4**2-72*h**2*y_1*y_4+36*h**2*y_1**2))*x_4+(24*x_1**3+(-36*h**2*y_4**2+72*h**2*y_1*y_4-36*h**2*y_1**2)*x_1))*(x_4**2-2*x_1*x_4+(x_1**2+(h**2*y_4**2-2*h**2*y_1*y_4+h**2*y_1**2)))**(1/2)+((12*h**4*y_4**4-96*h**4*y_1*y_4**3+144*h**4*y_1**2*y_4**2-96*h**4*y_1**3*y_4)*cmath.log(abs(-2*h))+((24*h**4*y_4**4-96*h**4*y_1*y_4**3+144*h**4*y_1**2*y_4**2-96*h**4*y_1**3*y_4+24*h**4*y_1**4)*cmath.log(abs(-1*y_4+y_1))+((48*h**4*y_1*y_4**3-72*h**4*y_1**2*y_4**2+48*h**4*y_1**3*y_4)*cmath.log(2)+(36*h**2*y_1*y_4*x_4**2-8*h**3*y_4**3*x_4+(8*h**3*y_4**3*x_1+(-3*h**4*y_4**4+28*h**4*y_1*y_4**3-42*h**4*y_1**2*y_4**2+28*h**4*y_1**3*y_4)))))))))/(288*h**2)

# Antiderivative to definite integral
def x4(x,y,u,v):
    global x_1, x_2, y_1, y_2, F_4
    x_1 = x; x_2 = y; y_1 = u; y_2 = v; F_4 = F4
    return Fxyuv_4()

def x4_x4out(x,y,u,v):
    global x_1, x_4, y_1, y_4, F_4out
    x_1 = x; x_4 = y; y_1 = u; y_4 = v; F_4out = F4_x4out
    return Fxuv_4out(x_4)

# Derive lower (tangent) and upper (secant) bound of integral
def tangent(x_1,x_4,y_1,y_4):
    step = (x_4[1] - x_4[0])/(numer_part + 1)
    f = lambda x_4: x4_x4out(x_1,x_4,y_1,y_4).real
    x4_array = np.linspace(x_4[0],x_4[1],numer_part + 1); x_left = x4_array[:-1]; x_right = x4_array[1:]
    x4_marray = np.linspace(x_4[0] - step/2,x_4[1] + step/2,numer_part + 2) # Midpoints
    e_marray = np.array(list(map(np.exp, -x4_marray))) # Decreasing
    f_marray = np.array(list(map(f, x4_marray))) # Increasing
    a_array = np.gradient(e_marray, step)[1:-1]; c_array = np.gradient(f_marray, step)[1:-1]
    x4_marray = x4_marray[1:-1]; e_marray = e_marray[1:-1]; f_marray = f_marray[1:-1]
    b_array = e_marray - a_array*x4_marray; d_array = f_marray - c_array*x4_marray
    x1_array = a_array*c_array*x_left**3/3 + (b_array*c_array+a_array*d_array)*x_left**2/2 + b_array*d_array*x_left
    x2_array = a_array*c_array*x_right**3/3 + (b_array*c_array+a_array*d_array)*x_right**2/2 + b_array*d_array*x_right
    underestim = np.sum(x2_array - x1_array)
    return underestim

def secant(x_1,x_4,y_1,y_4):
    step = (x_4[1] - x_4[0])/numer_part
    f = lambda x_4: x4_x4out(x_1,x_4,y_1,y_4).real
    x4_array = np.linspace(x_4[0],x_4[1],numer_part + 1)
    e_array = np.array(list(map(np.exp, -x4_array))) # Decreasing
    f_array = np.array(list(map(f, x4_array))) # Increasing
    x_left = x4_array[:-1]; x_right = x4_array[1:]
    e_left = e_array[:-1]; e_right = e_array[1:]
    f_left = f_array[:-1]; f_right = f_array[1:]
    a_array = (e_right - e_left)/step; b_array = e_left - a_array*x_left
    c_array = (f_right - f_left)/step; d_array = f_left - c_array*x_left
    x1_array = a_array*c_array*x_left**3/3 + (b_array*c_array+a_array*d_array)*x_left**2/2 + b_array*d_array*x_left
    x2_array = a_array*c_array*x_right**3/3 + (b_array*c_array+a_array*d_array)*x_right**2/2 + b_array*d_array*x_right
    overestim = np.sum(x2_array - x1_array)
    return overestim

def x4_tangent(x,y,u,v):
    global x_1; global x_4; global y_1; global y_4
    x_1 = x; x_4 = y; y_1 = u; y_4 = v
    return tangent(x_1,x_4,y_1,y_4)

def x4_secant(x,y,u,v):
    global x_1; global x_4; global y_1; global y_4
    x_1 = x; x_4 = y; y_1 = u; y_4 = v
    return secant(x_1,x_4,y_1,y_4)

# Calculate the expected segment length as a function of the 10-dimensional "box"
def x4_0101(box):
    x_0,x_1,x_2,x_3,x_4,y_0,y_1,y_2,y_3,y_4 = box
    return (y_2[1]-y_2[0])*(y_3[1]-y_3[0])*(y_4[1]-y_4[0])*(x_2[1]-x_2[0])*(x_3[1]-x_3[0])*(np.exp(-x_4[0])-np.exp(-x_4[1]))*x4(x_0,x_1,y_0,y_1)
def x4_0202(box):
    x_0,x_1,x_2,x_3,x_4,y_0,y_1,y_2,y_3,y_4 = box
    return (y_1[1]-y_1[0])*(y_3[1]-y_3[0])*(y_4[1]-y_4[0])*(x_1[1]-x_1[0])*(x_3[1]-x_3[0])*(np.exp(-x_4[0])-np.exp(-x_4[1]))*x4(x_0,x_2,y_0,y_2)
def x4_0303(box):
    x_0,x_1,x_2,x_3,x_4,y_0,y_1,y_2,y_3,y_4 = box
    return (y_1[1]-y_1[0])*(y_2[1]-y_2[0])*(y_4[1]-y_4[0])*(x_1[1]-x_1[0])*(x_2[1]-x_2[0])*(np.exp(-x_4[0])-np.exp(-x_4[1]))*x4(x_0,x_3,y_0,y_3)
def x4_1212(box):
    x_0,x_1,x_2,x_3,x_4,y_0,y_1,y_2,y_3,y_4 = box
    return (y_0[1]-y_0[0])*(y_3[1]-y_3[0])*(y_4[1]-y_4[0])*(x_0[1]-x_0[0])*(x_3[1]-x_3[0])*(np.exp(-x_4[0])-np.exp(-x_4[1]))*x4(x_1,x_2,y_1,y_2)
def x4_1313(box):
    x_0,x_1,x_2,x_3,x_4,y_0,y_1,y_2,y_3,y_4 = box
    return (y_0[1]-y_0[0])*(y_2[1]-y_2[0])*(y_4[1]-y_4[0])*(x_0[1]-x_0[0])*(x_2[1]-x_2[0])*(np.exp(-x_4[0])-np.exp(-x_4[1]))*x4(x_1,x_3,y_1,y_3)
def x4_2323(box):
    x_0,x_1,x_2,x_3,x_4,y_0,y_1,y_2,y_3,y_4 = box
    return (y_0[1]-y_0[0])*(y_1[1]-y_1[0])*(y_4[1]-y_4[0])*(x_0[1]-x_0[0])*(x_1[1]-x_1[0])*(np.exp(-x_4[0])-np.exp(-x_4[1]))*x4(x_2,x_3,y_2,y_3)

def x4_LB_3434(box):
    x_0,x_1,x_2,x_3,x_4,y_0,y_1,y_2,y_3,y_4 = box
    return (y_0[1]-y_0[0])*(y_1[1]-y_1[0])*(y_2[1]-y_2[0])*(x_0[1]-x_0[0])*(x_1[1]-x_1[0])*(x_2[1]-x_2[0])*x4_tangent(x_3,x_4,y_3,y_4)
def x4_UB_1414(box):
    x_0,x_1,x_2,x_3,x_4,y_0,y_1,y_2,y_3,y_4 = box
    return (y_0[1]-y_0[0])*(y_2[1]-y_2[0])*(y_3[1]-y_3[0])*(x_0[1]-x_0[0])*(x_2[1]-x_2[0])*(x_3[1]-x_3[0])*x4_secant(x_1,x_4,y_1,y_4)
def x4_UB_2424(box):
    x_0,x_1,x_2,x_3,x_4,y_0,y_1,y_2,y_3,y_4 = box
    return (y_0[1]-y_0[0])*(y_1[1]-y_1[0])*(y_3[1]-y_3[0])*(x_0[1]-x_0[0])*(x_1[1]-x_1[0])*(x_3[1]-x_3[0])*x4_secant(x_2,x_4,y_2,y_4)
def x4_UB_3434(box):
    x_0,x_1,x_2,x_3,x_4,y_0,y_1,y_2,y_3,y_4 = box
    return (y_0[1]-y_0[0])*(y_1[1]-y_1[0])*(y_2[1]-y_2[0])*(x_0[1]-x_0[0])*(x_1[1]-x_1[0])*(x_2[1]-x_2[0])*x4_secant(x_3,x_4,y_3,y_4)

# Calculate the full path lengths: 0-1-2-3-4, 0-1-3-2-4, 0-2-1-3-4, 0-2-3-1-4, 0-3-1-2-4, 0-3-2-1-4
def calc_01234_LB(box):
    return x4_0101(box) + x4_1212(box) + x4_2323(box) + x4_LB_3434(box)
def calc_01324_UB(box):
    return x4_0101(box) + x4_1313(box) + x4_2323(box) + x4_UB_2424(box)
def calc_02134_UB(box):
    return x4_0202(box) + x4_1212(box) + x4_1313(box) + x4_UB_3434(box)
def calc_02314_UB(box):
    return x4_0202(box) + x4_2323(box) + x4_1313(box) + x4_UB_1414(box)
def calc_03124_UB(box):
    return x4_0303(box) + x4_1313(box) + x4_1212(box) + x4_UB_2424(box)
def calc_03214_UB(box):
    return x4_0303(box) + x4_2323(box) + x4_1212(box) + x4_UB_1414(box)

# Input: 10-dimensional box; Output: net contribution of beta2
def contribution(box, label):
    if label == 0:
        return 0
    else:
        default = iv.mpf(calc_01234_LB(box))
        if label == 1:
            new = iv.mpf(calc_01324_UB(box))
        elif label == 2:
            new = iv.mpf(calc_02134_UB(box))
        elif label == 3:
            new = iv.mpf(calc_02314_UB(box))
        elif label == 4:
            new = iv.mpf(calc_03124_UB(box))
        elif label == 5:
            new = iv.mpf(calc_03214_UB(box))
        return max(0, (default - new).real)

def check_monotone(box_array):
    if (0 <= box_array[0][0] < box_array[0][1] <= box_array[1][0] < box_array[1][1] <= box_array[2][0] < box_array[2][1] <= box_array[3][0] < box_array[3][1] <= box_array[4][0] < box_array[4][1]):
        return True
    else:
        return False

# Decision Tree
def necessary_condition(box_array):
    if box_array[0][0] > box_array[3][0] > 0:
        box_array[1][0], box_array[2][0], box_array[3][0] = max(box_array[1][0], box_array[0][0]), max(box_array[2][0], box_array[0][0]), box_array[0][0]
    elif box_array[0][0] > box_array[2][0] > 0:
        box_array[1][0], box_array[2][0] = max(box_array[1][0], box_array[0][0]), box_array[0][0]
    elif box_array[0][0] > box_array[1][0] > 0:
        box_array[1][0] = box_array[0][0]
    if box_array[1][0] > box_array[4][0] > 0:
        box_array[2][0], box_array[3][0], box_array[4][0] = max(box_array[2][0], box_array[1][0]), max(box_array[3][0], box_array[1][0]), box_array[1][0]
    elif box_array[1][0] > box_array[3][0] > 0:
        box_array[2][0], box_array[3][0] = max(box_array[2][0], box_array[1][0]), box_array[1][0]
    elif box_array[1][0] > box_array[2][0] > 0:
        box_array[2][0] = box_array[1][0]
    if box_array[2][0] > box_array[4][0] > 0:
        box_array[3][0], box_array[4][0] = max(box_array[3][0], box_array[2][0]), box_array[2][0]
    elif box_array[2][0] > box_array[3][0] > 0:
        box_array[3][0] = box_array[2][0]
    if box_array[3][0] > box_array[4][0] > 0:
        box_array[4][0] = box_array[3][0]
    if box_array[4][1] == 0:
        box_array[4][1] = max(box_array[4][0], box_array[3][1], box_array[2][1], box_array[1][1], box_array[0][1]) + 1
    else:
        box_array[4][1] = min(box_array[4][1], max(box_array[4][0], box_array[3][1], box_array[2][1], box_array[1][1], box_array[0][1]) + 1)
    if box_array[0][1] > box_array[3][1] > 0:
        box_array[0][1], box_array[1][1], box_array[2][1] = box_array[3][1], min(box_array[1][1], box_array[3][1]), min(box_array[2][1], box_array[3][1])
    elif box_array[1][1] > box_array[3][1] > 0:
        box_array[1][1], box_array[2][1] = box_array[3][1], min(box_array[2][1], box_array[3][1])
    elif box_array[2][1] > box_array[3][1] > 0:
        box_array[2][1] = box_array[3][1]
    if box_array[0][1] > box_array[2][1] > 0:
        box_array[0][1], box_array[1][1] = box_array[2][1], min(box_array[1][1], box_array[2][1])
    elif box_array[1][1] > box_array[2][1] > 0:
        box_array[1][1] = box_array[2][1]
    elif box_array[0][1] > box_array[1][1] > 0:
        box_array[0][1] = box_array[1][1]
    return box_array

def prune(box_array):
    for i in range(4):
        if (box_array[i+1][0] == 0):
            if (box_array[i][1] == 0):
                for j in range(i+1, min(i+3,5)):
                    if box_array[j][1] != 0:
                        for k in range(i, j):
                            level = box_array[i][0] + (k-i+1) * (box_array[j][1] - box_array[i][0])/(j-i+1)
                            if box_array[k][1] == 0:
                                box_array[k][1] = level
                            if box_array[k+1][0] == 0:
                                box_array[k+1][0] = level
                        break
            else:
                box_array[i+1][0] = box_array[i][0]
    for i in reversed(range(4)):
        if (box_array[i+1][0] != 0) and (box_array[i][1] == 0):
            box_array[i][1] = (box_array[i+1][1] + box_array[i+1][0])/2
    for i in range(4):
        if box_array[i+1][0] < box_array[i][1]:
            if (i <= 1) and (box_array[i][1] == box_array[i+1][1] == box_array[i+2][1] == box_array[i+3][1]):
                start, delta = box_array[i][0], (box_array[i+3][1] - box_array[i][0])/4
                if (box_array[i+1][0] < start + delta) and (box_array[i+2][0] < start + 2 * delta) and (box_array[i+3][0] < start + 3 * delta):
                    box_array[i+1][0], box_array[i+2][0], box_array[i+3][0] = start + delta, start + 2 * delta, start + 3 * delta
                    box_array[i][1], box_array[i+1][1], box_array[i+2][1] = start + delta, start + 2 * delta, start + 3 * delta
            elif (i <= 2) and (box_array[i][1] == box_array[i+1][1] == box_array[i+2][1]):
                start, delta = box_array[i][0], (box_array[i+2][1] - box_array[i][0])/3
                if (box_array[i+1][0] < start + delta) and (box_array[i+2][0] < start + 2 * delta):
                    box_array[i+1][0], box_array[i+2][0] = start + delta, start + 2 * delta
                    box_array[i][1], box_array[i+1][1] = start + delta, start + 2 * delta
            outer_avg, inner_avg = (box_array[i][0] + box_array[i+1][1])/2, (box_array[i][1] + box_array[i+1][0])/2
            if box_array[i][1] >= outer_avg >= box_array[i+1][0]:
                box_array[i][1], box_array[i+1][0] = outer_avg, outer_avg
            elif (inner_avg <= box_array[i][0]) or (outer_avg > box_array[i][1]):
                box_array[i+1][0] = box_array[i][1]
            elif (inner_avg >= box_array[i+1][1]) or (outer_avg < box_array[i+1][0]):
                box_array[i][1] = box_array[i+1][0]
    return box_array

def leaf2box(leaf):
    ancestor = leaf.ancestor
    box_array = np.zeros((10, 2))
    for i in range(5, 10):
        box_array[i][1] = 1
    for const in ancestor:
        if const[2] == 'L':
            cur = box_array[const[0]][1]
            box_array[const[0]][1] = const[1] if cur == 0 else min(cur, const[1])
        elif const[2] == 'R':
            cur = box_array[const[0]][0]
            box_array[const[0]][0] = const[1] if cur == 0 else max(cur, const[1])
    # Necessary conditions
    box_array = necessary_condition(box_array)
    # Prune the box to ensure validity
    box_array = prune(box_array)
    return box_array

def leaf_contribution(leaf):
    box_array = leaf2box(leaf)
    if check_monotone(box_array):
        try:
            saving = contribution(box_array, leaf.label) / (4 * np.sqrt(h))
            return saving
        except ValueError:
            return 0
    else:
        return 0
