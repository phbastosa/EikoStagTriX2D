import numpy as np

nx = 101
nz = 51

dx = 10.0
dz = 10.0

ns = 2
nr = 91

SPS = np.zeros((ns, 2))
RPS = np.zeros((nr, 2))
XPS = np.zeros((ns, 3))

SPS[:,0] = np.linspace(300, 700, ns)
SPS[:,1] = 0.0 

RPS[:,0] = np.linspace(50, 950, nr)
RPS[:,1] = 0.0 

XPS[:,0] = np.arange(ns)
XPS[:,1] = np.zeros(ns) 
XPS[:,2] = np.zeros(ns) + nr 

np.savetxt("../inputs/geometry/EikoStagTriX2D_SPS.txt", SPS, fmt = "%.2f", delimiter = ",")
np.savetxt("../inputs/geometry/EikoStagTriX2D_RPS.txt", RPS, fmt = "%.2f", delimiter = ",")
np.savetxt("../inputs/geometry/EikoStagTriX2D_XPS.txt", XPS, fmt = "%.0f", delimiter = ",")

vp = np.array([1500,1600,1800,2000,2500])
vs = np.array([   0, 950,1060,1180,1470])
ro = np.array([1000,2350,2400,2450,2500])
z = np.array([200, 50, 100, 100])

E = np.array([0.0, 0.07, 0.08, 0.10, 0.0])
D = np.array([0.0, 0.03, 0.05, 0.06, 0.0])

tilt = np.array([0, 10, 15, 20, 0]) * np.pi/180.0

S = np.zeros((nz, nx))
B = np.zeros((nz, nx))

C11 = np.zeros_like(S)
C13 = np.zeros_like(S)
C15 = np.zeros_like(S)
C33 = np.zeros_like(S)
C35 = np.zeros_like(S)
C55 = np.zeros_like(S)

C = np.zeros((3,3))
M = np.zeros((3,3))

c11 = c13 = c15 = 0
c33 = c35 = 0
c55 = 0

SI = 1e9

for i in range(len(vp)):
    
    layer = int(np.sum(z[:i])/dz)

    c33 = ro[i]*vp[i]**2 / SI
    c55 = ro[i]*vs[i]**2 / SI

    c11 = c33*(1.0 + 2.0*E[i])

    c13 = np.sqrt((c33 - c55)**2 + 2.0*D[i]*c33*(c33 - c55)) - c55

    C[0,0] = c11; C[0,1] = c13; C[0,2] = c15;  
    C[1,0] = c13; C[1,1] = c33; C[1,2] = c35  
    C[2,0] = c15; C[2,1] = c35; C[2,2] = c55; 

    c = np.cos(tilt[i])
    s = np.sin(tilt[i])

    sin2 = np.sin(2.0*tilt[i])
    cos2 = np.cos(2.0*tilt[i])

    M = np.array([[     c**2,     s**2, sin2],
                  [     s**2,     c**2,-sin2],
                  [-0.5*sin2, 0.5*sin2, cos2]])

    Cr = (M @ C @ M.T) * SI

    S[layer:] = 1.0 / vp[i]
    B[layer:] = 1.0 / ro[i]

    C11[layer:] = Cr[0,0]; C13[layer:] = Cr[0,1]; C15[layer:] = Cr[0,2] 
    C33[layer:] = Cr[1,1]; C35[layer:] = Cr[1,2]; C55[layer:] = Cr[2,2]

S.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX2D_slowness.bin")
B.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX2D_buoyancy.bin")

C11.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX2D_C11.bin")
C13.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX2D_C13.bin")
C15.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX2D_C15.bin")
C33.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX2D_C33.bin")
C35.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX2D_C35.bin")
C55.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/EikoStagTriX2D_C55.bin")
